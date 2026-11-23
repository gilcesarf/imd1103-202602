#!/usr/bin/env python3
"""E05 — Analise comparativa PG (results/e05) vs A2C (results/e04) no Pong.

Nao treina modelos nem duplica os dados brutos do E04. Usa somente execucoes
completas, com a mesma semente, orcamento, protocolo e sementes de avaliacao.

Exemplo (executar apos as 3 sementes do E04 e E05):
  python experimentos-unidade-3/analise_e05.py --steps 5000000 \\
    --seeds 42 123 2026 --pg-num-envs 32 --a2c-num-envs 50 \\
    --eval-episodes 10 --no-stop-on-convergence

Dados: results/e05/completo/<modo>/ e results/e04/completo/<modo>/.
Saida: analises/e05/steps_<steps>/<modo>/.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UNIT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = UNIT_DIR / "analises" / "e05"
THRESHOLD = 18.0
COLORS = {"PG": "tab:orange", "A2C": "tab:blue"}


@dataclass
class Run:
    algorithm: str
    seed: int
    directory: Path
    config: dict
    summary: dict
    episodes: list[dict]
    evaluation: list[dict]
    checkpoints: list[dict]
    checkpoint_eval: list[dict]
    updates: list[dict]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if not fields:
            return
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def num(value, fallback=float("nan")):
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except (ValueError, TypeError):
        return fallback


def find_runs(args, algorithm):
    is_pg = algorithm == "PG"
    expected_experiment = "E05" if is_pg else "E04"
    expected_id = "pg" if is_pg else "a2c"
    requested_envs = args.pg_num_envs if is_pg else args.a2c_num_envs
    mode = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    location = UNIT_DIR / "results" / expected_experiment.lower() / "completo" / mode
    if not location.is_dir():
        return {}
    selected = {}
    for folder in sorted(p for p in location.iterdir() if p.is_dir()):
        cf = folder / "config.json"
        sf = folder / "summary.json"
        if not cf.is_file() or not sf.is_file():
            continue
        conf = read_json(cf)
        if conf.get("experiment") != expected_experiment or conf.get("algorithm") != expected_id:
            continue
        if (int(conf.get("max_training_steps", -1)) != args.steps
            or bool(conf.get("stop_on_convergence", True)) != args.stop_on_convergence
            or int(conf.get("evaluation_episodes", -1)) != args.eval_episodes
            or int(conf.get("num_envs", -1)) != requested_envs):
            continue
        seed = int(conf.get("seed", -1))
        if seed not in args.seeds:
            continue
        summary = read_json(sf)
        if (not args.stop_on_convergence and
                int(summary.get("training_steps", -1)) < args.steps):
            continue  # no modo sem parada, exigir orcamento completo
        required = ["episodes.csv", "evaluation.csv", "checkpoints.csv",
                    "checkpoint_evaluations.csv", "updates.csv"]
        missing = [file for file in required if not (folder / file).is_file()]
        if missing:
            raise FileNotFoundError(f"Faltam arquivos em {folder}: {missing}")
        if seed in selected:
            raise ValueError(f"Execucoes duplicadas para {algorithm} seed {seed}: "
                             f"{selected[seed].directory} e {folder}")
        selected[seed] = Run(
            algorithm, seed, folder, conf, summary,
            read_csv(folder / "episodes.csv"),
            read_csv(folder / "evaluation.csv"),
            read_csv(folder / "checkpoints.csv"),
            read_csv(folder / "checkpoint_evaluations.csv"),
            read_csv(folder / "updates.csv"),
        )
    return selected


def checkpoint_series(run):
    rows = [r for r in run.checkpoints if r["checkpoint_type"] in {"periodic", "final"}]
    rows.sort(key=lambda r: num(r["training_steps"]))
    return rows


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning(runs, out):
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for run in runs:
        xs = [num(r["steps"]) / 1e6 for r in run.episodes]
        ys = [num(r["mean_reward_100"]) for r in run.episodes]
        if xs:
            ax.plot(xs, ys, label=f"{run.algorithm} seed {run.seed}",
                    color=COLORS[run.algorithm],
                    linestyle="-" if run.algorithm == "A2C" else "--",
                    linewidth=1.35, alpha=0.65 if len(runs) >= 4 else 0.9)
    ax.axhline(THRESHOLD, color="black", linestyle=":", label="Convergencia > +18")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(xlabel="Interacoes com o ambiente (milhoes)",
           ylabel="Media movel dos ultimos 100 jogos",
           title="E05 — Aprendizado: PG vs A2C no Pong")
    ax.grid(alpha=.22)
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, out)


def plot_checkpoints(runs, out):
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for run in runs:
        rows = checkpoint_series(run)
        if not rows:
            continue
        ax.plot([num(r["training_steps"]) / 1e6 for r in rows],
                [num(r["evaluation_reward_mean"]) for r in rows],
                marker="o", markersize=3, linewidth=1.3,
                color=COLORS[run.algorithm],
                linestyle="-" if run.algorithm == "A2C" else "--",
                label=f"{run.algorithm} seed {run.seed}", alpha=.85)
    ax.axhline(THRESHOLD, color="black", linestyle=":", label="Referencia +18")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(title="E05 — Recompensa em avaliacoes independentes (10 jogos)",
           xlabel="Interacoes no checkpoint (milhoes)",
           ylabel="Recompensa media por jogo")
    ax.grid(alpha=.22)
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, out)


def plot_final_distribution(runs, out):
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    run_order = sorted(runs, key=lambda run: (run.seed, run.algorithm != "PG"))
    rewards = [[num(e["reward"]) for e in run.evaluation] for run in run_order]
    positions = np.arange(1, len(run_order) + 1)
    boxes = ax.boxplot(rewards, positions=positions, patch_artist=True,
                       showmeans=True, meanline=True,
                       meanprops={"color": "green", "linestyle": "--"},
                       medianprops={"color": "darkorange"})
    for patch, run in zip(boxes["boxes"], run_order):
        patch.set_facecolor(COLORS[run.algorithm])
        patch.set_alpha(.3)
    ax.set_xticks(positions, [f"{run.algorithm}\n{run.seed}" for run in run_order])
    ax.axhline(THRESHOLD, color="black", linestyle=":", label="Referencia +18")
    ax.plot([], [], color="green", linestyle="--", label="Media")
    ax.plot([], [], color="darkorange", linestyle="-", label="Mediana")
    ax.set(title="E05 — Recompensas da avaliacao final por semente",
           ylabel="Recompensa por jogo (de -21 a +21)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    save_figure(fig, out)


def plot_best_and_final(runs, out):
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    ordered = sorted(runs, key=lambda run: (run.seed, run.algorithm != "PG"))
    xx = np.arange(len(ordered))
    best = [max((num(c["evaluation_reward_mean"]) for c in checkpoint_series(r)),
                default=float("nan")) for r in ordered]
    last = [num(r.summary["evaluation"]["reward_mean"]) for r in ordered]
    w = .36
    ax.bar(xx - w / 2, best, w, label="Melhor checkpoint observado")
    ax.bar(xx + w / 2, last, w, label="Checkpoint final")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_xticks(xx, [f"{r.algorithm}\n{r.seed}" for r in ordered])
    ax.set(title="E05 — Melhor avaliacao observada vs avaliacao final",
           ylabel="Recompensa media nos 10 jogos de avaliacao")
    ax.legend()
    ax.grid(alpha=.2, axis="y")
    save_figure(fig, out)


def plot_paired_final(runs, out):
    grouped = {(r.algorithm, r.seed): r for r in runs}
    seeds = sorted({r.seed for r in runs})
    pg = [num(grouped[("PG", s)].summary["evaluation"]["reward_mean"]) for s in seeds]
    a2c = [num(grouped[("A2C", s)].summary["evaluation"]["reward_mean"]) for s in seeds]
    xx = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(xx - .2, pg, .4, color=COLORS["PG"], label="PG")
    ax.bar(xx + .2, a2c, .4, color=COLORS["A2C"], label="A2C")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_xticks(xx, [str(s) for s in seeds])
    ax.set(title="E05 — Comparacao pareada da avaliacao final",
           xlabel="Semente de treinamento", ylabel="Recompensa media final")
    ax.legend()
    ax.grid(alpha=.2, axis="y")
    save_figure(fig, out)


def plot_resources(runs, out, field, label, filename):
    ordered = sorted(runs, key=lambda r: (r.seed, r.algorithm != "PG"))
    values = [num(r.summary[field]) for r in ordered]
    if field == "training_elapsed_seconds_excluding_evaluations":
        values = [v / 60 for v in values]
    fig, ax = plt.subplots(figsize=(11.5, 5))
    ax.bar(np.arange(len(ordered)), values,
           color=[COLORS[r.algorithm] for r in ordered], alpha=.82)
    ax.set_xticks(np.arange(len(ordered)),
                  [f"{r.algorithm}\n{r.seed}" for r in ordered])
    ax.set(title=f"E05 — {label}", ylabel=label)
    ax.grid(alpha=.2, axis="y")
    save_figure(fig, out / filename)


def plot_entropy(runs, out, smooth=250):
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for run in runs:
        xs, ys = [], []
        for row in run.updates:
            x, y = num(row["steps"]), num(row.get("entropy"))
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x / 1e6)
                ys.append(y)
        if not xs:
            continue
        raw = np.asarray(ys)
        w = min(smooth, len(raw))
        av = np.convolve(raw, np.ones(w), mode="same") / np.convolve(
            np.ones_like(raw), np.ones(w), mode="same")
        ax.plot(xs[::25], av[::25], label=f"{run.algorithm} seed {run.seed}",
                color=COLORS[run.algorithm],
                linestyle="-" if run.algorithm == "A2C" else "--",
                alpha=.75, linewidth=1.15)
    ax.set(title="E05 — Entropia da politica durante o treino (media suavizada)",
           xlabel="Interacoes (milhoes)", ylabel="Entropia (nats)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, out)


def run_summary(run):
    checkpoints = checkpoint_series(run)
    best = max(checkpoints, key=lambda c: num(c["evaluation_reward_mean"])) if checkpoints else None
    first_zero = next((row for row in run.episodes
                       if int(row["episode"]) >= 100 and num(row["mean_reward_100"]) >= 0), None)
    conv = run.summary.get("convergence")
    return {
        "algorithm": run.algorithm,
        "seed": run.seed,
        "training_steps": run.summary["training_steps"],
        "gradient_updates": run.summary["gradient_updates"],
        "episodes_completed": run.summary["episodes_completed"],
        "converged": bool(run.summary["converged"]),
        "first_convergence_steps": conv["steps"] if conv else "",
        "first_training_mean_nonnegative_steps": first_zero["steps"] if first_zero else "",
        "training_mean_last100": run.summary.get("training_reward_mean_last_100"),
        "final_evaluation_reward_mean": run.summary["evaluation"]["reward_mean"],
        "final_evaluation_reward_std": run.summary["evaluation"]["reward_std"],
        "best_checkpoint_steps": best["training_steps"] if best else "",
        "best_checkpoint_evaluation_mean": best["evaluation_reward_mean"] if best else "",
        "evaluation_delta_final_minus_best": (
            num(run.summary["evaluation"]["reward_mean"]) -
            num(best["evaluation_reward_mean"])) if best else "",
        "training_elapsed_seconds_excluding_evaluations": run.summary[
            "training_elapsed_seconds_excluding_evaluations"],
        "wrapped_interactions_per_training_second": run.summary[
            "wrapped_interactions_per_training_second"],
        "num_envs": run.config["num_envs"],
        "device": run.config["device"],
        "source_directory": str(run.directory),
    }


def analyze(args):
    pg = find_runs(args, "PG")
    a2c = find_runs(args, "A2C")
    paired = sorted(set(pg) & set(a2c))
    if not paired:
        raise ValueError("Nao existem sementes pareadas completas PG(E05) x A2C(E04). "
                         "Conclua os treinamentos das duas variantes antes da analise.")
    missing = {name: sorted(set(args.seeds) - set(runs))
               for name, runs in (("PG", pg), ("A2C", a2c))}
    if any(missing.values()):
        print(f"ATENCAO: sementes incompletas, analise apenas das pareadas {paired}. "
              f"Faltantes: {missing}")
        if args.require_all:
            raise ValueError("--require-all exige todos os pares solicitados.")

    runs = [run for seed in paired for run in (pg[seed], a2c[seed])]
    if any(run.config.get("evaluation_seed_base") != 404000 for run in runs):
        raise ValueError("As sementes de avaliacao nao estao padronizadas em 404000")
    if len({int(run.config.get("evaluation_max_episode_steps", -1)) for run in runs}) != 1:
        raise ValueError("Limite de passos das avaliacoes diferente entre E04 e E05")
    mode = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    out = OUTPUT_BASE / f"steps_{args.steps}" / mode
    out.mkdir(parents=True, exist_ok=True)

    rows = [run_summary(run) for run in runs]
    write_csv(out / "resumo_execucoes.csv", rows)
    write_csv(out / "episodios_treinamento.csv", [
        {"algorithm": run.algorithm, "training_seed": run.seed, **row}
        for run in runs for row in run.episodes])
    write_csv(out / "resumo_checkpoints.csv", [
        {"algorithm": run.algorithm, "training_seed": run.seed, **row}
        for run in runs for row in run.checkpoints])
    write_csv(out / "avaliacoes_checkpoints.csv", [
        {"algorithm": run.algorithm, "training_seed": run.seed, **row}
        for run in runs for row in run.checkpoint_eval])
    write_csv(out / "episodios_avaliacao.csv", [
        {"algorithm": run.algorithm, "training_seed": run.seed, **row}
        for run in runs for row in run.evaluation])

    algorithm_summaries = []
    for alg in ("PG", "A2C"):
        sub = [row for row in rows if row["algorithm"] == alg]
        final_means = [num(r["final_evaluation_reward_mean"]) for r in sub]
        algorithm_summaries.append({
            "algorithm": alg, "n_seeds": len(sub),
            "convergences": sum(bool(r["converged"]) for r in sub),
            "mean_final_evaluation_reward_by_seed": mean(final_means),
            "std_final_evaluation_reward_between_seeds": (
                stdev(final_means) if len(final_means) > 1 else None),
            "mean_throughput_wrapped_steps_s": mean([
                num(r["wrapped_interactions_per_training_second"]) for r in sub]),
            "mean_training_seconds_excluding_evaluation": mean([
                num(r["training_elapsed_seconds_excluding_evaluations"]) for r in sub]),
        })
    write_csv(out / "resumo_algoritmos.csv", algorithm_summaries)

    plot_learning(runs, out / "01_curvas_aprendizado.png")
    plot_checkpoints(runs, out / "02_avaliacoes_por_checkpoint.png")
    plot_final_distribution(runs, out / "03_distribuicao_avaliacao.png")
    plot_best_and_final(runs, out / "04_melhor_checkpoint_vs_final.png")
    plot_paired_final(runs, out / "05_comparacao_final_por_semente.png")
    plot_resources(runs, out, "wrapped_interactions_per_training_second",
                   "Interacoes/s (excluindo avaliacoes)", "06_throughput.png")
    plot_resources(runs, out, "training_elapsed_seconds_excluding_evaluations",
                   "Tempo de treinamento (minutos)", "07_tempo_treinamento.png")
    plot_entropy(runs, out / "08_entropia_politica.png", args.smooth_updates)

    result = {
        "experiment": "E05", "environment": "PongNoFrameskip-v4",
        "criteria": {
            "max_training_steps": args.steps,
            "stop_on_convergence": args.stop_on_convergence,
            "eval_episodes": args.eval_episodes,
            "seeds_requested": args.seeds,
            "paired_seeds_analyzed": paired,
            "missing_seeds_by_method": missing,
            "pg_num_envs": args.pg_num_envs,
            "a2c_num_envs": args.a2c_num_envs,
        },
        "runs": rows,
        "algorithm_summaries": algorithm_summaries,
        "notes": [
            "PG E05 usa Chapter11/05_pong_pg.py; A2C reutiliza E04 Chapter12/02_pong_a2c.py.",
            "Comparacao das configuracoes completas, nao uma ablacao isolada do critico.",
            "PG original: 32 envs/lista PTAN, n-step=9, baseline media movel; A2C: 50 SyncVectorEnv, n-step=4, critico V(s).",
            "Orcamento=interacoes de env.step com wrappers, nao frames brutos ALE.",
            "Politicas estocasticas, mesmos seeds e limite de passos na avaliacao.",
            "Resultados de checkpoint tem apenas 10 jogos; melhor checkpoint e uma selecao com incerteza.",
            "Tempo e throughput podem sofrer interferencia de outras cargas no mesmo Mac.",
            "Gradientes nao foram comparados: PG mede pos-clipping e A2C pre-clipping.",
        ],
    }
    (out / "resumo_analise.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    for record in algorithm_summaries:
        print(f"E05 {record['algorithm']} | n={record['n_seeds']} | "
              f"convergencias={record['convergences']} | "
              f"avaliacao media final={record['mean_final_evaluation_reward_by_seed']:.2f}")
    print(f"E05 | sementes pareadas={paired} | resultados={out}")


def main():
    p = argparse.ArgumentParser(description="E05 — PG vs A2C no Pong")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--pg-num-envs", type=int, default=32)
    p.add_argument("--a2c-num-envs", type=int, default=50)
    p.add_argument("--smooth-updates", type=int, default=250)
    p.add_argument("--require-all", action="store_true",
                   help="Falha caso falte uma das sementes de algum metodo")
    flags = p.add_mutually_exclusive_group()
    flags.add_argument("--stop-on-convergence", action="store_true",
                       dest="stop_on_convergence")
    flags.add_argument("--no-stop-on-convergence", action="store_false",
                       dest="stop_on_convergence")
    p.set_defaults(stop_on_convergence=True)
    args = p.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

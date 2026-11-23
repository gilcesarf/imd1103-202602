#!/usr/bin/env python3
"""E07 — Analise A3C-grad Pong (Chapter12/03_a3c_grad.py) com referencia A2C E04.

Reutiliza CSVs do E04 sem modifica-los. Requer resultados completos por semente
para a comparacao pareada; `--require-all` exige todas as sementes solicitadas.
O E04 usou MPS e E07 usa CPU compartilhada; vazao nao isola efeito do algoritmo.
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

ROOT = Path(__file__).resolve().parent
THRESHOLD = 18.0
COLORS = {"A2C": "tab:blue", "A3C-grad": "tab:orange"}


@dataclass
class Run:
    name: str
    seed: int
    directory: Path
    config: dict
    summary: dict
    episodes: list[dict]
    evaluation: list[dict]
    checkpoints: list[dict]
    updates: list[dict]
    workers: list[dict]


def load_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def store_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def number(v, default=float("nan")):
    try:
        return float(v) if v not in ("", None) else default
    except (ValueError, TypeError):
        return default


def collect(args, algorithm, protocol):
    is_a3c = algorithm == "A3C-grad"
    exp = "e07" if is_a3c else "e04"
    expected = "a3c_grad" if is_a3c else "a2c"
    folder = ROOT / "results" / exp / "completo" / protocol
    selected = {}
    if not folder.is_dir():
        return selected
    for directory in sorted(p for p in folder.iterdir() if p.is_dir()):
        cp, sp = directory / "config.json", directory / "summary.json"
        if not cp.is_file() or not sp.is_file():
            continue
        config = load_json(cp)
        if config.get("experiment") != exp.upper() or config.get("algorithm") != expected:
            continue
        if (int(config.get("max_training_steps", -1)) != args.steps or
            bool(config.get("stop_on_convergence", True)) != args.stop_on_convergence or
            int(config.get("evaluation_episodes", -1)) != args.eval_episodes):
            continue
        if (is_a3c and (int(config.get("num_processes", -1)) != args.processes or
                        int(config.get("num_envs_per_process", -1)) != args.num_envs)):
            continue
        if not is_a3c and int(config.get("num_envs", -1)) != args.a2c_num_envs:
            continue
        seed = int(config.get("seed", -1))
        if seed not in args.seeds:
            continue
        summary = load_json(sp)
        if not args.stop_on_convergence and int(summary.get("training_steps", 0)) < args.steps:
            continue
        required = ["episodes.csv", "evaluation.csv", "checkpoints.csv", "updates.csv"]
        if is_a3c:
            required.append("worker_stats.csv")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Arquivos ausentes em {directory}: {missing}")
        if seed in selected:
            raise ValueError(f"Execucoes duplicadas: {algorithm} seed={seed}")
        selected[seed] = Run(
            algorithm, seed, directory, config, summary,
            load_csv(directory / "episodes.csv"),
            load_csv(directory / "evaluation.csv"),
            load_csv(directory / "checkpoints.csv"),
            load_csv(directory / "updates.csv"),
            load_csv(directory / "worker_stats.csv") if is_a3c else [],
        )
    return selected


def checkpoints(run: Run):
    return sorted((c for c in run.checkpoints
                   if c["checkpoint_type"] in ("periodic", "final")),
                  key=lambda c: number(c["training_steps"]))


def first_zero(run: Run):
    for r in run.episodes:
        if int(r["episode"]) >= 100 and number(r["mean_reward_100"]) >= 0:
            return int(float(r["steps"]))
    return None


def summary_row(run: Run):
    s = run.summary
    if run.name == "A2C":
        convergence = s.get("convergence")
        first_conv = convergence.get("steps") if isinstance(convergence, dict) else None
        last100 = s.get("training_reward_mean_last_100")
        final_mean = s.get("evaluation", {}).get("reward_mean")
        final_std = s.get("evaluation", {}).get("reward_std")
        workers = 1
    else:
        first_conv = s.get("first_convergence_steps")
        last100 = s.get("train_mean_reward_last100")
        final_mean = s.get("final_evaluation_reward_mean")
        final_std = s.get("final_evaluation_reward_std")
        workers = run.config["num_processes"]
    cpts = checkpoints(run)
    best = max(cpts, key=lambda c: number(c["evaluation_reward_mean"])) if cpts else None
    elapsed = number(s.get("training_elapsed_seconds_excluding_evaluations"))
    steps = int(s["training_steps"])
    return {
        "algorithm": run.name, "seed": run.seed, "training_steps": steps,
        "environments": (run.config.get("num_envs") if run.name == "A2C"
                         else run.config["num_envs_total"]),
        "workers": workers, "device": run.config["device"],
        "episodes_completed": s.get("episodes_completed"),
        "gradient_updates": s.get("gradient_updates"),
        "converged": s.get("converged"), "first_convergence_steps": first_conv,
        "first_nonnegative_steps": first_zero(run),
        "train_mean_reward_last100": last100,
        "final_evaluation_reward_mean": final_mean,
        "final_evaluation_reward_std": final_std,
        "best_checkpoint_steps": number(best["training_steps"]) if best else None,
        "best_checkpoint_evaluation_mean": number(best["evaluation_reward_mean"]) if best else None,
        "training_elapsed_seconds": elapsed,
        "throughput_interactions_per_second": steps / elapsed if elapsed > 0 else None,
        "run_directory": run.directory.name,
    }


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def learning_curves(runs: list[Run], path: Path):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for r in runs:
        x = np.array([number(v["steps"]) for v in r.episodes]) / 1e6
        y = np.array([number(v["mean_reward_100"]) for v in r.episodes])
        if x.size:
            ax.plot(x, y, label=f"{r.name} seed {r.seed}",
                    color=COLORS[r.name], linestyle="--" if r.name == "A3C-grad" else "-",
                    linewidth=1.15, alpha=.72)
    ax.axhline(THRESHOLD, color="black", linestyle=":", label="Convergencia > +18")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(title="E07 — Curvas de aprendizado A2C vs A3C-grad",
           xlabel="Interacoes dos ambientes (milhoes)",
           ylabel="Recompensa media dos ultimos 100 jogos")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    save(fig, path)


def checkpoint_curves(runs: list[Run], path: Path):
    fig, ax = plt.subplots(figsize=(12, 5.3))
    for r in runs:
        c = checkpoints(r)
        if not c:
            continue
        ax.plot([number(v["training_steps"]) / 1e6 for v in c],
                [number(v["evaluation_reward_mean"]) for v in c],
                marker="o", markersize=3, linewidth=1.2,
                label=f"{r.name} seed {r.seed}", color=COLORS[r.name],
                linestyle="--" if r.name == "A3C-grad" else "-")
    ax.axhline(THRESHOLD, linestyle=":", color="black", label="Referencia +18")
    ax.set(title="E07 — Avaliacoes independentes nos checkpoints",
           xlabel="Interacoes dos ambientes (milhoes)",
           ylabel="Recompensa media nas partidas de avaliacao")
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=8)
    save(fig, path)


def grouped_bars(pairs, key, title, ylabel, path, divisor=1.0):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(pairs))
    for offs, idx, name in ((-.18, 0, "A2C"), (.18, 1, "A3C-grad")):
        vals = [number(pair[idx][key]) / divisor for pair in pairs]
        ax.bar(x + offs, vals, width=.36, label=name, color=COLORS[name])
    ax.set_xticks(x, [str(pair[0]["seed"]) for pair in pairs])
    ax.set(xlabel="Semente", ylabel=ylabel, title=title)
    ax.grid(alpha=.18, axis="y")
    ax.legend()
    save(fig, path)


def plot_gradients(runs: list[Run], path: Path):
    fig, ax = plt.subplots(figsize=(12, 5.3))
    for run in runs:
        if run.name != "A3C-grad":
            continue
        # Amostragem para reduzir tamanho da figura com muitas atualizacoes.
        stride = max(1, len(run.updates) // 3000)
        u = run.updates[::stride]
        if u:
            ax.plot([number(r["steps"]) / 1e6 for r in u],
                    [number(r["aggregated_grad_rms"]) for r in u],
                    label=f"seed {run.seed}", linewidth=.75, alpha=.7)
    ax.set(title="E07 — RMS do gradiente agregado antes do clipping central",
           xlabel="Interacoes com os ambientes (milhoes)",
           ylabel="RMS do gradiente agregado")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, path)


def plot_worker_load(runs: list[Run], path: Path):
    a3c = [r for r in runs if r.name == "A3C-grad"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    if a3c:
        x = np.arange(len(a3c))
        width = .8 / max(1, max(len(r.workers) for r in a3c))
        for wid in range(max(len(r.workers) for r in a3c)):
            counts = []
            for run in a3c:
                row = next((v for v in run.workers if int(v["worker_id"]) == wid), None)
                counts.append(number(row["gradient_messages"], 0) if row else 0)
            ax.bar(x - .4 + width * (wid + .5), counts, width=width,
                   label=f"worker {wid}")
        ax.set_xticks(x, [str(r.seed) for r in a3c])
    ax.set(title="E07 — Contribuicao de gradientes por processo",
           xlabel="Semente", ylabel="Pacotes de gradientes enviados")
    ax.legend(ncol=4)
    ax.grid(alpha=.2, axis="y")
    save(fig, path)


def plot_best_final(pairs, path):
    fig, ax = plt.subplots(figsize=(11, 5.1))
    labels, best, final = [], [], []
    for a, b in pairs:
        for run in (a, b):
            labels.append(f"{run['algorithm']}\n{run['seed']}")
            best.append(number(run["best_checkpoint_evaluation_mean"]))
            final.append(number(run["final_evaluation_reward_mean"]))
    x = np.arange(len(labels))
    ax.bar(x - .17, best, width=.34, label="Melhor checkpoint")
    ax.bar(x + .17, final, width=.34, label="Checkpoint final")
    ax.set_xticks(x, labels)
    ax.set(title="E07 — Melhor checkpoint observado vs avaliacao final",
           ylabel="Recompensa media de avaliacao")
    ax.legend()
    ax.grid(alpha=.2, axis="y")
    save(fig, path)


def analyze(args):
    protocol = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    a2c = collect(args, "A2C", protocol)
    a3c = collect(args, "A3C-grad", protocol)
    overlap = sorted(set(a2c) & set(a3c))
    missing = {"a2c": sorted(set(args.seeds) - set(a2c)),
               "a3c_grad": sorted(set(args.seeds) - set(a3c))}
    if any(missing.values()):
        print(f"Dados incompletos; sementes pareadas={overlap}; faltantes={missing}")
        if args.require_all:
            raise ValueError("--require-all exige execucoes completas das duas variantes")
    if not overlap:
        raise ValueError("Sem sementes pareadas. Conclua E04 e E07 antes da analise comparativa")
    runs = [run for seed in overlap for run in (a2c[seed], a3c[seed])]
    summary = [summary_row(r) for r in runs]
    pairs = [(summary_row(a2c[s]), summary_row(a3c[s])) for s in overlap]
    for seed in overlap:
        left, right = a2c[seed].config, a3c[seed].config
        for field in ("evaluation_seed_base", "evaluation_max_episode_steps",
                      "convergence_window", "convergence_mean_reward_strict_greater_than"):
            if left.get(field) != right.get(field):
                raise ValueError(f"Protocolo incompativel seed={seed}, campo={field}")
    out = ROOT / "analises" / "e07" / f"steps_{args.steps}" / protocol
    out.mkdir(parents=True, exist_ok=True)
    store_csv(out / "resumo_execucoes.csv", summary)
    store_csv(out / "comparacao_pareada.csv", [{
        "seed": a["seed"],
        "a2c_final_reward": a["final_evaluation_reward_mean"],
        "a3c_final_reward": b["final_evaluation_reward_mean"],
        "delta_a3c_minus_a2c": number(b["final_evaluation_reward_mean"]) - number(a["final_evaluation_reward_mean"]),
        "a2c_throughput": a["throughput_interactions_per_second"],
        "a3c_throughput": b["throughput_interactions_per_second"],
        "device_a2c": a["device"], "device_a3c": b["device"],
        "warning": "Device e configuracoes diferem; nao atribuir diferenca apenas ao algoritmo",
    } for a, b in pairs])
    store_csv(out / "episodios_treinamento.csv", [
        {"algorithm": r.name, "seed": r.seed, **e} for r in runs for e in r.episodes])
    store_csv(out / "resumo_checkpoints.csv", [
        {"algorithm": r.name, "seed": r.seed, **c} for r in runs for c in r.checkpoints])
    store_csv(out / "episodios_avaliacao.csv", [
        {"algorithm": r.name, "seed": r.seed, **e} for r in runs for e in r.evaluation])
    store_csv(out / "atualizacoes_a3c.csv", [
        {"seed": r.seed, **u} for r in runs if r.name == "A3C-grad" for u in r.updates])
    store_csv(out / "contribuicoes_workers.csv", [
        {"seed": r.seed, **w} for r in runs if r.name == "A3C-grad" for w in r.workers])
    learning_curves(runs, out / "01_curvas_aprendizado.png")
    checkpoint_curves(runs, out / "02_avaliacoes_checkpoints.png")
    grouped_bars(pairs, "final_evaluation_reward_mean",
                 "E07 — Desempenho final das politicas", "Recompensa media final",
                 out / "03_avaliacao_final.png")
    grouped_bars(pairs, "throughput_interactions_per_second",
                 "E07 — Vazao observada (E04 MPS x E07 CPU)",
                 "Interacoes/s (dispositivos diferentes)", out / "04_vazao.png")
    plot_gradients(runs, out / "05_gradientes_a3c.png")
    plot_worker_load(runs, out / "06_contribuicoes_workers.png")
    grouped_bars(pairs, "training_elapsed_seconds",
                 "E07 — Tempo de treinamento (MPS x CPU; nao isolado)",
                 "Minutos de treinamento", out / "07_tempo.png", divisor=60)
    plot_best_final(pairs, out / "08_melhor_vs_final.png")
    ratios = [number(b["throughput_interactions_per_second"]) /
              number(a["throughput_interactions_per_second"]) for a, b in pairs]
    record = {
        "experiment": "E07", "environment": "PongNoFrameskip-v4",
        "criteria": {"max_training_steps": args.steps,
                     "stop_on_convergence": args.stop_on_convergence,
                     "evaluation_episodes": args.eval_episodes,
                     "a2c_num_envs": args.a2c_num_envs,
                     "a3c_processes": args.processes,
                     "a3c_num_envs_per_process": args.num_envs,
                     "seeds_requested": args.seeds, "seeds_paired": overlap},
        "runs": summary,
        "mean_final_a2c": mean(number(a["final_evaluation_reward_mean"]) for a, _ in pairs),
        "mean_final_a3c": mean(number(b["final_evaluation_reward_mean"]) for _, b in pairs),
        "observed_mean_throughput_ratio_a3c_over_a2c": mean(ratios),
        "observed_throughput_ratio_sd": stdev(ratios) if len(ratios) > 1 else None,
        "notes": [
            "E07 usa quatro workers de oito ambientes por padrao e gradientes calculados em processos independentes.",
            "E04 calcula os gradientes e aplica Adam no processo principal com 50 ambientes sincronos.",
            "E07 rede compartilhada e otimizador CPU; E04 treinado no MPS do Mac: vazao e tempo NAO isolam efeito da arquitetura.",
            "E04 usa 50 ambientes e batch 128; E07 usa 4x8, batch grad 64 e agrega 2 lotes: configuracoes distintas.",
            "As recompensas representam avaliacoes com a mesma politica estocastica e seeds fixas.",
            "As timestamps dos jogos no E07 aproximam o contador agregado ao receber o pacote de gradientes.",
            "No E07, a vantagem usa squeeze(-1) para remover broadcasting incorreto no codigo original.",
            "Os workers usam snapshots locais da rede compartilhada; modelo global atualizado pelo processo principal.",
            "A3C-grad com processo principal unico de Adam difere de Hogwild A3C classico com otimizadores paralelos.",
        ],
    }
    (out / "resumo_analise.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"E07 | sementes pareadas={overlap} | final A2C={record['mean_final_a2c']:.2f} "
          f"| A3C-grad={record['mean_final_a3c']:.2f}")
    print(f"Arquivos gerados em: {out}")


def main():
    p = argparse.ArgumentParser(description="E07 — Analise A3C-grad x A2C Pong")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=8,
                   help="Ambientes por processo A3C-grad")
    p.add_argument("--a2c-num-envs", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--require-all", action="store_true")
    flag = p.add_mutually_exclusive_group()
    flag.add_argument("--stop-on-convergence", action="store_true", dest="stop_on_convergence")
    flag.add_argument("--no-stop-on-convergence", action="store_false", dest="stop_on_convergence")
    p.set_defaults(stop_on_convergence=True)
    args = p.parse_args()
    if min(args.steps, args.processes, args.num_envs, args.a2c_num_envs,
           args.eval_episodes) <= 0:
        p.error("Numeros devem ser positivos")
    analyze(args)


if __name__ == "__main__":
    main()

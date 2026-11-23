#!/usr/bin/env python3
"""E06 — Comparar SyncVectorEnv (E04 ou E06) com AsyncVectorEnv (E06) no Pong.

Reutiliza E04 como referencia sincrona por padrao, sem duplicar dados brutos.
Somente sementes pareadas, protocolo identico, orcamento completo e configuracao
A2C coincidente entram na comparacao. Nao compara frames ALE com passos Gym.

Exemplo apos completar as tres sementes de E04 e E06:
  python experimentos-unidade-3/analise_e06.py \
    --steps 5000000 --seeds 42 123 2026 --num-envs 50 \
    --eval-episodes 10 --no-stop-on-convergence --require-all

Para comparar dois pilotos realizados no E06 (um sync e outro async), use:
  --sync-source e06
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
OUT_BASE = ROOT / "analises" / "e06"
THRESHOLD = 18.0


@dataclass
class Run:
    mode: str
    seed: int
    directory: Path
    config: dict
    summary: dict
    episodes: list[dict]
    evaluation: list[dict]
    checkpoints: list[dict]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def store_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def n(value, default=float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mode_dir(protocol: str, mode: str, sync_source: str) -> Path:
    if mode == "sync" and sync_source == "e04":
        return ROOT / "results" / "e04" / "completo" / protocol
    return ROOT / "results" / "e06" / "completo" / protocol / mode


def collect(args, protocol: str, mode: str) -> dict[int, Run]:
    directory = mode_dir(protocol, mode, args.sync_source)
    expected_exp = "E04" if mode == "sync" and args.sync_source == "e04" else "E06"
    if not directory.is_dir():
        return {}
    results = {}
    for item in sorted(x for x in directory.iterdir() if x.is_dir()):
        cf = item / "config.json"
        sf = item / "summary.json"
        if not cf.is_file() or not sf.is_file():
            continue
        conf = load_json(cf)
        if conf.get("experiment") != expected_exp or conf.get("algorithm") != "a2c":
            continue
        if conf.get("vectorization") != mode:
            continue
        if (int(conf.get("max_training_steps", -1)) != args.steps
                or int(conf.get("num_envs", -1)) != args.num_envs
                or int(conf.get("evaluation_episodes", -1)) != args.eval_episodes
                or bool(conf.get("stop_on_convergence", True)) != args.stop_on_convergence):
            continue
        seed = int(conf.get("seed", -1))
        if seed not in args.seeds:
            continue
        summ = load_json(sf)
        if not args.stop_on_convergence and int(summ.get("training_steps", 0)) < args.steps:
            continue
        mandatory = ("episodes.csv", "evaluation.csv", "checkpoints.csv",
                     "updates.csv", "checkpoint_evaluations.csv")
        missing = [f for f in mandatory if not (item / f).is_file()]
        if missing:
            raise FileNotFoundError(f"Dados incompletos em {item}: {missing}")
        if seed in results:
            raise ValueError(f"Execucao duplicada para {mode} seed={seed}: "
                             f"{results[seed].directory} e {item}")
        results[seed] = Run(mode, seed, item, conf, summ,
                            load_csv(item / "episodes.csv"),
                            load_csv(item / "evaluation.csv"),
                            load_csv(item / "checkpoints.csv"))
    return results


def compare_config(a: Run, b: Run):
    if a.seed != b.seed:
        raise ValueError("Sementes nao pareadas")
    for key in ("evaluation_seed_base", "evaluation_max_episode_steps", "device",
                "evaluation_interval_steps", "gradient_metrics_interval_updates",
                "convergence_window", "convergence_mean_reward_strict_greater_than",
                "step_definition"):
        if a.config.get(key) != b.config.get(key):
            raise ValueError(f"Protocolo incompativel seed {a.seed}: {key}: "
                             f"sync={a.config.get(key)}, async={b.config.get(key)}")
    if a.config.get("train_params") != b.config.get("train_params"):
        raise ValueError(f"Hiperparametros A2C diferem na semente {a.seed}")
    if len(a.evaluation) != len(b.evaluation):
        raise ValueError(f"Episodios finais distintos seed={a.seed}")


def first_nonnegative(run: Run):
    for row in run.episodes:
        if int(row["episode"]) >= 100 and n(row["mean_reward_100"]) >= 0:
            return int(row["steps"])
    return None


def run_summary(run: Run) -> dict:
    summ = run.summary
    checkpoints = [r for r in run.checkpoints if r.get("checkpoint_type") in {"periodic", "final"}]
    best = max(checkpoints, key=lambda r: n(r["evaluation_reward_mean"])) if checkpoints else None
    elapsed = n(summ.get("training_elapsed_seconds_excluding_evaluations"))
    steps = int(summ["training_steps"])
    throughput = steps / elapsed if elapsed > 0 else float("nan")
    return {
        "vectorization": run.mode,
        "seed": run.seed,
        "training_steps": steps,
        "num_envs": run.config["num_envs"],
        "device": run.config["device"],
        "vector_steps": summ.get("vector_steps"),
        "gradient_updates": summ.get("gradient_updates"),
        "episodes_completed": summ.get("episodes_completed"),
        "converged": bool(summ.get("converged")),
        "first_convergence_steps": (summ.get("convergence") or {}).get("steps"),
        "first_nonnegative_training_100_steps": first_nonnegative(run),
        "final_training_mean_100": summ.get("training_reward_mean_last_100"),
        "final_eval_mean": summ["evaluation"]["reward_mean"],
        "final_eval_std": summ["evaluation"]["reward_std"],
        "best_checkpoint_steps": best["training_steps"] if best else None,
        "best_checkpoint_eval_mean": best["evaluation_reward_mean"] if best else None,
        "training_elapsed_seconds": elapsed,
        "throughput_interactions_per_second": throughput,
        "evaluation_and_checkpoint_seconds": summ.get("checkpoint_and_evaluation_elapsed_seconds"),
        "source": str(run.directory),
    }


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def grouped_bar(pairs, filename, title, ylabel, field, divide=1):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(pairs))
    v1 = [n(a[field]) / divide for a, _ in pairs]
    v2 = [n(b[field]) / divide for _, b in pairs]
    ax.bar(x - .18, v1, width=.36, label="SyncVectorEnv — A2C")
    ax.bar(x + .18, v2, width=.36, label="AsyncVectorEnv — A2C")
    ax.set_xticks(x, [str(a["seed"]) for a, _ in pairs])
    ax.set(title=title, xlabel="Semente", ylabel=ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    save(fig, filename)


def plot_speedup(pairs, out):
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ratios = [n(b["throughput_interactions_per_second"]) /
              n(a["throughput_interactions_per_second"]) for a, b in pairs]
    x = np.arange(len(pairs))
    ax.bar(x, ratios)
    ax.axhline(1., linestyle="--", color="black", label="Mesma vazao")
    ax.set_xticks(x, [str(a["seed"]) for a, _ in pairs])
    ax.set(title="E06 — Razao de vazao: async / sync",
           ylabel="Fator de aceleracao (>1 favorece async)", xlabel="Semente")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    save(fig, out)


def plot_learning(runs, out):
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for run in runs:
        xs = [n(r["steps"]) / 1e6 for r in run.episodes]
        ys = [n(r["mean_reward_100"]) for r in run.episodes]
        if xs:
            ax.plot(xs, ys, linestyle="-" if run.mode == "sync" else "--",
                    alpha=.78, linewidth=1.25,
                    label=f"{run.mode} seed {run.seed}")
    ax.axhline(THRESHOLD, linestyle=":", color="black", label="Convergencia > +18")
    ax.axhline(0., color="gray", linewidth=.7)
    ax.set(title="E06 — Recompensa durante o treino: coleta sync e async",
           xlabel="Interacoes com os ambientes (milhoes)",
           ylabel="Media movel dos ultimos 100 jogos")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    save(fig, out)


def plot_checkpoints(runs, out):
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for run in runs:
        points = sorted((r for r in run.checkpoints
                         if r["checkpoint_type"] in {"periodic", "final"}),
                        key=lambda r: n(r["training_steps"]))
        if not points:
            continue
        ax.plot([n(r["training_steps"]) / 1e6 for r in points],
                [n(r["evaluation_reward_mean"]) for r in points],
                marker="o", markersize=3, linewidth=1.1,
                linestyle="-" if run.mode == "sync" else "--",
                label=f"{run.mode} seed {run.seed}")
    ax.axhline(THRESHOLD, linestyle=":", color="black", label="Referencia +18")
    ax.set(title="E06 — Avaliacao independente dos checkpoints",
           xlabel="Interacoes com os ambientes (milhoes)",
           ylabel="Media de recompensa (jogos de avaliacao)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    save(fig, out)


def plot_first_zero(pairs, out):
    # Intervalo em interacoes; compara eficiencia amostral, nao tempo real.
    if not any(a["first_nonnegative_training_100_steps"] is not None or
               b["first_nonnegative_training_100_steps"] is not None for a, b in pairs):
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    x = np.arange(len(pairs))
    for offset, mode in ((-.18, "sync"), (.18, "async")):
        values = [n(a["first_nonnegative_training_100_steps"]
                    if mode == "sync" else b["first_nonnegative_training_100_steps"])/1e6
                  for a, b in pairs]
        ax.bar(x + offset, values, width=.36, label=mode)
    ax.set_xticks(x, [str(a["seed"]) for a, _ in pairs])
    ax.set(title="E06 — Primeiro resultado nao negativo da media movel (100 jogos)",
           xlabel="Semente", ylabel="Milhoes de interacoes")
    ax.legend()
    ax.grid(alpha=.2, axis="y")
    save(fig, out)


def analyze(args):
    protocol = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    sync = collect(args, protocol, "sync")
    async_ = collect(args, protocol, "async")
    seeds = sorted(set(sync) & set(async_))
    missing = {"sync": sorted(set(args.seeds) - set(sync)),
               "async": sorted(set(args.seeds) - set(async_))}
    if any(missing.values()):
        print(f"Sementes incompletas; pares validos={seeds}; faltantes={missing}")
        if args.require_all:
            raise ValueError("--require-all exige todas as sementes nas duas modalidades")
    if not seeds:
        raise ValueError("Nenhuma semente pareada completa. Rode E04 sync e E06 async "
                         "ou pilotos E06 sync/async com --sync-source e06")
    runs = []
    summaries = []
    pairs = []
    for seed in seeds:
        a, b = sync[seed], async_[seed]
        compare_config(a, b)
        sa, sb = run_summary(a), run_summary(b)
        runs.extend((a, b))
        summaries.extend((sa, sb))
        pairs.append((sa, sb))

    out = OUT_BASE / f"steps_{args.steps}" / protocol
    out.mkdir(parents=True, exist_ok=True)
    store_csv(out / "resumo_execucoes.csv", summaries)
    pair_rows = []
    for a, b in pairs:
        t_sync, t_async = a["training_elapsed_seconds"], b["training_elapsed_seconds"]
        f_sync, f_async = a["throughput_interactions_per_second"], b["throughput_interactions_per_second"]
        pair_rows.append({
            "seed": a["seed"],
            "sync_seconds": t_sync, "async_seconds": t_async,
            "sync_interactions_s": f_sync, "async_interactions_s": f_async,
            "async_div_sync_speedup": f_async / f_sync,
            "time_saved_seconds_async": t_sync - t_async,
            "sync_final_evaluation_mean": a["final_eval_mean"],
            "async_final_evaluation_mean": b["final_eval_mean"],
            "delta_async_minus_sync_final_eval": b["final_eval_mean"] - a["final_eval_mean"],
        })
    store_csv(out / "comparacao_pareada.csv", pair_rows)
    store_csv(out / "episodios_treinamento.csv", [
        {"vectorization": r.mode, "seed": r.seed, **e}
        for r in runs for e in r.episodes])
    store_csv(out / "resumo_checkpoints.csv", [
        {"vectorization": r.mode, "seed": r.seed, **c}
        for r in runs for c in r.checkpoints])
    store_csv(out / "episodios_avaliacao.csv", [
        {"vectorization": r.mode, "seed": r.seed, **e}
        for r in runs for e in r.evaluation])

    grouped_bar(pairs, out / "01_vazao_comparativa.png",
                "E06 — Vazao total: coleta e treinamento A2C",
                "Interacoes de ambientes por segundo", "throughput_interactions_per_second")
    grouped_bar(pairs, out / "02_tempo_comparativo.png",
                "E06 — Tempo de treinamento (exclui avaliacoes)",
                "Minutos", "training_elapsed_seconds", divide=60)
    plot_speedup(pairs, out / "03_speedup.png")
    plot_learning(runs, out / "04_curvas_aprendizado.png")
    plot_checkpoints(runs, out / "05_avaliacoes_checkpoints.png")
    grouped_bar(pairs, out / "06_avaliacao_final.png",
                "E06 — Recompensa final por semente",
                "Recompensa media (10 partidas no protocolo completo)", "final_eval_mean")
    plot_first_zero(pairs, out / "07_primeira_media_nao_negativa.png")

    ratio = [row["async_div_sync_speedup"] for row in pair_rows]
    record = {
        "experiment": "E06", "environment": "PongNoFrameskip-v4",
        "sync_source": args.sync_source,
        "criteria": {"max_training_steps": args.steps,
                     "stop_on_convergence": args.stop_on_convergence,
                     "evaluation_episodes": args.eval_episodes,
                     "num_envs": args.num_envs,
                     "requested_seeds": args.seeds,
                     "paired_seeds": seeds},
        "runs": summaries, "paired_comparison": pair_rows,
        "mean_sync_throughput": mean([n(a["throughput_interactions_per_second"]) for a, _ in pairs]),
        "mean_async_throughput": mean([n(b["throughput_interactions_per_second"]) for _, b in pairs]),
        "mean_paired_speedup": mean(ratio),
        "std_paired_speedup_between_seeds": stdev(ratio) if len(ratio) > 1 else None,
        "notes": [
            "AsyncVectorEnv usa subprocessos para avancar ambientes; atualizacoes e inferencia da rede permanecem centralizadas: nao e A3C.",
            "Cada step de treino conta uma interacao de cada subambiente wrapped, nao um frame bruto ALE.",
            "O tempo comparado exclui duracao de checkpoints e avaliacoes, mas pode incluir esperas e custos da propria execucao.",
            "Em chamadas vetorizadas, a coleta espera o lote de todos os ambientes; nao ha atualizacao de gradientes assincrona.",
            "Comparacao de vazao e exploratoria, sem controle do uso concorrente da maquina por outros processos.",
            "E04 sync e E06 async podem divergir estocasticamente apesar dos mesmos hiperparametros e sementes; nao atribuir diferencas de recompensa apenas ao tipo de vetor.",
        ],
    }
    (out / "resumo_analise.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"E06 | sementes pareadas={seeds} | media vazao sync="
          f"{record['mean_sync_throughput']:.1f} int/s | async="
          f"{record['mean_async_throughput']:.1f} int/s | speedup medio="
          f"{record['mean_paired_speedup']:.3f}x")
    print(f"Graficos e relatorios: {out}")


def main():
    p = argparse.ArgumentParser(description="E06 — Comparar vetorizacao sync x async A2C Pong")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--num-envs", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--sync-source", choices=("e04", "e06"), default="e04",
                   help="Padrao: reutiliza as execucoes sync do E04")
    p.add_argument("--require-all", action="store_true")
    flag = p.add_mutually_exclusive_group()
    flag.add_argument("--stop-on-convergence", action="store_true", dest="stop_on_convergence")
    flag.add_argument("--no-stop-on-convergence", action="store_false", dest="stop_on_convergence")
    p.set_defaults(stop_on_convergence=True)
    args = p.parse_args()
    if min(args.steps, args.num_envs, args.eval_episodes) < 1:
        p.error("Todos os parametros numericos devem ser positivos")
    analyze(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E04 — Análise do A2C no Pong: recompensas, crítico, vantagem e gradientes.

Exemplo:
  python experimentos-unidade-3/analise_e04.py --steps 4000000 \\
    --no-stop-on-convergence --seeds 42

Lê results/e04/completo/<modo>/<execucao>/ e grava
analises/e04/steps_<orcamento>/<modo>/.
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
RESULTS_DIR = UNIT_DIR / "results" / "e04" / "completo"
OUTPUT_BASE = UNIT_DIR / "analises" / "e04"
THRESHOLD = 18.0


@dataclass
class Run:
    seed: int
    directory: Path
    config: dict
    summary: dict
    episodes: list[dict]
    evaluation: list[dict]
    checkpoints: list[dict]
    checkpoint_eval: list[dict]
    updates: list[dict]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as inp:
        return list(csv.DictReader(inp))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if fields is None:
        if not rows:
            return
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def load_runs(args) -> list[Run]:
    mode = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    folder = RESULTS_DIR / mode
    if not folder.is_dir():
        raise FileNotFoundError(f"Nenhum resultado disponivel: {folder}")
    selected = {}
    for directory in sorted(p for p in folder.iterdir() if p.is_dir()):
        cfile = directory / "config.json"
        if not cfile.is_file():
            continue
        config = read_json(cfile)
        if config.get("experiment") != "E04" or config.get("algorithm") != "a2c":
            continue
        if (int(config.get("max_training_steps", -1)) != args.steps
            or bool(config.get("stop_on_convergence", True)) != args.stop_on_convergence
            or int(config.get("evaluation_episodes", -1)) != args.eval_episodes
            or (args.num_envs is not None and int(config.get("num_envs", -1)) != args.num_envs)):
            continue
        seed = int(config.get("seed", -1))
        if seed not in args.seeds:
            continue
        if seed in selected:
            raise ValueError(f"Execucoes duplicadas para seed={seed}: "
                             f"{selected[seed].directory} e {directory}. "
                             "Selecione --num-envs ou mova a execucao duplicada.")
        required = ["summary.json", "episodes.csv", "updates.csv", "evaluation.csv",
                    "checkpoints.csv", "checkpoint_evaluations.csv"]
        missing = [f for f in required if not (directory / f).is_file()]
        if missing:
            raise FileNotFoundError(f"Arquivos ausentes em {directory}: {missing}")
        selected[seed] = Run(
            seed=seed, directory=directory, config=config,
            summary=read_json(directory / "summary.json"),
            episodes=read_csv(directory / "episodes.csv"),
            evaluation=read_csv(directory / "evaluation.csv"),
            checkpoints=read_csv(directory / "checkpoints.csv"),
            checkpoint_eval=read_csv(directory / "checkpoint_evaluations.csv"),
            updates=read_csv(directory / "updates.csv"),
        )
    if not selected:
        raise ValueError("Nao foram encontradas execucoes completas para os filtros informados.")
    missing_seeds = sorted(set(args.seeds) - set(selected))
    if missing_seeds:
        print(f"Aviso: nao existem execucoes para as sementes: {missing_seeds}")
    return [selected[k] for k in sorted(selected)]


def save_figure(path: Path):
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def series_plot(
    runs: list[Run], filename: Path, title: str, x: str, y: str,
    ylabel: str, smoothing: int = 1, yline: float | None = None,
    xlabel: str = "Interacoes com os subambientes", log_y: bool = False,
):
    plt.figure(figsize=(12.5, 5.4))
    any_data = False
    for run in runs:
        data = run.episodes if x == "episodes" else run.updates
        xv, yv = [], []
        for row in data:
            xf, yf = read_float(row.get("steps")), read_float(row.get(y))
            if np.isfinite(xf) and np.isfinite(yf) and (not log_y or yf > 0):
                xv.append(xf)
                yv.append(yf)
        if not xv:
            continue
        if smoothing > 1:
            vals = np.asarray(yv, dtype=float)
            window = min(smoothing, len(vals))
            valid = np.convolve(vals, np.ones(window), mode="same")
            counts = np.convolve(np.ones_like(vals), np.ones(window), mode="same")
            yv = (valid / counts).tolist()
        plt.plot(xv, yv, label=f"Semente {run.seed}", alpha=0.85)
        any_data = True
    if yline is not None:
        plt.axhline(yline, linestyle=":", color="black", label=f"Criterio > {yline:g}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale("log")
    if any_data:
        plt.legend()
    save_figure(filename)


def eval_plot(runs: list[Run], output: Path):
    plt.figure(figsize=(12, 5.5))
    for run in runs:
        periods = [r for r in run.checkpoints if r["checkpoint_type"] in {"periodic", "final"}]
        periods.sort(key=lambda r: read_float(r["training_steps"]))
        x = [float(r["training_steps"]) for r in periods]
        y = [float(r["evaluation_reward_mean"]) for r in periods]
        if x:
            (ln,) = plt.plot(x, y, marker="o", label=f"Semente {run.seed}")
            fc = [r for r in run.checkpoints if r["checkpoint_type"] == "first_convergence"]
            for c in fc:
                plt.scatter(float(c["training_steps"]), float(c["evaluation_reward_mean"]),
                            marker="X", s=110, color=ln.get_color(), zorder=4)
    plt.axhline(THRESHOLD, linestyle=":", color="black", label="Referencia +18")
    plt.title("E04 — Avaliacao independente por checkpoint")
    plt.xlabel("Interacoes dos subambientes no checkpoint")
    plt.ylabel("Recompensa media de avaliacao")
    plt.legend()
    save_figure(output)


def final_boxplot(runs: list[Run], output: Path):
    groups = [[float(v["reward"]) for v in r.evaluation] for r in runs]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.boxplot(groups, tick_labels=[f"Semente {r.seed}" for r in runs],
               showmeans=True, meanline=True,
               meanprops={"color": "green", "linestyle": "--"},
               medianprops={"color": "darkorange"})
    ax.axhline(THRESHOLD, linestyle=":", label="Referencia +18")
    ax.set_title("E04 — Distribuicao da recompensa na avaliacao final")
    ax.set_ylabel("Recompensa por jogo (-21 a +21)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def throughput_plot(runs: list[Run], output: Path):
    xs = np.arange(len(runs))
    speeds = [float(r.summary["wrapped_interactions_per_training_second"]) for r in runs]
    plt.figure(figsize=(9.5, 5.2))
    plt.bar(xs, speeds)
    plt.xticks(xs, [f"Semente {r.seed}" for r in runs])
    plt.title("E04 — Vazao media de coleta e treinamento")
    plt.ylabel("Interacoes de subambientes por segundo (sem avaliacoes)")
    save_figure(output)


def analyze(args):
    runs = load_runs(args)
    mode = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    out = OUTPUT_BASE / f"steps_{args.steps}" / mode
    out.mkdir(parents=True, exist_ok=True)

    merged_episodes, merged_updates, merged_checkpoints = [], [], []
    merged_eval, merged_checkpoint_eval, summaries = [], [], []
    for r in runs:
        for rows, target in ((r.episodes, merged_episodes), (r.updates, merged_updates),
                             (r.checkpoints, merged_checkpoints), (r.evaluation, merged_eval),
                             (r.checkpoint_eval, merged_checkpoint_eval)):
            target.extend({"algorithm": "A2C", "seed": r.seed, **row} for row in rows)
        conv = r.summary.get("convergence")
        summaries.append({
            "algorithm": "A2C", "seed": r.seed, "training_steps": r.summary["training_steps"],
            "vector_steps": r.summary["vector_steps"],
            "gradient_updates": r.summary["gradient_updates"],
            "episodes_completed": r.summary["episodes_completed"],
            "converged": bool(r.summary["converged"]),
            "first_convergence_steps": conv["steps"] if conv else "",
            "first_convergence_episodes": conv["episodes"] if conv else "",
            "train_mean_reward_last100": r.summary.get("training_reward_mean_last_100"),
            "final_evaluation_reward_mean": r.summary["evaluation"]["reward_mean"],
            "final_evaluation_reward_std": r.summary["evaluation"]["reward_std"],
            "training_elapsed_seconds_excluding_evaluations": r.summary[
                "training_elapsed_seconds_excluding_evaluations"],
            "wrapped_interactions_per_training_second": r.summary[
                "wrapped_interactions_per_training_second"],
            "num_envs": r.config["num_envs"], "device": r.config["device"],
            "directory": r.directory.name,
        })

    for name, rows in [
        ("resumo_execucoes.csv", summaries),
        ("episodios_treinamento.csv", merged_episodes),
        ("atualizacoes_treinamento.csv", merged_updates),
        ("resumo_checkpoints.csv", merged_checkpoints),
        ("avaliacoes_checkpoints.csv", merged_checkpoint_eval),
        ("episodios_avaliacao.csv", merged_eval),
    ]:
        write_csv(out / name, rows)

    converged = [r for r in summaries if r["converged"]]
    avgs = [float(r["final_evaluation_reward_mean"]) for r in summaries]
    record = {
        "algorithm": "A2C", "n_seeds": len(runs),
        "convergences": len(converged), "convergence_rate": len(converged)/len(runs),
        "mean_training_steps_to_first_convergence_when_converged": (
            mean([float(r["first_convergence_steps"]) for r in converged]) if converged else None),
        "mean_final_evaluation_reward_by_seed": mean(avgs),
        "std_final_evaluation_reward_between_seeds": stdev(avgs) if len(avgs)>1 else 0.,
        "mean_throughput_wrapped_steps_s": mean([
            float(r["wrapped_interactions_per_training_second"]) for r in summaries]),
    }
    write_csv(out / "resumo_algoritmos.csv", [record])

    series_plot(runs, out / "01_curvas_aprendizado.png",
                "E04 — Evolucao da recompensa media de 100 jogos", "episodes",
                "mean_reward_100", "Recompensa media dos ultimos 100 jogos",
                yline=THRESHOLD)
    eval_plot(runs, out / "02_avaliacoes_por_checkpoint.png")
    final_boxplot(runs, out / "03_distribuicao_avaliacao.png")
    metrics = [
        ("loss_policy", "04_perda_politica.png", "Perda da politica (Actor)", False),
        ("loss_value", "05_perda_critico.png", "Perda MSE do critico (Value)", True),
        ("entropy", "06_entropia_politica.png", "Entropia da politica (nats)", False),
        ("advantage_mean", "07_vantagem_media.png", "Vantagem media (retorno - V)", False),
        ("policy_grad_rms", "08_gradiente_politica.png", "RMS gradiente da politica", True),
    ]
    for key, fname, label, log_y in metrics:
        series_plot(runs, out / fname, f"E04 — {label}", "updates", key,
                    label, smoothing=args.smooth_updates, log_y=log_y)
    throughput_plot(runs, out / "09_throughput.png")
    full_summary = {
        "experiment": "E04", "environment": "PongNoFrameskip-v4",
        "criteria": {"max_training_steps": args.steps,
                     "stop_on_convergence": args.stop_on_convergence,
                     "evaluation_episodes": args.eval_episodes,
                     "seeds_selected": args.seeds, "num_envs": args.num_envs},
        "runs": summaries, "summary_algorithms": [record],
        "notes": [
            "Passos de treino = soma de env.step dos subambientes (com wrappers).",
            "Quadros ALE nao sao equivalentes aos passos de treino: wrappers usam frameskip.",
            "Convergencia: media de 100 jogos completos > +18; nao confundir com avaliacao.",
            "A avaliacao usa politica estocastica com seeds independentes do treino.",
            "Gradiente registra a variancia ENTRE coordenadas (nao entre amostras independentes).",
            "O exemplo original foi corrigido apenas no broadcasting da vantagem.",
            "A2C com ambientes SyncVectorEnv: nao constitui A3C nem gradientes assincronos.",
        ],
    }
    (out / "resumo_analise.json").write_text(
        json.dumps(full_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"E04 | {len(runs)} execucao(oes) | convergencias={len(converged)} "
          f"| recompensa media avaliacao final={record['mean_final_evaluation_reward_by_seed']:.2f}")
    print(f"Graficos, CSVs e analise: {out}")


def main():
    p = argparse.ArgumentParser(description="Analise E04: A2C Pong")
    p.add_argument("--steps", type=int, default=4_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--smooth-updates", type=int, default=100)
    flag = p.add_mutually_exclusive_group()
    flag.add_argument("--stop-on-convergence", action="store_true", dest="stop_on_convergence")
    flag.add_argument("--no-stop-on-convergence", action="store_false", dest="stop_on_convergence")
    p.set_defaults(stop_on_convergence=True)
    args = p.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

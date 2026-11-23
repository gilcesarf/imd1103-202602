#!/usr/bin/env python3
"""E03: analise de Policy Gradient (Chapter12/01_cartpole_pg.py), com/sem baseline.

Execucao:
  python experimentos-unidade-3/analise_e03.py --steps 200000 --no-stop-on-convergence

Entrada: results/e03/completo/<modo>/<execucao>/{episodes,evaluation,gradients,...}.csv
Saida: analises/e03/steps_<N>/<modo>/

Metodologia das metricas:
  grad_var_components: variancia das coordenadas do gradiente da perda PG
      no mesmo update (equivale ao grad_var registrado no codigo original).
  grad_temporal_var_window: variancia dos gradientes por parametro ao longo
      dos ultimos 50 updates, media por coordenada. Politica se altera.
  paired_grad_var_*: em checkpoints diagnosticos, variancia entre as 8
      contribuicoes individuais de um MESMO minibatch, para escalas com
      e sem baseline, mantendo theta e dados fixos. As amostras n-step
      consecutivas podem ser correlacionadas; NAO inferir variancia IID.

Todos os caminhos de figura usam nomes relativos para facilitar relatorios MD.
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

UNIT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = UNIT_DIR / "results" / "e03" / "completo"
OUTPUT_BASE_DIR = UNIT_DIR / "analises" / "e03"
DEFAULT_BUDGET_STEPS = 200_000
DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_EVAL_EPISODES = 20
ALGORITHMS = ("pg", "pg_baseline")
ALGORITHM_LABELS = {"pg": "PG sem baseline", "pg_baseline": "PG com baseline"}
CONVERGENCE_THRESHOLD = 450.0


@dataclass
class Run:
    algorithm: str
    seed: int
    directory: Path
    config: dict
    summary: dict
    episodes: list[dict]
    evaluations: list[dict]
    checkpoints: list[dict]
    checkpoint_evaluations: list[dict]
    gradients: list[dict]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"Sem dados para gerar: {path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_runs(
    budget_steps: int, stop_on_convergence: bool,
    expected_seeds: tuple[int, ...], eval_episodes: int,
) -> list[Run]:
    results_mode_dir = RESULTS_DIR / (
        "stop_on_convergence" if stop_on_convergence else "no_stop_on_convergence"
    )
    if not results_mode_dir.is_dir():
        raise FileNotFoundError(f"Diretorio de resultados nao encontrado: {results_mode_dir}")

    selected: dict[tuple[str, int], Run] = {}
    skipped = 0
    for directory in sorted(p for p in results_mode_dir.iterdir() if p.is_dir()):
        config_path = directory / "config.json"
        if not config_path.is_file():
            continue
        config = read_json(config_path)
        if config.get("experiment") != "E03":
            continue
        try:
            valid = (
                int(config.get("max_training_steps", -1)) == budget_steps
                and bool(config.get("stop_on_convergence", True)) == stop_on_convergence
                and int(config.get("evaluation_episodes", -1)) == eval_episodes
            )
        except (ValueError, TypeError):
            valid = False
        if not valid:
            skipped += 1
            continue

        algorithm = str(config.get("algorithm", "")).lower()
        seed = int(config.get("seed", -1))
        if algorithm not in ALGORITHMS or seed not in expected_seeds:
            skipped += 1
            continue
        key = (algorithm, seed)
        if key in selected:
            raise ValueError(
                f"Execucoes duplicadas para {algorithm}/seed={seed}: "
                f"{selected[key].directory.name} e {directory.name}. "
                "Mova ou remova a duplicata deste modo de resultados."
            )

        needed = ["summary.json", "episodes.csv", "evaluation.csv", "gradients.csv"]
        if not stop_on_convergence:
            needed.extend(["checkpoints.csv", "checkpoint_evaluations.csv"])
        missing = [name for name in needed if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Arquivos ausentes em {directory}: {missing}")

        summary = read_json(directory / "summary.json")
        episodes = read_csv(directory / "episodes.csv")
        evaluations = read_csv(directory / "evaluation.csv")
        gradients = read_csv(directory / "gradients.csv")
        checkpoints = read_csv(directory / "checkpoints.csv") if not stop_on_convergence else []
        checkpoint_evals = (
            read_csv(directory / "checkpoint_evaluations.csv")
            if not stop_on_convergence else []
        )

        steps_done = int(summary.get("training_steps", -1))
        if int(summary.get("max_training_steps", -1)) != budget_steps:
            raise ValueError(f"Orcamento inconsistente no summary de {directory.name}")
        if not (0 < steps_done <= budget_steps):
            raise ValueError(f"Numero de passos invalido em {directory.name}")
        if not stop_on_convergence and steps_done != budget_steps:
            raise ValueError(f"Treinamento sem parada nao chegou a {budget_steps}: {directory.name}")
        if not stop_on_convergence and summary.get("stop_reason") != "step_budget":
            raise ValueError(f"Motivo de parada incorreto em {directory.name}")
        if not episodes or len(episodes) != int(summary.get("episodes_completed", -1)):
            raise ValueError(f"Quantidade de episodios inconsistente em {directory.name}")
        if not gradients or len(gradients) != int(summary.get("gradient_updates", -1)):
            raise ValueError(f"Quantidade de gradientes inconsistente em {directory.name}")
        if any(int(gradients[i]["update"]) >= int(gradients[i + 1]["update"])
               for i in range(len(gradients) - 1)):
            raise ValueError(f"Updates nao crescentes em {directory.name}")
        if len(evaluations) != eval_episodes:
            raise ValueError(f"Esperados {eval_episodes} episodios finais em {directory.name}")
        if any(
            int(episodes[i]["steps"]) >= int(episodes[i + 1]["steps"])
            for i in range(len(episodes) - 1)
        ):
            raise ValueError(f"Passos nao crescentes em {directory.name}")

        if not stop_on_convergence:
            final = [c for c in checkpoints if c["checkpoint_type"] == "final"]
            if len(final) != 1 or int(final[0]["training_steps"]) != budget_steps:
                raise ValueError(f"Checkpoint final incorreto em {directory.name}")
            ids = [c["checkpoint_id"] for c in checkpoints]
            if len(ids) != len(set(ids)):
                raise ValueError(f"IDs de checkpoints repetidos em {directory.name}")
            for cp in checkpoints:
                records = [r for r in checkpoint_evals if r["checkpoint_id"] == cp["checkpoint_id"]]
                if len(records) != eval_episodes:
                    raise ValueError(f"Checkpoint {cp['checkpoint_id']} com "
                                     f"{len(records)} avaliacoes; esperado {eval_episodes}")
            if bool(summary["converged"]):
                first = [c for c in checkpoints if c["checkpoint_type"] == "first_convergence"]
                if len(first) != 1 or int(first[0]["training_steps"]) != int(summary["convergence"]["steps"]):
                    raise ValueError(f"Primeira convergencia nao foi preservada em {directory.name}")

        selected[key] = Run(
            algorithm, seed, directory, config, summary, episodes, evaluations,
            checkpoints, checkpoint_evals, gradients,
        )

    missing_keys = {(alg, seed) for alg in ALGORITHMS for seed in expected_seeds} - set(selected)
    if missing_keys:
        raise ValueError(
            "Execucoes ausentes: "
            + ", ".join(f"{alg}/seed={seed}" for alg, seed in sorted(missing_keys))
        )
    print(f"Orcamento: {budget_steps} | parada na convergencia: {stop_on_convergence} | "
          f"execucoes: {len(selected)} | ignoradas: {skipped}")
    return [selected[(alg, seed)] for alg in ALGORITHMS for seed in expected_seeds]


def extract_data(runs: list[Run], eval_episodes: int) -> dict[str, list[dict]]:
    rows_summary: list[dict] = []
    rows_training: list[dict] = []
    rows_evaluation: list[dict] = []
    rows_checkpoints: list[dict] = []
    rows_checkpoint_eval: list[dict] = []

    for run in runs:
        rewards = [float(row["reward"]) for row in run.evaluations]
        moving = [float(row["mean_reward_100"]) for row in run.episodes]
        steps = [int(row["steps"]) for row in run.episodes]
        peak_idx = max(range(len(moving)), key=moving.__getitem__)
        trough_idx = min(range(peak_idx, len(moving)), key=moving.__getitem__)
        s = run.summary
        duration = float(s["training_elapsed_seconds"])
        total_steps = int(s["training_steps"])
        convergence = s.get("convergence")
        first_step = int(convergence["steps"]) if convergence else None
        after = [moving[i] for i, t in enumerate(steps) if first_step is not None and t >= first_step]
        initial_conv_mean = float(convergence["mean_reward_100"]) if convergence else None
        cp_at_first = next(
            (cp for cp in run.checkpoints if cp["checkpoint_type"] == "first_convergence"),
            None,
        )
        first_eval = (
            float(cp_at_first["evaluation_reward_mean"]) if cp_at_first
            else mean(rewards) if convergence and s.get("stop_reason") == "convergence"
            else None
        )

        rows_summary.append({
            "algoritmo": run.algorithm.upper(),
            "baseline": (
                "nenhum" if run.algorithm == "pg"
                else "media acumulada dos retornos n-step"
            ),
            "semente": run.seed,
            "parada_na_convergencia": bool(run.config.get("stop_on_convergence", True)),
            "convergiu": bool(s["converged"]),  # Compatibilidade com CSV anterior.
            "convergiu_alguma_vez": bool(s["converged"]),
            "primeira_convergencia_passos": first_step if first_step is not None else "",
            "primeira_convergencia_episodio": convergence["episodes"] if convergence else "",
            "motivo_parada": s["stop_reason"],
            "passos_treinamento": total_steps,
            "episodios_treinamento": int(s["episodes_completed"]),
            "atualizacoes_gradiente": int(s["gradient_updates"]),
            "tempo_treino_segundos": duration,
            "recompensa_media_treino_ultimos100": moving[-1],
            "recompensa_media_avaliacao": mean(rewards),
            "desvio_padrao_avaliacao": stdev(rewards) if len(rewards) > 1 else 0.0,
            "avaliacoes_ge450": sum(r >= CONVERGENCE_THRESHOLD for r in rewards),
            "avaliacoes_eq500": sum(r == 500.0 for r in rewards),
            "pico_media_movel": moving[peak_idx],
            "passos_pico": steps[peak_idx],
            "menor_media_apos_pico": moving[trough_idx],
            "passos_menor_media_apos_pico": steps[trough_idx],
            "menor_media_movel_apos_primeira_convergencia": min(after) if after else "",
            "queda_maxima_relativa_a_primeira_convergencia": (
                max(0.0, initial_conv_mean - min(after)) if after else ""
            ),
            "fracao_episodios_apos_convergencia_acima_450": (
                sum(v >= CONVERGENCE_THRESHOLD for v in after) / len(after) if after else ""
            ),
            "avaliacao_primeira_convergencia": first_eval if first_eval is not None else "",
            "variacao_avaliacao_final_vs_primeira_convergencia": (
                mean(rewards) - first_eval if first_eval is not None else ""
            ),
            "tempo_por_1000_passos": duration / total_steps * 1000,
        })

        for episode in run.episodes:
            rows_training.append({
                "algoritmo": run.algorithm.upper(),
                "semente": run.seed,
                "execucao": run.directory.name,
                **episode,
            })
        for evaluation in run.evaluations:
            rows_evaluation.append({
                "algoritmo": run.algorithm.upper(),
                "semente_treinamento": run.seed,
                "execucao": run.directory.name,
                **evaluation,
            })
        for cp in run.checkpoints:
            rows_checkpoints.append({
                "algoritmo": run.algorithm.upper(),
                "semente": run.seed,
                "execucao": run.directory.name,
                **cp,
            })
        for evaluation in run.checkpoint_evaluations:
            rows_checkpoint_eval.append({
                "algoritmo": run.algorithm.upper(),
                "semente_treinamento": run.seed,
                "execucao": run.directory.name,
                **evaluation,
            })

    rows_gradients: list[dict] = []
    rows_diagnostics: list[dict] = []
    rows_gradient_summary: list[dict] = []
    for run in runs:
        grad_rows = run.gradients
        for g in grad_rows:
            augmented = {
                "algoritmo": run.algorithm.upper(),
                "semente": run.seed,
                "execucao": run.directory.name,
                **g,
            }
            rows_gradients.append(augmented)
            if g["paired_grad_var_no_baseline"]:
                rows_diagnostics.append(augmented)
        from statistics import median
        def as_values(col: str) -> list[float]:
            return [float(r[col]) for r in grad_rows if r.get(col, "") != ""]
        ratio_values = as_values("paired_grad_var_ratio")
        rows_gradient_summary.append({
            "algoritmo": run.algorithm.upper(),
            "semente": run.seed,
            "atualizacoes": len(grad_rows),
            "diagnosticos_pareados": len(ratio_values),
            "grad_rms_media": mean(as_values("grad_rms")),
            "grad_rms_mediana": median(as_values("grad_rms")),
            "grad_max_maximo": max(as_values("grad_max")),
            "grad_var_components_media": mean(as_values("grad_var_components")),
            "grad_temporal_var_window_media": (
                mean(as_values("grad_temporal_var_window"))
                if as_values("grad_temporal_var_window") else ""
            ),
            "paired_grad_var_ratio_mediana": median(ratio_values) if ratio_values else "",
            "paired_grad_var_ratio_fracao_menor_1": (
                sum(v < 1 for v in ratio_values) / len(ratio_values)
                if ratio_values else ""
            ),
            "paired_grad_var_no_baseline_media": (
                mean(as_values("paired_grad_var_no_baseline"))
                if ratio_values else ""
            ),
            "paired_grad_var_with_baseline_media": (
                mean(as_values("paired_grad_var_with_baseline"))
                if ratio_values else ""
            ),
        })

    rows_algorithms: list[dict] = []
    for algorithm in ALGORITHMS:
        matching = [r for r in rows_summary if r["algoritmo"].lower() == algorithm]
        eval_means = [r["recompensa_media_avaliacao"] for r in matching]
        converged = [r for r in matching if r["convergiu_alguma_vez"]]
        rows_algorithms.append({
            "algoritmo": algorithm.upper(),
            "baseline": (
                "nenhum" if algorithm == "pg"
                else "media acumulada dos retornos n-step"
            ),
            "quantidade_sementes": len(matching),
            "convergencias": len(converged),
            "taxa_convergencia": len(converged) / len(matching),
            "media_recompensa_avaliacao_por_semente": mean(eval_means),
            "dp_recompensa_avaliacao_entre_sementes": (
                stdev(eval_means) if len(eval_means) > 1 else 0.0
            ),
            "media_passos_convergencia_apenas_convergidos": (
                mean([r["primeira_convergencia_passos"] for r in converged])
                if converged else ""
            ),  # Coluna mantida para compatibilidade.
            "media_passos_primeira_convergencia_apenas_convergidos": (
                mean([r["primeira_convergencia_passos"] for r in converged])
                if converged else ""
            ),
            "avaliacoes_ge450_total": sum(r["avaliacoes_ge450"] for r in matching),
            "avaliacoes_eq500_total": sum(r["avaliacoes_eq500"] for r in matching),
            "episodios_avaliacao_total": len(matching) * eval_episodes,
        })

    return {
        "resumo_execucoes": rows_summary,
        "resumo_algoritmos": rows_algorithms,
        "episodios_treinamento": rows_training,
        "episodios_avaliacao": rows_evaluation,
        "resumo_checkpoints": rows_checkpoints,
        "avaliacoes_checkpoints": rows_checkpoint_eval,
        "resumo_gradientes": rows_gradient_summary,
        "gradientes_treinamento": rows_gradients,
        "gradientes_diagnosticos": rows_diagnostics,
    }


def configure_axes(title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)


def finish_plot(name: str, output_dir: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_dir / name, dpi=160, bbox_inches="tight")
    plt.close()


def plot_learning(runs: list[Run], budget_steps: int, output_dir: Path) -> None:
    plt.figure(figsize=(13, 6))
    for run in runs:
        x = [int(r["steps"]) for r in run.episodes]
        y = [float(r["mean_reward_100"]) for r in run.episodes]
        linestyle = "-" if run.algorithm == "pg" else "--"
        plt.plot(x, y, linestyle=linestyle, linewidth=1.9,
                 label=f"{ALGORITHM_LABELS[run.algorithm]} | semente {run.seed}")
    plt.axhline(CONVERGENCE_THRESHOLD, linestyle=":", linewidth=1.5,
                label="Critério de convergência: 450")
    plt.xlim(0, budget_steps)
    plt.ylim(0, 510)
    configure_axes("E03 — Policy Gradient n-step sem/com baseline no CartPole-v1",
                   "Interações reais com o ambiente", "Recompensa média dos últimos 100 episódios")
    plt.legend(fontsize=9, ncol=2)
    finish_plot("01_curvas_aprendizado.png", output_dir)


def plot_evaluation(runs: list[Run], output_dir: Path) -> None:
    groups = [[float(row["reward"]) for row in run.evaluations] for run in runs]
    labels = [f"{ALGORITHM_LABELS[run.algorithm]}\n{run.seed}" for run in runs]
    plt.figure(figsize=(12, 6))
    # Artistas originais Matplotlib: legenda coerente com as cores/estilos do boxplot.
    bp = plt.boxplot(groups, tick_labels=labels, showmeans=True, meanline=True)
    threshold_line = plt.axhline(
        CONVERGENCE_THRESHOLD, linestyle=":", linewidth=1.5,
        label="Referência: recompensa 450 (não é teste de convergência)",
    )
    configure_axes("E03 — Distribuição das recompensas na avaliação final",
                   "Algoritmo e semente de treinamento", "Recompensa por episódio")
    plt.ylim(0, 525)
    plt.legend(
        handles=[bp["medians"][0], bp["means"][0], bp["boxes"][0],
                 bp["whiskers"][0], bp["fliers"][0], threshold_line],
        labels=["Mediana", "Média", "Caixa: Q1–Q3 (50% central)",
                "Bigodes: até 1,5 × AIQ", "Valores atípicos", "Referência 450"],
        loc="lower left", ncol=2, fontsize=9,
    )
    finish_plot("02_distribuicao_avaliacao.png", output_dir)


def smooth_xy(rows: list[dict], column: str, bucket: int = 200) -> tuple[list[float], list[float]]:
    """Media em grupos de updates contiguos, ignorando valores ausentes."""
    xs, ys = [], []
    for offset in range(0, len(rows), bucket):
        group = [r for r in rows[offset:offset + bucket] if r.get(column, "") != ""]
        if not group:
            continue
        xs.append(mean(int(r["steps"]) for r in group))
        ys.append(mean(float(r[column]) for r in group))
    return xs, ys


def plot_gradient_metric(runs: list[Run], column: str, filename: str,
                         title: str, ylabel: str, out: Path,
                         use_log: bool = False, sparse: bool = False) -> None:
    plt.figure(figsize=(12, 6))
    for run in runs:
        if sparse:
            x = [int(r["steps"]) for r in run.gradients if r[column]]
            y = [float(r[column]) for r in run.gradients if r[column]]
        else:
            x, y = smooth_xy(run.gradients, column, bucket=200)
        # Mesmo estilo de linha adotado nos experimentos anteriores.
        plt.plot(x, y, linewidth=1.7,
                 linestyle="-" if run.algorithm == "pg" else "--",
                 label=f"{ALGORITHM_LABELS[run.algorithm]} | semente {run.seed}")
    if use_log:
        plt.yscale("log")
    configure_axes(title, "Interações reais com o ambiente", ylabel)
    plt.legend(ncol=2, fontsize=9)
    finish_plot(filename, out)


def plot_paired_diagnostics(runs: list[Run], out: Path) -> None:
    plt.figure(figsize=(12, 6))
    for run in runs:
        records = [r for r in run.gradients if r["paired_grad_var_ratio"]]
        x = [int(r["steps"]) for r in records]
        y = [float(r["paired_grad_var_ratio"]) for r in records]
        plt.plot(x, y, linestyle="-" if run.algorithm == "pg" else "--",
                 linewidth=1.15, alpha=0.6,
                 label=f"{ALGORITHM_LABELS[run.algorithm]} | semente {run.seed}")
    plt.axhline(1, color="black", linewidth=1.2, linestyle=":",
                label="Razão 1 (sem mudança de dispersão)")
    plt.yscale("log")
    configure_axes(
        "E03 — Diagnóstico pareado: dispersão com baseline / sem baseline",
        "Interações reais com o ambiente",
        "Razão das dispersões de gradientes individuais no mesmo minibatch",
    )
    plt.legend(ncol=2, fontsize=9)
    finish_plot("05_diagnostico_pareado.png", out)

def plot_checkpoint_evaluations(
    runs: list[Run], budget_steps: int, output_dir: Path,
) -> None:
    plt.figure(figsize=(13, 6))
    convergence_label_used = False
    for run in runs:
        fixed = sorted(
            [c for c in run.checkpoints if c["checkpoint_type"] in ("periodic", "final")],
            key=lambda c: int(c["training_steps"]),
        )
        line, = plt.plot(
            [int(c["training_steps"]) for c in fixed],
            [float(c["evaluation_reward_mean"]) for c in fixed],
            marker="o", linestyle="-" if run.algorithm == "pg" else "--",
            linewidth=1.7, label=f"{ALGORITHM_LABELS[run.algorithm]} | semente {run.seed}",
        )
        first = next(
            (c for c in run.checkpoints if c["checkpoint_type"] == "first_convergence"),
            None,
        )
        if first is not None:
            plt.scatter(
                int(first["training_steps"]), float(first["evaluation_reward_mean"]),
                color=line.get_color(), marker="X", s=110, zorder=5,
                label="Avaliação na 1ª convergência" if not convergence_label_used else "_nolegend_",
            )
            convergence_label_used = True
    plt.axhline(CONVERGENCE_THRESHOLD, linestyle=":", linewidth=1.3,
                label="Referência: recompensa 450")
    plt.xlim(0, budget_steps)
    plt.ylim(0, 510)
    configure_axes("E03 — Avaliação independente ao longo do treinamento",
                   "Interações de treinamento no checkpoint", "Recompensa média de avaliação")
    plt.legend(fontsize=9, ncol=2)
    finish_plot("07_avaliacoes_por_checkpoint.png", output_dir)


def plot_after_convergence(runs: list[Run], output_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    plotted = 0
    for run in runs:
        first = run.summary.get("convergence")
        if not first:
            continue
        initial_steps = int(first["steps"])
        x = [int(row["steps"]) - initial_steps for row in run.episodes
             if int(row["steps"]) >= initial_steps]
        y = [float(row["mean_reward_100"]) for row in run.episodes
             if int(row["steps"]) >= initial_steps]
        plt.plot(x, y, linewidth=1.9,
                 linestyle="-" if run.algorithm == "pg" else "--",
                 label=f"{ALGORITHM_LABELS[run.algorithm]} | semente {run.seed}")
        plotted += 1
    plt.axhline(CONVERGENCE_THRESHOLD, linestyle=":", linewidth=1.4,
                label="Critério de convergência: 450")
    plt.ylim(0, 510)
    configure_axes("E03 — Recompensa após a primeira convergência (somente quem convergiu)",
                   "Interações decorridas desde a primeira convergência",
                   "Recompensa média dos últimos 100 episódios")
    plt.legend(fontsize=9, ncol=2)
    if plotted:
        finish_plot("08_apos_primeira_convergencia.png", output_dir)
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisa as execucoes completas do E03")
    parser.add_argument("--steps", type=int, default=DEFAULT_BUDGET_STEPS)
    parser.add_argument(
        "--stop-on-convergence", action=argparse.BooleanOptionalAction, default=True,
        help="Filtra execucoes com parada na convergencia (padrao: ligado).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_episodes <= 0:
        parser.error("--steps e --eval-episodes devem ser positivos")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contem duplicatas")

    mode = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    output_dir = OUTPUT_BASE_DIR / f"steps_{args.steps}" / mode
    runs = load_runs(args.steps, args.stop_on_convergence, tuple(args.seeds), args.eval_episodes)
    data = extract_data(runs, args.eval_episodes)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("resumo_execucoes", "resumo_algoritmos", "episodios_treinamento",
                 "episodios_avaliacao", "resumo_gradientes", "gradientes_treinamento",
                 "gradientes_diagnosticos"):
        write_csv(output_dir / f"{name}.csv", data[name])
    if not args.stop_on_convergence:
        write_csv(output_dir / "resumo_checkpoints.csv", data["resumo_checkpoints"])
        write_csv(output_dir / "avaliacoes_checkpoints.csv", data["avaliacoes_checkpoints"])

    metadata = {
        "experimento": "E03",
        "ambiente": "CartPole-v1",
        "baseline_do_livro": "media acumulada de retornos n-step; sem critico V(s)",
        "criterio_selecao": {
            "max_training_steps": args.steps,
            "stop_on_convergence": args.stop_on_convergence,
            "evaluation_episodes": args.eval_episodes,
            "seeds": args.seeds,
        },
        "pasta_origem": str(RESULTS_DIR / mode),
        "pasta_destino": str(output_dir),
        "limiar_media_movel_100_episodios": CONVERGENCE_THRESHOLD,
        "execucoes": [
            {
                "algoritmo": r.algorithm,
                "baseline": r.summary.get("baseline"),
                "semente": r.seed,
                "diretorio": r.directory.name,
                "passos_executados": r.summary["training_steps"],
                "convergiu_alguma_vez": bool(r.summary["converged"]),
                "primeira_convergencia": r.summary.get("convergence"),
                "politica_avaliacao": r.summary.get("evaluation", {}).get("policy"),
            }
            for r in runs
        ],
        "resumo_algoritmos": data["resumo_algoritmos"],
        "resumo_gradientes": data["resumo_gradientes"],
        "notas": [
            "Orcamento configurado e modo de parada sao filtrados separadamente.",
            "A primeira convergencia e um marco, nao uma garantia de estabilidade posterior.",
            "A media de avaliacao final pode diferir da media movel de treinamento.",
            "Avaliacao dos checkpoints usa 20 episodios (ou --eval-episodes), "
            "com as mesmas sementes em cada checkpoint.",
            "O desvio padrao dos episodios de avaliacao e amostral (n-1).",
            "Convergencia e definida pela media dos ultimos 100 episodios de TREINO >= 450.",
            "Checkpoints intermediarios registram pesos; nao permitem retomada integral do treino.",
            "Baseline do livro: media acumulada de exp.reward (retornos n-step), incluindo a amostra atual; nao V(s).",
            "grad_var_components nao estima variancia entre amostras; paired_grad_var compara minibatch fixo, n-step correlacionado.",
        ],
    }
    with (output_dir / "resumo_analise.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    plot_learning(runs, args.steps, output_dir)
    plot_evaluation(runs, output_dir)
    plot_gradient_metric(
        runs, "grad_rms", "03_gradientes_magnitude.png",
        "E03 — Magnitude RMS do gradiente de política",
        "RMS dos componentes do gradiente", output_dir,
    )
    plot_gradient_metric(
        runs, "grad_var_components", "04_variancia_componentes.png",
        "E03 — Variância entre componentes do gradiente (métrica do livro)",
        "Variância entre coordenadas do gradiente", output_dir,
        use_log=True,
    )
    plot_paired_diagnostics(runs, output_dir)
    plot_gradient_metric(
        runs, "grad_temporal_var_window", "06_variacao_temporal.png",
        "E03 — Variação temporal dos gradientes em janela de updates",
        "Variância temporal média por coordenada", output_dir,
        use_log=True, sparse=True,
    )
    if not args.stop_on_convergence:
        plot_checkpoint_evaluations(runs, args.steps, output_dir)
        plot_after_convergence(runs, output_dir)

    print(f"\nArtefatos gerados em: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}")
    for row in data["resumo_algoritmos"]:
        print(f"{row['algoritmo']}: {row['convergencias']}/{row['quantidade_sementes']} "
              f"convergiram | avaliacao FINAL media={row['media_recompensa_avaliacao_por_semente']:.2f}")


if __name__ == "__main__":
    main()

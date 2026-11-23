#!/usr/bin/env python3
"""E03 — Policy Gradient com e sem baseline no CartPole-v1 (Capitulo 12).

Executar na raiz do projeto:
  python experimentos-unidade-3/e03_cartpole.py --algorithm both \
    --seeds 42 123 2026 --steps 200000 --eval-episodes 20 \
    --no-stop-on-convergence

Referencia: Chapter12/01_cartpole_pg.py (sem editar o codigo do livro).
Arquitetura e hiperparametros originais: PGN, GAMMA=.99, LR=.001,
ENTROPY_BETA=.01, BATCH_SIZE=8, REWARD_STEPS=10, Adam.
Baseline do E03: media ACUMULADA dos retornos n-step (exp.reward),
incluindo a experiencia atual. NAO e a centragem por episodio do E02.

Convergencia: 100 episodios, media dos ultimos 100 >=450. Parada opcional.
Modo sem parada: checkpoints/avaliacoes a cada 50 mil env.step e no marco
da primeira convergencia (sem influenciar RNG de treinamento).

Gradientes: grad_var_components = var entre coordenadas do vetor policy-grad
(do livro); grad_temporal_var_window = dispersao de gradientes de updates
consecutivos sob pesos/politicas em evolucao; paired_grad_var_* = dispersao
entre gradientes de amostras de um mesmo minibatch, para duas centragens
contrafactuais calculadas NO MESMO theta e sobre os mesmos dados.
Os itens do minibatch sao correlacionados (transicoes n-step consecutivas),
logo esta ultima nao e uma estimativa de amostragem IID.

Pastas: results/e03/completo/{stop_on_convergence,no_stop_on_convergence}/<run>/.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
import json
import random
from collections import deque
import runpy
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import ptan
import torch
from torch import nn, optim
from torch.nn import functional as F

UNIT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = UNIT_DIR / "Chapter12"
RESULTS_DIR = UNIT_DIR / "results" / "e03" / "completo"
ENV_ID = "CartPole-v1"
CONVERGENCE_WINDOW = 100
CONVERGENCE_REWARD = 450.0
EVALUATION_SEED_BASE = 100_000
EVALUATION_COLUMNS = ("episode", "seed", "reward", "steps", "terminated", "truncated")
CHECKPOINT_COLUMNS = (
    "checkpoint_id", "checkpoint_type", "training_steps", "gradient_updates",
    "model_path", "evaluation_reward_mean", "evaluation_reward_std",
    "evaluation_reward_min", "evaluation_reward_max",
)
CHECKPOINT_EVAL_COLUMNS = (
    "checkpoint_id", "checkpoint_type", "training_steps", *EVALUATION_COLUMNS,
)


class StepCounter(gym.Wrapper):
    """Contabiliza env.step() reais, inclusive quando o PTAN emite itens extras."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.total_steps = 0

    def step(self, action):
        result = self.env.step(action)
        self.total_steps += 1
        return result


def read_reference(filename: str) -> dict[str, Any]:
    path = CHAPTER_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Codigo original nao encontrado: {path}")
    return runpy.run_path(str(path), run_name="__e03_reference__")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


GRADIENT_COLUMNS = (
    "steps", "update", "experience_samples", "loss_policy", "loss_entropy",
    "loss_total", "entropy", "kl_old_new", "grad_rms", "grad_max",
    "grad_var_components", "grad_temporal_var_window", "mean_batch_reward",
    "mean_batch_baseline", "mean_batch_scale", "paired_grad_var_no_baseline",
    "paired_grad_var_with_baseline", "paired_grad_var_ratio",
)


def paired_gradient_diagnostic(
    net: nn.Module, selected_logp: torch.Tensor,
    rewards: np.ndarray, baselines: np.ndarray,
) -> tuple[float, float, float]:
    """Dispersao dos gradientes de amostras no MESMO modelo e minibatch.

    Calcula o gradiente de -log_pi(a|s) para cada amostra; depois aplica
    contrafactualmente os escaladores r e (r-b) nas mesmas observacoes.
    Nao altera os pesos nem os gradientes .grad do otimizador.
    """
    params = tuple(net.parameters())
    per_sample = []
    for logp in selected_logp:
        chunks = torch.autograd.grad(-logp, params, retain_graph=True)
        per_sample.append(torch.cat([g.detach().flatten() for g in chunks]).cpu().numpy())
    raw = np.stack(per_sample).astype(np.float64)
    g_no = raw * rewards[:, None]
    g_with = raw * (rewards - baselines)[:, None]
    var_no = float(np.var(g_no, axis=0, ddof=1).mean())
    var_with = float(np.var(g_with, axis=0, ddof=1).mean())
    ratio = float(var_with / var_no) if var_no > 0 else float("nan")
    return var_no, var_with, ratio


def train_pg(
    env: StepCounter,
    algorithm: str,
    max_steps: int,
    on_episode: Callable[..., bool],
    on_step: Callable[..., None],
    on_gradient: Callable[[dict], None],
    gradient_diagnostic_every: int,
    gradient_window: int,
) -> tuple[nn.Module, int, dict]:
    """Adapta Chapter12/01_cartpole_pg.py conservando o update do livro."""
    ref = read_reference("01_cartpole_pg.py")
    net = ref["PGN"](env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(
        net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True,
    )
    source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=ref["GAMMA"], steps_count=ref["REWARD_STEPS"],
    )
    optimizer = optim.Adam(net.parameters(), lr=ref["LEARNING_RATE"])
    use_baseline = algorithm == "pg_baseline"
    rewards_sum = 0.0
    experience_samples = 0
    updates = 0
    last_loss: float | None = None
    gradient_history = deque(maxlen=gradient_window)
    states: list = []
    actions: list[int] = []
    scales: list[float] = []
    raw_rewards: list[float] = []
    baselines: list[float] = []
    source_it = iter(source)

    while env.total_steps < max_steps:
        exp = next(source_it)
        experience_samples += 1
        reward_nstep = float(exp.reward)
        rewards_sum += reward_nstep
        running_baseline = rewards_sum / experience_samples
        states.append(exp.state)
        actions.append(int(exp.action))
        raw_rewards.append(reward_nstep)
        baselines.append(running_baseline)
        scales.append(reward_nstep - running_baseline if use_baseline else reward_nstep)

        stopped = False
        for reward in source.pop_total_rewards():
            if on_episode(env.total_steps, float(reward), None, last_loss, net, updates):
                stopped = True
        if stopped:
            break

        if len(states) >= ref["BATCH_SIZE"]:
            states_t = torch.as_tensor(np.asarray(states, dtype=np.float32))
            actions_t = torch.as_tensor(actions, dtype=torch.long)
            scales_t = torch.as_tensor(scales, dtype=torch.float32)

            optimizer.zero_grad()
            logits = net(states_t)
            log_probs = F.log_softmax(logits, dim=1)
            selected = log_probs[torch.arange(len(states)), actions_t]
            loss_policy = -(scales_t * selected).mean()

            updates += 1
            grad_no = grad_with = grad_ratio = float("nan")
            diagnostic_due = updates % gradient_diagnostic_every == 0
            if diagnostic_due:
                grad_no, grad_with, grad_ratio = paired_gradient_diagnostic(
                    net, selected, np.asarray(raw_rewards, dtype=np.float64),
                    np.asarray(baselines, dtype=np.float64),
                )

            # O livro registra o gradiente da perda de politica ANTES de
            # acrescentar o termo de entropia.
            loss_policy.backward(retain_graph=True)
            grad_vector = np.concatenate([
                p.grad.detach().cpu().numpy().ravel()
                for p in net.parameters() if p.grad is not None
            ]).astype(np.float64)
            grad_history_prev = len(gradient_history)
            gradient_history.append(grad_vector)
            temporal_var = (
                float(np.var(np.stack(gradient_history), axis=0, ddof=1).mean())
                if diagnostic_due and grad_history_prev > 0 else float("nan")
            )

            probs = F.softmax(logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            loss_entropy = -ref["ENTROPY_BETA"] * entropy
            loss_entropy.backward()
            optimizer.step()

            with torch.no_grad():
                new_probs = F.softmax(net(states_t), dim=1)
                kl = (probs.detach() * (
                    probs.detach().clamp_min(1e-10).log()
                    - new_probs.clamp_min(1e-10).log()
                )).sum(dim=1).mean()

            loss_total = float((loss_policy + loss_entropy).item())
            last_loss = loss_total
            on_gradient({
                "steps": env.total_steps,
                "update": updates,
                "experience_samples": experience_samples,
                "loss_policy": float(loss_policy.item()),
                "loss_entropy": float(loss_entropy.item()),
                "loss_total": loss_total,
                "entropy": float(entropy.item()),
                "kl_old_new": float(kl.item()),
                "grad_rms": float(np.sqrt(np.mean(grad_vector ** 2))),
                "grad_max": float(np.max(np.abs(grad_vector))),
                "grad_var_components": float(np.var(grad_vector)),
                "grad_temporal_var_window": temporal_var,
                "mean_batch_reward": float(np.mean(raw_rewards)),
                "mean_batch_baseline": float(np.mean(baselines)),
                "mean_batch_scale": float(np.mean(scales)),
                "paired_grad_var_no_baseline": grad_no,
                "paired_grad_var_with_baseline": grad_with,
                "paired_grad_var_ratio": grad_ratio,
            })
            states.clear()
            actions.clear()
            scales.clear()
            raw_rewards.clear()
            baselines.clear()
        on_step(env.total_steps, net, updates)

    return net, updates, {
        "train_params": {name: ref[name] for name in (
            "GAMMA", "LEARNING_RATE", "ENTROPY_BETA", "BATCH_SIZE", "REWARD_STEPS",
        )},
        "baseline": (
            "running_mean_nstep_returns_including_current_sample" if use_baseline else "none"
        ),
        "gradient_diagnostic_every_updates": gradient_diagnostic_every,
        "gradient_temporal_window_updates": gradient_window,
        "pending_minibatch_samples": len(states),
        "experience_samples": experience_samples,
        "note": "Minibatch incompleto ao fim do orcamento nao gera atualizacao.",
    }


def evaluate(
    net: nn.Module,
    algorithm: str,
    episodes: int,
    reinforce_eval_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Avalia politica congelada: sorteio softmax (padrao) ou argmax.

    As duas variantes PG usam exatamente o mesmo protocolo.
    RNG NumPy local por episodio: nao interfere no treino.
    """
    env = gym.make(ENV_ID)
    was_training = net.training
    net.eval()
    rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for i in range(episodes):
                eval_seed = EVALUATION_SEED_BASE + i
                obs, _ = env.reset(seed=eval_seed)
                rng = np.random.default_rng(eval_seed)
                terminated = truncated = False
                total_reward = 0.0
                n_steps = 0
                while not (terminated or truncated):
                    obs_t = torch.as_tensor(
                        np.asarray(obs, dtype=np.float32)
                    ).unsqueeze(0)
                    logits_or_q = net(obs_t)[0]
                    if reinforce_eval_mode == "greedy":
                        action = int(torch.argmax(logits_or_q).item())
                    else:
                        probs = F.softmax(logits_or_q, dim=-1).cpu().numpy()
                        action = int(rng.choice(len(probs), p=probs))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    n_steps += 1
                rows.append({
                    "episode": i + 1,
                    "seed": eval_seed,
                    "reward": total_reward,
                    "steps": n_steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })
    finally:
        env.close()
        if was_training:
            net.train()

    totals = [r["reward"] for r in rows]
    lengths = [r["steps"] for r in rows]
    return {
        "episodes": episodes,
        "policy": reinforce_eval_mode,
        "seed_base": EVALUATION_SEED_BASE,
        "reward_mean": float(np.mean(totals)),
        "reward_std": float(np.std(totals, ddof=1)) if episodes > 1 else 0.0,
        "reward_min": float(min(totals)),
        "reward_max": float(max(totals)),
        "episode_length_mean": float(np.mean(lengths)),
    }, rows


def run_one(
    algorithm: str,
    seed: int,
    max_steps: int,
    eval_episodes: int,
    reinforce_eval_mode: str,
    report_every: int,
    stop_on_convergence: bool,
    eval_every: int,
    gradient_diagnostic_every: int,
    gradient_window: int,
) -> dict[str, Any]:
    seed_everything(seed)
    env = StepCounter(gym.make(ENV_ID))
    env.reset(seed=seed)
    env.action_space.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    mode = "stop" if stop_on_convergence else "nostop"
    run_id = f"{algorithm}_seed{seed}_steps{max_steps}_{mode}_{timestamp}"
    result_mode = "stop_on_convergence" if stop_on_convergence else "no_stop_on_convergence"
    run_dir = RESULTS_DIR / result_mode / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "experiment": "E03",
        "algorithm": algorithm,
        "environment": ENV_ID,
        "seed": seed,
        "max_training_steps": max_steps,
        "stop_on_convergence": stop_on_convergence,
        "evaluation_interval_steps": None if stop_on_convergence else eval_every,
        "min_episodes_for_convergence": CONVERGENCE_WINDOW,
        "convergence_mean_reward": CONVERGENCE_REWARD,
        "evaluation_episodes": eval_episodes,
        "gradient_diagnostic_every_updates": gradient_diagnostic_every,
        "gradient_temporal_window_updates": gradient_window,
        "evaluation_seed_base": EVALUATION_SEED_BASE,
        "reinforce_evaluation_mode": reinforce_eval_mode,
        "baseline": (
            "none" if algorithm == "pg"
            else "running_mean_nstep_returns_including_current_sample"
        ),
    }
    write_json(run_dir / "config.json", config)

    print(f"\nE03 | {algorithm} | seed={seed} | max_steps={max_steps} | "
          f"stop_on_convergence={stop_on_convergence}")
    print(f"Saida: {run_dir}")
    rewards: list[float] = []
    convergence: dict | None = None
    checkpoint_rows: list[dict] = []
    checkpoint_evaluations: list[dict] = []
    milestones_done: set[int] = set()
    milestones = set(range(eval_every, max_steps, eval_every)) | {max_steps}
    checkpoint_seconds_total = 0.0
    start = time.perf_counter()

    def save_checkpoint(kind: str, steps: int, net: nn.Module, updates: int) -> None:
        """Guarda pesos e avaliacao pontual, sem interferir no treino."""
        nonlocal checkpoint_seconds_total
        checkpoint_start = time.perf_counter()
        checkpoint_id = f"{kind}_steps{steps}"
        relative_path = Path("checkpoints") / f"{checkpoint_id}.pt"
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": net.state_dict(),
            "experiment": "E03",
            "algorithm": algorithm,
            "seed": seed,
            "checkpoint_type": kind,
            "training_steps": steps,
            "gradient_updates": updates,
            "note": "Checkpoint de pesos; nao inclui estado integral para retomar treino.",
        }, path)
        result, rows = evaluate(net, algorithm, eval_episodes, reinforce_eval_mode)
        checkpoint_rows.append({
            "checkpoint_id": checkpoint_id,
            "checkpoint_type": kind,
            "training_steps": steps,
            "gradient_updates": updates,
            "model_path": str(relative_path),
            "evaluation_reward_mean": result["reward_mean"],
            "evaluation_reward_std": result["reward_std"],
            "evaluation_reward_min": result["reward_min"],
            "evaluation_reward_max": result["reward_max"],
        })
        for row in rows:
            checkpoint_evaluations.append({
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": kind,
                "training_steps": steps,
                **row,
            })
        checkpoint_seconds_total += time.perf_counter() - checkpoint_start
        print(f"  Checkpoint {checkpoint_id}: "
              f"avaliacao={result['reward_mean']:.2f} +/- {result['reward_std']:.2f}")

    with ExitStack() as stack:
        output = stack.enter_context((run_dir / "episodes.csv").open(
            "w", newline="", encoding="utf-8"))
        gradient_output = stack.enter_context((run_dir / "gradients.csv").open(
            "w", newline="", encoding="utf-8"))
        grad_writer = csv.DictWriter(gradient_output, fieldnames=GRADIENT_COLUMNS)
        grad_writer.writeheader()

        def record_gradient(row: dict) -> None:
            grad_writer.writerow({k: ("" if not np.isfinite(v) else v) if isinstance(v, float) else v
                                  for k, v in row.items()})
            if row["update"] % 100 == 0:
                gradient_output.flush()

        fields = ["steps", "episode", "reward", "mean_reward_100", "epsilon", "last_loss"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        def record_episode(
            step: int, reward: float, epsilon: float | None,
            last_loss: float | None, net: nn.Module, updates: int,
        ) -> bool:
            nonlocal convergence
            rewards.append(reward)
            episode = len(rewards)
            avg = float(np.mean(rewards[-CONVERGENCE_WINDOW:]))
            writer.writerow({
                "steps": step,
                "episode": episode,
                "reward": reward,
                "mean_reward_100": avg,
                "epsilon": "" if epsilon is None else epsilon,
                "last_loss": "" if last_loss is None else last_loss,
            })
            output.flush()
            reached = episode >= CONVERGENCE_WINDOW and avg >= CONVERGENCE_REWARD
            first_reached = reached and convergence is None
            if first_reached:
                convergence = {
                    "steps": step,
                    "episodes": episode,
                    "mean_reward_100": avg,
                }
                if not stop_on_convergence:
                    save_checkpoint("first_convergence", step, net, updates)
            if episode == 1 or episode % report_every == 0 or first_reached:
                print(f"Passos: {step:7d} | episodios: {episode:4d} | "
                      f"reward: {reward:6.1f} | mean_100: {avg:6.2f}"
                      + (" | PRIMEIRA CONVERGENCIA" if first_reached else ""))
            return bool(reached and stop_on_convergence)

        def record_step(step: int, net: nn.Module, updates: int) -> None:
            if stop_on_convergence or step not in milestones or step in milestones_done:
                return
            kind = "final" if step == max_steps else "periodic"
            save_checkpoint(kind, step, net, updates)
            milestones_done.add(step)

        try:
            net, updates, extras = train_pg(
                env, algorithm, max_steps, record_episode, record_step,
                record_gradient, gradient_diagnostic_every, gradient_window,
            )
        finally:
            env.close()

    training_wall_seconds = time.perf_counter() - start
    checkpoint_seconds_during_training = checkpoint_seconds_total
    train_seconds = training_wall_seconds - checkpoint_seconds_during_training
    training_steps = env.total_steps
    if stop_on_convergence:
        evaluation, final_eval_rows = evaluate(
            net, algorithm, eval_episodes, reinforce_eval_mode,
        )
    else:
        if training_steps != max_steps:
            raise RuntimeError(f"Treinamento terminou em {training_steps}, esperado {max_steps}")
        # Quando o ultimo passo produz mais de uma experiencia PTAN, o marco
        # final pode ja ter sido registrado. Nao o avalie novamente.
        if max_steps not in milestones_done:
            save_checkpoint("final", training_steps, net, updates)
            milestones_done.add(training_steps)
        final_id = f"final_steps{training_steps}"
        final_eval_rows = [
            {key: row[key] for key in EVALUATION_COLUMNS}
            for row in checkpoint_evaluations if row["checkpoint_id"] == final_id
        ]
        if len(final_eval_rows) != eval_episodes:
            raise RuntimeError("Avaliacao do checkpoint final incompleta")
        final_rewards = [row["reward"] for row in final_eval_rows]
        final_lengths = [row["steps"] for row in final_eval_rows]
        evaluation = {
            "episodes": eval_episodes,
            "policy": reinforce_eval_mode,
            "seed_base": EVALUATION_SEED_BASE,
            "reward_mean": float(np.mean(final_rewards)),
            "reward_std": float(np.std(final_rewards, ddof=1)) if eval_episodes > 1 else 0.0,
            "reward_min": float(min(final_rewards)),
            "reward_max": float(max(final_rewards)),
            "episode_length_mean": float(np.mean(final_lengths)),
        }
        write_csv(run_dir / "checkpoints.csv", CHECKPOINT_COLUMNS, checkpoint_rows)
        write_csv(run_dir / "checkpoint_evaluations.csv", CHECKPOINT_EVAL_COLUMNS,
                  checkpoint_evaluations)

    write_csv(run_dir / "evaluation.csv", EVALUATION_COLUMNS, final_eval_rows)
    eval_and_write_seconds = time.perf_counter() - start - training_wall_seconds
    summary = {
        **config,
        "stop_reason": (
            "convergence" if (stop_on_convergence and convergence is not None)
            else "step_budget"
        ),
        "converged": convergence is not None,
        "convergence": convergence,  # PRIMEIRA convergencia, mesmo se treinou ate o fim.
        "training_steps": training_steps,
        "episodes_completed": len(rewards),
        "gradient_updates": updates,
        "training_reward_mean_last_100": (
            float(np.mean(rewards[-CONVERGENCE_WINDOW:])) if rewards else None
        ),
        "training_reward_max": float(max(rewards)) if rewards else None,
        "training_elapsed_seconds": round(train_seconds, 3),
        "evaluation_elapsed_seconds": round(checkpoint_seconds_during_training + eval_and_write_seconds, 3),
        "checkpoint_evaluations_elapsed_seconds": round(checkpoint_seconds_total, 3),
        "evaluation": evaluation,
        "checkpoint_evaluations": checkpoint_rows if not stop_on_convergence else [],
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "ptan": getattr(ptan, "__version__", "unknown"),
            "numpy": np.__version__,
        },
        "reference": {
            "pg": "Chapter12/01_cartpole_pg.py",
            "pg_baseline": "Chapter12/01_cartpole_pg.py",
        }[algorithm],
        "details": extras,
    }
    write_json(run_dir / "summary.json", summary)
    torch.save({"model_state_dict": net.state_dict(), "summary": summary}, run_dir / "model.pt")
    print(f"Finalizado | motivo={summary['stop_reason']} | "
          f"convergiu={'sim' if convergence is not None else 'nao'} | "
          f"passos={training_steps} | episodios={len(rewards)} | "
          f"updates={updates} | media_treino={summary['training_reward_mean_last_100']} | "
          f"media_avaliacao={evaluation['reward_mean']:.2f} "
          f"(+/- {evaluation['reward_std']:.2f})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E03 CartPole — Policy Gradient cap. 12 sem/com baseline",
    )
    parser.add_argument("--algorithm", choices=("pg", "pg_baseline", "both"), required=True)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="Uma semente (padrao: 42)")
    seeds.add_argument("--seeds", nargs="+", type=int, help="Lista de sementes")
    parser.add_argument("--steps", type=int, default=200_000, help="Limite de env.step por treino")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--stop-on-convergence", action=argparse.BooleanOptionalAction, default=True,
        help="Parar na primeira media movel >= 450 (padrao: ligado). "
             "Com --no-stop-on-convergence executa o orcamento integral.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=50_000,
        help="Intervalo das avaliacoes intermediarias, somente sem parada (padrao: 50000)",
    )
    parser.add_argument(
        "--policy-eval", "--reinforce-eval", dest="policy_eval",
        choices=("stochastic", "greedy"), default="stochastic",
        help="Avaliacao das duas politicas PG (padrao: stochastic)",
    )
    parser.add_argument("--gradient-diagnostic-every", type=int, default=200,
                        help="A cada N updates mede a dispersao de gradientes individuais (padrao: 200)")
    parser.add_argument("--gradient-window", type=int, default=50,
                        help="Janela de updates para a variacao temporal (padrao: 50)")
    parser.add_argument("--report-every", type=int, default=20)
    args = parser.parse_args()
    if min(args.steps, args.eval_episodes, args.eval_every, args.report_every,
           args.gradient_diagnostic_every) <= 0 or args.gradient_window < 2:
        parser.error("Parametros numericos devem ser positivos; --gradient-window >=2")

    run_seeds = args.seeds if args.seeds is not None else [
        args.seed if args.seed is not None else 42
    ]
    if len(set(run_seeds)) != len(run_seeds):
        parser.error("--seeds nao pode conter sementes duplicadas")
    algorithms = ("pg", "pg_baseline") if args.algorithm == "both" else (args.algorithm,)
    for seed in run_seeds:
        for algorithm in algorithms:
            run_one(
                algorithm=algorithm,
                seed=seed,
                max_steps=args.steps,
                eval_episodes=args.eval_episodes,
                reinforce_eval_mode=args.policy_eval,
                report_every=args.report_every,
                stop_on_convergence=args.stop_on_convergence,
                eval_every=args.eval_every,
                gradient_diagnostic_every=args.gradient_diagnostic_every,
                gradient_window=args.gradient_window,
            )
    print("\nE03: todas as execucoes solicitadas foram finalizadas.")


if __name__ == "__main__":
    main()

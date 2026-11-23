#!/usr/bin/env python3
"""E02 — REINFORCE convencional x REINFORCE com baseline no CartPole-v1.

Local: experimentos-unidade-3/e02_cartpole.py
Execute a partir da raiz do projeto, com o venv da Unidade 3 ativo:

  python experimentos-unidade-3/e02_cartpole.py --algorithm both \
      --seeds 42 123 2026 --steps 200000 --eval-episodes 20 \
      --no-stop-on-convergence

  # Protocolo com parada antecipada (habilitada por padrao):
  python experimentos-unidade-3/e02_cartpole.py --algorithm both \
      --seeds 42 123 2026 --steps 200000 --eval-episodes 20

Implementacoes de referencia (codigo original sem alteracoes):
  Chapter11/02_cartpole_reinforce.py          — sem baseline
  Chapter11/03_cartpole_reinforce_baseline.py — media dos retornos G_t

IMPORTANTE: o baseline adotado no livro subtrai a media dos retornos
DESCONTADOS do mesmo episodio, nao um critico de valor V(s) aprendido.
Ambas as variantes usam a mesma arquitetura, gamma, LR e tamanho de
minibatch do livro; a unica diferenca no update e a centragem de G_t.

Convergencia: pelo menos 100 episodios e media movel dos ultimos 100 >=450.
No modo sem parada, primeira convergencia e apenas marco: salva e avalia
checkpoint, e continua ate o orcamento. Avaliacoes independentes e
checkpoints a cada 50.000 passos, por padrao, alem do final.

Saidas: results/e02/completo/<modo>/<execucao>/
  config.json, episodes.csv, evaluation.csv, summary.json, model.pt
  e, quando sem parada:
  checkpoints.csv, checkpoint_evaluations.csv, checkpoints/*.pt

Cada passo e uma chamada real a env.step(). A avaliacao usa um ambiente
e um RNG local, sem alterar o estado aleatorio do treinamento.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
CHAPTER_DIR = UNIT_DIR / "Chapter11"
RESULTS_DIR = UNIT_DIR / "results" / "e02" / "completo"
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
    return runpy.run_path(str(path), run_name="__e02_reference__")


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


def train_reinforce(
    env: StepCounter,
    algorithm: str,
    max_steps: int,
    on_episode: Callable[..., bool],
    on_step: Callable[..., None],
) -> tuple[nn.Module, int, dict]:
    """Replica REINFORCE do cap. 11; unica diferenca e o retorno centrado.

    O baseline do exemplo do livro e a media dos retornos descontados
    do PROPRIO episodio, nao uma rede de valor nem uma media por minibatch.
    No algoritmo com baseline, calc_qvals() subtrai essa media de cada G_t.
    """
    reference_file = (
        "02_cartpole_reinforce.py" if algorithm == "reinforce"
        else "03_cartpole_reinforce_baseline.py"
    )
    ref = read_reference(reference_file)
    net = ref["PGN"](env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(
        net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True,
    )
    source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=ref["GAMMA"],
    )
    optimizer = optim.Adam(net.parameters(), lr=ref["LEARNING_RATE"])

    # Episodios inteiros; nenhum gradiente e calculado antes de completar
    # EPISODES_TO_TRAIN episodios, como nos dois exemplos originais.
    batch_states: list = []
    batch_actions: list[int] = []
    batch_qvals: list[float] = []
    cur_states: list = []
    cur_actions: list[int] = []
    cur_rewards: list[float] = []
    complete_batch_episodes = 0
    last_loss: float | None = None
    updates = 0
    stopped_at_convergence = False
    source_it = iter(source)

    while env.total_steps < max_steps:
        exp = next(source_it)
        cur_states.append(exp.state)
        cur_actions.append(int(exp.action))
        cur_rewards.append(float(exp.reward))

        if exp.last_state is None:
            batch_states.extend(cur_states)
            batch_actions.extend(cur_actions)
            batch_qvals.extend(ref["calc_qvals"](cur_rewards))
            cur_states.clear()
            cur_actions.clear()
            cur_rewards.clear()
            complete_batch_episodes += 1

        for reward in source.pop_total_rewards():
            if on_episode(env.total_steps, float(reward), None, last_loss, net, updates):
                stopped_at_convergence = True
        if stopped_at_convergence:
            # O episodio que dispara a parada nao precisa gerar um update,
            # seguindo o comportamento dos codigos do livro e do experimento anterior.
            break

        if complete_batch_episodes >= ref["EPISODES_TO_TRAIN"]:
            optimizer.zero_grad()
            states_t = torch.as_tensor(np.asarray(batch_states))
            actions_t = torch.as_tensor(np.asarray(batch_actions))
            qvals_t = torch.as_tensor(np.asarray(batch_qvals))
            log_prob_t = F.log_softmax(net(states_t), dim=1)
            selected_log_prob = log_prob_t[range(len(batch_states)), actions_t]
            loss = -(qvals_t * selected_log_prob).mean()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            updates += 1

            complete_batch_episodes = 0
            batch_states.clear()
            batch_actions.clear()
            batch_qvals.clear()

        on_step(env.total_steps, net, updates)

    return net, updates, {
        "train_params": {
            name: ref[name] for name in (
                "GAMMA", "LEARNING_RATE", "EPISODES_TO_TRAIN",
            )
        },
        "baseline": (
            "none" if algorithm == "reinforce"
            else "mean_discounted_returns_within_each_episode"
        ),
        "pending_complete_episodes": complete_batch_episodes,
        "pending_partial_episode_samples": len(cur_rewards),
        "note": "Episodios/batches incompletos ao fim do orcamento nao geram atualizacao.",
    }


def evaluate(
    net: nn.Module,
    algorithm: str,
    episodes: int,
    reinforce_eval_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Avalia politica congelada: sorteio softmax (padrao) ou argmax.

    As duas variantes REINFORCE usam exatamente o mesmo protocolo.
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
        "experiment": "E02",
        "algorithm": algorithm,
        "environment": ENV_ID,
        "seed": seed,
        "max_training_steps": max_steps,
        "stop_on_convergence": stop_on_convergence,
        "evaluation_interval_steps": None if stop_on_convergence else eval_every,
        "min_episodes_for_convergence": CONVERGENCE_WINDOW,
        "convergence_mean_reward": CONVERGENCE_REWARD,
        "evaluation_episodes": eval_episodes,
        "evaluation_seed_base": EVALUATION_SEED_BASE,
        "reinforce_evaluation_mode": reinforce_eval_mode,
        "baseline": (
            "none" if algorithm == "reinforce"
            else "mean_discounted_returns_within_each_episode"
        ),
    }
    write_json(run_dir / "config.json", config)

    print(f"\nE02 | {algorithm} | seed={seed} | max_steps={max_steps} | "
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
            "experiment": "E02",
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

    with (run_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as output:
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
            net, updates, extras = train_reinforce(
                env, algorithm, max_steps, record_episode, record_step,
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
            "reinforce": "Chapter11/02_cartpole_reinforce.py",
            "reinforce_baseline": "Chapter11/03_cartpole_reinforce_baseline.py",
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
        description="E02 CartPole — REINFORCE sem/com baseline",
    )
    parser.add_argument("--algorithm", choices=("reinforce", "reinforce_baseline", "both"), required=True)
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
        "--reinforce-eval", choices=("stochastic", "greedy"), default="stochastic",
        help="Avaliacao das duas politicas REINFORCE (padrao: stochastic)",
    )
    parser.add_argument("--report-every", type=int, default=20)
    args = parser.parse_args()
    if min(args.steps, args.eval_episodes, args.eval_every, args.report_every) <= 0:
        parser.error("--steps, --eval-episodes, --eval-every e --report-every devem ser positivos")

    run_seeds = args.seeds if args.seeds is not None else [
        args.seed if args.seed is not None else 42
    ]
    if len(set(run_seeds)) != len(run_seeds):
        parser.error("--seeds nao pode conter sementes duplicadas")
    algorithms = ("reinforce", "reinforce_baseline") if args.algorithm == "both" else (args.algorithm,)
    for seed in run_seeds:
        for algorithm in algorithms:
            run_one(
                algorithm=algorithm,
                seed=seed,
                max_steps=args.steps,
                eval_episodes=args.eval_episodes,
                reinforce_eval_mode=args.reinforce_eval,
                report_every=args.report_every,
                stop_on_convergence=args.stop_on_convergence,
                eval_every=args.eval_every,
            )
    print("\nE02: todas as execucoes solicitadas foram finalizadas.")


if __name__ == "__main__":
    main()

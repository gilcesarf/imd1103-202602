#!/usr/bin/env python3
"""E05 — Policy Gradient (PG) no Pong; comparacao posterior com o A2C do E04.

Referencia de treinamento: Chapter11/05_pong_pg.py. O arquivo original nao e alterado.
Mantem hiperparametros e algoritmo originais; adiciona contagem de env.step,
sementes, checkpoints, avaliacao independente e parada opcional.

Com --no-stop-on-convergence, acompanha 5 milhoes de interacoes por semente;
este script treina SOMENTE PG. A2C e reutilizado diretamente de results/e04
pelo analise_e05.py. Contador de interacoes: soma de env.step() dos subambientes
com wrappers Atari, nao quadros ALE brutos nem iteracoes do iterador PTAN.

Na raiz do repositorio:
  python experimentos-unidade-3/e05_pong.py --check-env
  python experimentos-unidade-3/e05_pong.py --seeds 42 123 2026 \\
    --steps 5000000 --eval-episodes 10 --eval-every 500000 \\
    --log-every-steps 100000 --no-stop-on-convergence
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

import gymnasium as gym
import numpy as np
import ptan
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

UNIT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = UNIT_DIR / "Chapter11"
REF_PATH = CHAPTER_DIR / "05_pong_pg.py"
RESULTS_DIR = UNIT_DIR / "results" / "e05" / "completo"
ENV_ID = "PongNoFrameskip-v4"
CONVERGENCE_WINDOW = 100
CONVERGENCE_THRESHOLD = 18.0
# Mesmas sementes e regra de amostragem estocastica usadas em e04_pong.py.
EVAL_SEED_BASE = 404000

EPISODE_COLUMNS = (
    "steps", "episode", "reward", "mean_reward_100", "episode_length", "last_loss",
)
UPDATE_COLUMNS = (
    "steps", "update", "experience_samples", "loss_policy", "loss_entropy", "loss_total",
    "entropy", "baseline", "mean_batch_scale", "std_batch_scale", "kl_old_new",
    "policy_grad_rms", "policy_grad_max", "policy_grad_var_components",
    "grad_norm_before_clip",
)
EVAL_COLUMNS = ("episode", "seed", "reward", "steps", "terminated", "truncated")
CHECKPOINT_COLUMNS = (
    "checkpoint_id", "checkpoint_type", "training_steps", "gradient_updates",
    "model_path", "evaluation_reward_mean", "evaluation_reward_std",
    "evaluation_reward_min", "evaluation_reward_max",
)
CHECKPOINT_EVAL_COLUMNS = (
    "checkpoint_id", "checkpoint_type", "training_steps", *EVAL_COLUMNS,
)


def reference():
    if not REF_PATH.is_file():
        raise FileNotFoundError(f"Codigo original nao encontrado: {REF_PATH}")
    sys.path.insert(0, str(CHAPTER_DIR))
    try:
        return runpy.run_path(str(REF_PATH), run_name="__e05_reference__")
    finally:
        sys.path.pop(0)


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_ID))


class SharedStepCounter:
    def __init__(self):
        self.total_steps = 0


class CountedEnv(gym.Wrapper):
    """Conta passos reais, sem misturar experiencias PTAN e env.step()."""

    def __init__(self, env, counter: SharedStepCounter, initial_seed: int):
        super().__init__(env)
        self.counter = counter
        self.initial_seed = initial_seed
        self.first_reset = True

    def reset(self, **kwargs):
        if self.first_reset:
            kwargs.setdefault("seed", self.initial_seed)
            self.first_reset = False
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        self.counter.total_steps += 1
        return result


def choose_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS indisponivel")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponivel")
    return device


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def write_csv(path: Path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def gradient_metrics(params):
    """Metrica registrada DEPOIS do clipping, como no PG original."""
    sqsum = gradsum = 0.0
    n = 0
    max_abs = 0.0
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad.detach().float()
        sqsum += float(grad.square().sum().item())
        gradsum += float(grad.sum().item())
        n += grad.numel()
        max_abs = max(max_abs, float(grad.abs().max().item()))
    if not n:
        return (float("nan"), float("nan"), float("nan"))
    mu = gradsum / n
    return float(np.sqrt(sqsum / n)), max_abs, max(0., sqsum / n - mu * mu)


def evaluate(net, device, count: int, max_eval_steps: int):
    """Mesma regra do E04: politica estocastica, seeds 404000+i, sem update.

    A avaliacao usa RNGs locais e nao altera o RNG NumPy do treinamento.
    """
    was_training = net.training
    net.eval()
    envs = []
    try:
        envs = [make_env() for _ in range(count)]
        observations = [env.reset(seed=EVAL_SEED_BASE + i)[0]
                        for i, env in enumerate(envs)]
        rngs = [np.random.default_rng(EVAL_SEED_BASE + i) for i in range(count)]
        reward_sums = [0.0] * count
        lengths = [0] * count
        terminateds = [False] * count
        truncateds = [False] * count
        active = list(range(count))
        with torch.no_grad():
            while active:
                arr = np.asarray([observations[i] for i in active])
                logits = net(torch.as_tensor(arr, dtype=torch.float32, device=device))
                probs_batch = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
                done = []
                for j, i in enumerate(active):
                    probs = probs_batch[j]
                    probs /= probs.sum()
                    action = int(rngs[i].choice(len(probs), p=probs))
                    obs, reward, terminated, truncated, _ = envs[i].step(action)
                    reward_sums[i] += float(reward)
                    lengths[i] += 1
                    terminateds[i] = bool(terminated)
                    truncateds[i] = bool(truncated) or (lengths[i] >= max_eval_steps and not terminated)
                    observations[i] = obs
                    if terminateds[i] or truncateds[i]:
                        done.append(i)
                if done:
                    completed = set(done)
                    active = [i for i in active if i not in completed]
        rows = [{
            "episode": i + 1, "seed": EVAL_SEED_BASE + i,
            "reward": reward_sums[i], "steps": lengths[i],
            "terminated": terminateds[i], "truncated": truncateds[i],
        } for i in range(count)]
    finally:
        for env in envs:
            env.close()
        net.train(was_training)
    values = np.asarray([r["reward"] for r in rows], dtype=np.float64)
    summary = {
        "episodes": count, "policy": "stochastic", "seed_base": EVAL_SEED_BASE,
        "reward_mean": float(values.mean()),
        "reward_std": float(values.std(ddof=1)) if count > 1 else 0.0,
        "reward_min": float(values.min()), "reward_max": float(values.max()),
        "truncated_count": sum(r["truncated"] for r in rows),
    }
    return summary, rows


def check_env(num_envs, device, ref):
    envs = [make_env() for _ in range(min(num_envs, 2))]
    try:
        obs, _ = envs[0].reset(seed=42)
        net = ref["common"].AtariPGN(envs[0].observation_space.shape,
                                      envs[0].action_space.n).to(device)
        with torch.no_grad():
            logits = net(torch.as_tensor(obs[None], dtype=torch.float32, device=device))
        print(f"OK {ENV_ID} | shape={envs[0].observation_space.shape} | "
              f"acoes={envs[0].action_space.n} | logits={tuple(logits.shape)} | "
              f"device={device} | PG ambientes configurados={num_envs}")
    finally:
        for env in envs:
            env.close()


def run_one(seed, args, ref, device):
    seed_everything(seed)
    mode = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    tag = "stop" if args.stop_on_convergence else "nostop"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_id = f"pg_seed{seed}_steps{args.steps}_envs{args.num_envs}_{tag}_{stamp}"
    out = RESULTS_DIR / mode / run_id
    out.mkdir(parents=True, exist_ok=False)

    train_params = {key.lower(): ref[key] for key in (
        "GAMMA", "LEARNING_RATE", "ENTROPY_BETA", "BATCH_SIZE", "REWARD_STEPS",
        "BASELINE_STEPS", "GRAD_L2_CLIP", "ENV_COUNT")}
    config = {
        "experiment": "E05", "algorithm": "pg", "environment": ENV_ID,
        "seed": seed, "max_training_steps": args.steps,
        "stop_on_convergence": args.stop_on_convergence,
        "evaluation_interval_steps": None if args.stop_on_convergence else args.eval_every,
        "evaluation_episodes": args.eval_episodes,
        "evaluation_policy": "stochastic", "evaluation_seed_base": EVAL_SEED_BASE,
        "evaluation_max_episode_steps": args.max_eval_steps,
        "num_envs": args.num_envs, "vectorization": "ptan_list_of_envs",
        "device": str(device), "convergence_window": CONVERGENCE_WINDOW,
        "convergence_mean_reward_strict_greater_than": CONVERGENCE_THRESHOLD,
        "reference": "Chapter11/05_pong_pg.py", "train_params": train_params,
        "step_definition": "sum of wrapped env.step calls across list of environments",
        "ale_raw_frames_equal_steps": False,
        "gradient_metrics_interval_updates": args.grad_metrics_every,
    }
    write_json(out / "config.json", config)
    print(f"\nE05 PG | seed={seed} | ambientes={args.num_envs} | budget={args.steps} "
          f"| modo={mode} | device={device}", flush=True)
    print(f"Resultados: {out}", flush=True)

    counter = SharedStepCounter()
    envs = [CountedEnv(make_env(), counter, seed + i) for i in range(args.num_envs)]
    network = ref["common"].AtariPGN(envs[0].observation_space.shape,
                                      envs[0].action_space.n).to(device)
    agent = ptan.agent.PolicyAgent(network, apply_softmax=True, device=device)
    source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=ref["GAMMA"], steps_count=ref["REWARD_STEPS"])
    optimizer = optim.Adam(network.parameters(), lr=ref["LEARNING_RATE"], eps=1e-3)
    baseline_buffer = ref["MeanBuffer"](ref["BASELINE_STEPS"])

    episode_rows = []
    checkpoint_rows = []
    checkpoint_eval_rows = []
    training_rewards = []
    convergence = None
    updates = 0
    experience_samples = 0
    last_loss = None
    batch_states, batch_actions, batch_scales = [], [], []
    next_milestone = args.eval_every
    eval_seconds = 0.0
    start = time.perf_counter()
    prev_log = start
    prev_log_eval_seconds = 0.0
    prev_log_steps = 0
    stop_reason = "step_budget"

    def save_checkpoint(kind: str, steps: int):
        nonlocal eval_seconds
        t0 = time.perf_counter()
        ckpt_id = f"{kind}_steps{steps}"
        rel_path = Path("checkpoints") / f"{ckpt_id}.pt"
        path = out / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": {k: v.detach().cpu() for k, v in network.state_dict().items()},
            "experiment": "E05", "algorithm": "pg", "seed": seed,
            "checkpoint_type": kind, "training_steps": steps,
            "gradient_updates": updates,
            "note": "Apenas pesos, sem otimizador, buffer e estados dos ambientes",
        }, path)
        stats, rows = evaluate(network, device, args.eval_episodes, args.max_eval_steps)
        checkpoint_rows.append({
            "checkpoint_id": ckpt_id, "checkpoint_type": kind,
            "training_steps": steps, "gradient_updates": updates,
            "model_path": str(rel_path),
            "evaluation_reward_mean": stats["reward_mean"],
            "evaluation_reward_std": stats["reward_std"],
            "evaluation_reward_min": stats["reward_min"],
            "evaluation_reward_max": stats["reward_max"],
        })
        checkpoint_eval_rows.extend({
            "checkpoint_id": ckpt_id, "checkpoint_type": kind,
            "training_steps": steps, **row,
        } for row in rows)
        eval_seconds += time.perf_counter() - t0
        print(f"Checkpoint {ckpt_id} | avaliacao={stats['reward_mean']:.2f} "
              f"+/- {stats['reward_std']:.2f}", flush=True)
        return stats, rows

    try:
        with (out / "updates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(UPDATE_COLUMNS))
            writer.writeheader()
            for exp in source:
                experience_samples += 1
                baseline_buffer.add(float(exp.reward))
                baseline = float(baseline_buffer.mean())
                batch_states.append(np.asarray(exp.state))
                batch_actions.append(int(exp.action))
                batch_scales.append(float(exp.reward) - baseline)
                for reward, length in source.pop_rewards_steps():
                    reward = float(reward)
                    training_rewards.append(reward)
                    ep_index = len(training_rewards)
                    mean100 = float(np.mean(training_rewards[-CONVERGENCE_WINDOW:]))
                    episode_rows.append({
                        "steps": counter.total_steps, "episode": ep_index,
                        "reward": reward, "mean_reward_100": mean100,
                        "episode_length": int(length),
                        "last_loss": "" if last_loss is None else last_loss,
                    })
                    is_converged = (ep_index >= CONVERGENCE_WINDOW
                                    and mean100 > CONVERGENCE_THRESHOLD)
                    if is_converged and convergence is None:
                        convergence = {
                            "steps": counter.total_steps, "episodes": ep_index,
                            "mean_reward_100": mean100,
                        }
                        print(f"PRIMEIRA CONVERGENCIA: {convergence}", flush=True)
                        if not args.stop_on_convergence:
                            save_checkpoint("first_convergence", counter.total_steps)
                    if ep_index == 1 or ep_index % args.report_every == 0:
                        print(f"steps={counter.total_steps:,} | jogos={ep_index:,} | "
                              f"media100={mean100:.2f} | reward={reward:.1f}", flush=True)
                    if args.stop_on_convergence and is_converged:
                        stop_reason = "convergence"
                        break
                if stop_reason == "convergence":
                    break

                if len(batch_states) >= ref["BATCH_SIZE"]:
                    states_t = torch.as_tensor(np.asarray(batch_states), device=device)
                    actions_t = torch.as_tensor(batch_actions, dtype=torch.long, device=device)
                    scales_t = torch.as_tensor(batch_scales, dtype=torch.float32, device=device)
                    optimizer.zero_grad()
                    logits = network(states_t)
                    log_probs = F.log_softmax(logits, dim=1)
                    selected = log_probs[torch.arange(len(batch_actions), device=device), actions_t]
                    loss_policy = -(scales_t * selected).mean()
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum(dim=1).mean()
                    loss_entropy = -ref["ENTROPY_BETA"] * entropy
                    loss = loss_policy + loss_entropy
                    loss.backward()
                    grad_norm_before = nn_utils.clip_grad_norm_(network.parameters(),
                                                                ref["GRAD_L2_CLIP"])
                    updates += 1
                    if updates % args.grad_metrics_every == 0:
                        grad_rms, grad_max, grad_var = gradient_metrics(network.parameters())
                    else:
                        grad_rms = grad_max = grad_var = ""
                    # KL da politica antes/depois da atualizacao: diagnostico (livro).
                    old_prob = probs.detach()
                    optimizer.step()
                    with torch.no_grad():
                        new_log_probs = F.log_softmax(network(states_t), dim=1)
                        kl_old_new = float(F.kl_div(new_log_probs, old_prob, reduction="batchmean").item())
                    last_loss = float(loss.item())
                    writer.writerow({
                        "steps": counter.total_steps, "update": updates,
                        "experience_samples": experience_samples,
                        "loss_policy": float(loss_policy.item()),
                        "loss_entropy": float(loss_entropy.item()),
                        "loss_total": last_loss, "entropy": float(entropy.item()),
                        "baseline": baseline,
                        "mean_batch_scale": float(np.mean(batch_scales)),
                        "std_batch_scale": float(np.std(batch_scales)),
                        "kl_old_new": kl_old_new,
                        "policy_grad_rms": grad_rms,
                        "policy_grad_max": grad_max,
                        "policy_grad_var_components": grad_var,
                        "grad_norm_before_clip": float(grad_norm_before.item()),
                    })
                    batch_states.clear()
                    batch_actions.clear()
                    batch_scales.clear()
                    if updates % 100 == 0:
                        stream.flush()

                if not args.stop_on_convergence:
                    while next_milestone < args.steps and counter.total_steps >= next_milestone:
                        if not any(row["training_steps"] == counter.total_steps
                                   for row in checkpoint_rows):
                            save_checkpoint("periodic", counter.total_steps)
                        next_milestone += args.eval_every

                if counter.total_steps >= args.steps:
                    break
                if counter.total_steps - prev_log_steps >= args.log_every_steps:
                    now = time.perf_counter()
                    dt = now - prev_log - (eval_seconds - prev_log_eval_seconds)
                    fps = (counter.total_steps - prev_log_steps) / max(1e-9, dt)
                    prev_log = now
                    prev_log_eval_seconds = eval_seconds
                    prev_log_steps = counter.total_steps
                    print(f"steps={counter.total_steps:,} | updates={updates:,} | "
                          f"jogos={len(training_rewards):,} | "
                          f"throughput={fps:.1f} interacoes/s", flush=True)
    finally:
        for env in envs:
            env.close()

    training_wall_seconds = time.perf_counter() - start
    final_stats, final_rows = save_checkpoint("final", counter.total_steps)
    write_csv(out / "episodes.csv", EPISODE_COLUMNS, episode_rows)
    write_csv(out / "checkpoints.csv", CHECKPOINT_COLUMNS, checkpoint_rows)
    write_csv(out / "checkpoint_evaluations.csv", CHECKPOINT_EVAL_COLUMNS,
              checkpoint_eval_rows)
    write_csv(out / "evaluation.csv", EVAL_COLUMNS, final_rows)
    elapsed_total = time.perf_counter() - start
    training_seconds = max(0., training_wall_seconds - eval_seconds)
    summary = {
        **config, "run_id": run_id, "stop_reason": stop_reason,
        "converged": convergence is not None, "convergence": convergence,
        "training_steps": counter.total_steps, "experience_samples": experience_samples,
        "episodes_completed": len(training_rewards), "gradient_updates": updates,
        "training_reward_mean_last_100": (
            float(np.mean(training_rewards[-CONVERGENCE_WINDOW:])) if training_rewards else None),
        "training_elapsed_seconds_excluding_evaluations": round(training_seconds, 3),
        "checkpoint_and_evaluation_elapsed_seconds": round(eval_seconds, 3),
        "total_elapsed_seconds": round(elapsed_total, 3),
        "wrapped_interactions_per_training_second": round(
            counter.total_steps / max(training_seconds, 1e-9), 2),
        "evaluation": final_stats,
    }
    write_json(out / "summary.json", summary)
    print(f"FINAL E05 PG | steps={counter.total_steps:,} | "
          f"jogos={len(training_rewards):,} | convergiu={convergence is not None} | "
          f"aval={final_stats['reward_mean']:.2f}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description="E05 — PG Pong do Capitulo 11")
    p.add_argument("--check-env", action="store_true", help="Verifica Pong e rede PG sem treinar")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=None,
                   help="Padrao 32 ambientes do PG original, nao 50 do A2C")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500_000)
    p.add_argument("--max-eval-steps", type=int, default=20_000)
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    p.add_argument("--report-every", type=int, default=20)
    p.add_argument("--log-every-steps", type=int, default=100_000)
    p.add_argument("--grad-metrics-every", type=int, default=10)
    flags = p.add_mutually_exclusive_group()
    flags.add_argument("--stop-on-convergence", action="store_true",
                       dest="stop_on_convergence")
    flags.add_argument("--no-stop-on-convergence", action="store_false",
                       dest="stop_on_convergence")
    p.set_defaults(stop_on_convergence=True)
    args = p.parse_args()
    ref = reference()
    if args.num_envs is None:
        args.num_envs = int(ref["ENV_COUNT"])
    if min(args.steps, args.num_envs, args.eval_episodes, args.eval_every,
           args.max_eval_steps, args.report_every, args.log_every_steps,
           args.grad_metrics_every) <= 0:
        p.error("Todos os valores de contagem devem ser positivos")
    device = choose_device(args.device)
    if args.check_env:
        check_env(args.num_envs, device, ref)
        return
    for seed in args.seeds:
        run_one(seed, args, ref, device)


if __name__ == "__main__":
    main()

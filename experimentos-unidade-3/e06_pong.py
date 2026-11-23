#!/usr/bin/env python3
"""E06 — A2C Pong: vetorizacao sincrona vs assincrona (Chapter12/02_pong_a2c.py).

O original do livro permanece inalterado. A rede, bootstrap n-step, perdas A2C
e hiperparametros seguem E04. Apenas a implementacao dos ambientes muda:
SyncVectorEnv vs AsyncVectorEnv. O modelo e atualizado no processo principal;
nao se trata de A3C (gradientes assincronos).

O orcamento --steps conta INTERACOES dos subambientes com os wrappers Atari;
nao conta quadros brutos do ALE nem iteracoes do iterador PTAN.

Exemplo de preflight:
  python experimentos-unidade-3/e06_pong.py --check-env --vector-mode async --num-envs 2
Exemplo de piloto:
  python experimentos-unidade-3/e06_pong.py --seeds 42 --steps 10000 \
    --vector-mode async --num-envs 2 --eval-episodes 2 --eval-every 5000 \
    --no-stop-on-convergence
Exemplo completo (apos validar o uso de memoria e processos):
  python experimentos-unidade-3/e06_pong.py --seeds 42 123 2026 \
    --steps 5000000 --num-envs 50 --vector-mode async \
    --eval-episodes 10 --eval-every 500000 --no-stop-on-convergence
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
from typing import Any

import gymnasium as gym
import numpy as np
import ptan
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

UNIT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = UNIT_DIR / "Chapter12"
REF_PATH = CHAPTER_DIR / "02_pong_a2c.py"
RESULTS_DIR = UNIT_DIR / "results" / "e06" / "completo"
ENV_ID = "PongNoFrameskip-v4"
CONVERGENCE_WINDOW = 100
CONVERGENCE_THRESHOLD = 18.0
EVAL_SEED_BASE = 404000

EPISODE_COLUMNS = (
    "steps", "vector_steps", "episode", "reward", "mean_reward_100",
    "episode_length", "last_loss",
)
UPDATE_COLUMNS = (
    "steps", "vector_steps", "update", "experience_samples", "loss_policy",
    "loss_value", "loss_entropy", "loss_total", "advantage_mean", "advantage_std",
    "value_mean", "value_target_mean", "entropy", "policy_grad_rms",
    "policy_grad_max", "policy_grad_var_components", "grad_norm_before_clip",
)
EVAL_COLUMNS = ("episode", "seed", "reward", "steps", "terminated", "truncated")
CHECKPOINT_COLUMNS = (
    "checkpoint_id", "checkpoint_type", "training_steps", "vector_steps",
    "gradient_updates", "model_path", "evaluation_reward_mean", "evaluation_reward_std",
    "evaluation_reward_min", "evaluation_reward_max",
)
CHECKPOINT_EVAL_COLUMNS = (
    "checkpoint_id", "checkpoint_type", "training_steps", *EVAL_COLUMNS,
)


def reference() -> dict[str, Any]:
    if not REF_PATH.is_file():
        raise FileNotFoundError(f"Codigo original nao encontrado: {REF_PATH}")
    sys.path.insert(0, str(CHAPTER_DIR))
    try:
        return runpy.run_path(str(REF_PATH), run_name="__e06_reference__")
    finally:
        sys.path.pop(0)


def make_env():
    """Emprega exatamente os wrappers Atari utilizados no capitulo 12."""
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_ID))


class CountedVectorEnv:
    """Proxy para medir passos reais dos NUM_ENVS sem modificar o Gymnasium."""
    def __init__(self, env):
        self.env = env
        self.num_envs = int(env.num_envs)
        self.vector_steps = 0
        self.total_steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        result = self.env.step(actions)
        self.vector_steps += 1
        self.total_steps += self.num_envs
        return result

    def close(self):
        self.env.close()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    result = torch.device(value)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device mps solicitado, mas nao esta disponivel")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device cuda solicitado, mas nao esta disponivel")
    return result


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def evaluate(net, device, count: int, max_eval_steps: int):
    """Politica estocastica avaliada em lote, com RNG local por episodio.

    Inferencia agrupada reduz o overhead do Pong sem alterar o algoritmo.
    Nao altera os estados aleatorios do treino, nem os pesos da rede.
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
                obs_array = np.asarray([observations[i] for i in active])
                obs_t = torch.as_tensor(obs_array, device=device, dtype=torch.float32)
                logits, _ = net(obs_t)
                probs_batch = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
                finished = []
                for idx_in_batch, i in enumerate(active):
                    probs = probs_batch[idx_in_batch]
                    probs /= probs.sum()
                    action = int(rngs[i].choice(len(probs), p=probs))
                    obs, reward, term, trunc, _ = envs[i].step(action)
                    reward_sums[i] += float(reward)
                    lengths[i] += 1
                    terminateds[i] = bool(term)
                    truncateds[i] = bool(trunc) or (lengths[i] >= max_eval_steps and not term)
                    observations[i] = obs
                    if term or truncateds[i]:
                        finished.append(i)
                if finished:
                    completed = set(finished)
                    active = [i for i in active if i not in completed]
        rows = [{
            "episode": i + 1,
            "seed": EVAL_SEED_BASE + i,
            "reward": reward_sums[i],
            "steps": lengths[i],
            "terminated": terminateds[i],
            "truncated": truncateds[i],
        } for i in range(count)]
    finally:
        for env in envs:
            env.close()
        net.train(was_training)
    values = np.asarray([r["reward"] for r in rows], dtype=np.float64)
    summary = {
        "episodes": count,
        "policy": "stochastic",
        "seed_base": EVAL_SEED_BASE,
        "reward_mean": float(values.mean()),
        "reward_std": float(values.std(ddof=1)) if count > 1 else 0.0,
        "reward_min": float(values.min()),
        "reward_max": float(values.max()),
        "truncated_count": sum(r["truncated"] for r in rows),
    }
    return summary, rows


def gradient_metrics(params):
    """Gradiente APENAS de politica, como grad_l2/max/var do original."""
    sqsum = 0.0
    gradsum = 0.0
    n = 0
    max_value = 0.0
    for param in params:
        if param.grad is None:
            continue
        v = param.grad.detach().float()
        sqsum += float((v * v).sum().item())
        gradsum += float(v.sum().item())
        n += v.numel()
        max_value = max(max_value, float(v.abs().max().item()))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = gradsum / n
    return (float(np.sqrt(sqsum / n)), max_value, max(0.0, sqsum / n - mean * mean))


def build_vector_env(mode: str, num_envs: int, mp_context: str):
    """Factories top-level picklable para multiprocessing no macOS (spawn)."""
    import functools
    factories = [functools.partial(make_env) for _ in range(num_envs)]
    if mode == "async":
        return gym.vector.AsyncVectorEnv(factories, context=mp_context,
                                         shared_memory=True)
    return gym.vector.SyncVectorEnv(factories)


def check_env(num_envs: int, device: torch.device, vector_mode: str, mp_context: str):
    """Teste minimo de import, ROMs, wrappers, rede e API vetorizada."""
    vec = build_vector_env(vector_mode, num_envs, mp_context)
    try:
        obs, _ = vec.reset(seed=42)
        acts = vec.action_space.sample()
        obs2, rewards, terminated, truncated, _ = vec.step(acts)
        print(f"OK {ENV_ID} | modo={vector_mode} | ambientes={num_envs} | shape={vec.single_observation_space.shape} "
              f"| acoes={vec.single_action_space.n} | batch={obs.shape} "
              f"| rewards={rewards.shape} | device={device}")
        ref = reference()
        net = ref["common"].AtariA2C(vec.single_observation_space.shape,
                                       vec.single_action_space.n).to(device)
        with torch.no_grad():
            logits, values = net(torch.as_tensor(obs[:1], dtype=torch.float32, device=device))
        print(f"OK AtariA2C | logits={tuple(logits.shape)} | value={tuple(values.shape)}")
        del obs2, terminated, truncated
    finally:
        vec.close()


def run_one(seed: int, args: argparse.Namespace, ref: dict, device: torch.device):
    seed_everything(seed)
    mode = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    tag = "stop" if args.stop_on_convergence else "nostop"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_id = f"a2c_{args.vector_mode}_seed{seed}_steps{args.steps}_envs{args.num_envs}_{tag}_{stamp}"
    run_dir = RESULTS_DIR / mode / args.vector_mode / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    constants = {name.lower(): ref[name] for name in (
        "GAMMA", "LEARNING_RATE", "ENTROPY_BETA", "BATCH_SIZE", "REWARD_STEPS", "CLIP_GRAD")}
    config = {
        "experiment": "E06", "algorithm": "a2c", "environment": ENV_ID,
        "seed": seed, "max_training_steps": args.steps,
        "stop_on_convergence": args.stop_on_convergence,
        "evaluation_interval_steps": None if args.stop_on_convergence else args.eval_every,
        "evaluation_episodes": args.eval_episodes,
        "evaluation_policy": "stochastic", "evaluation_seed_base": EVAL_SEED_BASE,
        "evaluation_max_episode_steps": args.max_eval_steps,
        "num_envs": args.num_envs, "vectorization": args.vector_mode,
        "multiprocessing_context": args.mp_context if args.vector_mode == "async" else None,
        "device": str(device),
        "convergence_window": CONVERGENCE_WINDOW,
        "convergence_mean_reward_strict_greater_than": CONVERGENCE_THRESHOLD,
        "gradient_metrics_interval_updates": args.grad_metrics_every,
        "reference": "Chapter12/02_pong_a2c.py", "batch_advantage_fix": "squeeze(-1)",
        "step_definition": "sum of wrapped env.step calls across vector environments",
        "ale_raw_frames_equal_steps": False,
        "train_params": constants,
    }
    write_json(run_dir / "config.json", config)
    print(f"\nE06 | vector={args.vector_mode} | seed={seed} | ambientes={args.num_envs} | budget={args.steps} "
          f"| modo={mode} | device={device}", flush=True)
    print(f"Resultados: {run_dir}", flush=True)

    vector_env = build_vector_env(args.vector_mode, args.num_envs, args.mp_context)
    env = CountedVectorEnv(vector_env)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    net = ref["common"].AtariA2C(
        env.single_observation_space.shape, env.single_action_space.n).to(device)
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    source = ptan.experience.VectorExperienceSourceFirstLast(
        env, agent, gamma=ref["GAMMA"], steps_count=ref["REWARD_STEPS"], env_seed=seed)
    optimizer = optim.Adam(net.parameters(), lr=ref["LEARNING_RATE"], eps=1e-3)

    episode_rows = []
    checkpoint_rows = []
    checkpoint_eval_rows = []
    training_rewards = []
    convergence = None
    updates = 0
    experience_samples = 0
    last_loss = None
    batch = []
    next_milestone = args.eval_every
    eval_seconds = 0.0
    start = time.perf_counter()
    prev_log = start
    prev_steps = 0
    total_episodes_with_rewards = 0
    stop_reason = "step_budget"

    def save_checkpoint(kind: str, steps: int):
        nonlocal eval_seconds
        t0 = time.perf_counter()
        name = f"{kind}_steps{steps}"
        rel = Path("checkpoints") / f"{name}.pt"
        p = run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}
        torch.save({"model_state_dict": state, "experiment": "E06", "seed": seed,
                    "algorithm": "a2c", "checkpoint_type": kind,
                    "training_steps": steps, "gradient_updates": updates,
                    "note": "Apenas pesos, sem estado do otimizador ou dos ambientes"}, p)
        evaluation, rows = evaluate(net, device, args.eval_episodes, args.max_eval_steps)
        checkpoint_rows.append({
            "checkpoint_id": name, "checkpoint_type": kind, "training_steps": steps,
            "vector_steps": env.vector_steps, "gradient_updates": updates,
            "model_path": str(rel),
            "evaluation_reward_mean": evaluation["reward_mean"],
            "evaluation_reward_std": evaluation["reward_std"],
            "evaluation_reward_min": evaluation["reward_min"],
            "evaluation_reward_max": evaluation["reward_max"],
        })
        checkpoint_eval_rows.extend({"checkpoint_id": name, "checkpoint_type": kind,
                                     "training_steps": steps, **r} for r in rows)
        eval_seconds += time.perf_counter() - t0
        print(f"Checkpoint {name} | avaliacao={evaluation['reward_mean']:.2f} "
              f"+/- {evaluation['reward_std']:.2f}", flush=True)
        return evaluation, rows

    try:
        with (run_dir / "updates.csv").open("w", newline="", encoding="utf-8") as f:
            update_writer = csv.DictWriter(f, fieldnames=list(UPDATE_COLUMNS))
            update_writer.writeheader()
            for exp in source:
                batch.append(exp)
                experience_samples += 1
                rewards_and_steps = source.pop_rewards_steps()
                for reward, length in rewards_and_steps:
                    total_episodes_with_rewards += 1
                    reward = float(reward)
                    training_rewards.append(reward)
                    avg = float(np.mean(training_rewards[-CONVERGENCE_WINDOW:]))
                    episode_rows.append({
                        "steps": env.total_steps, "vector_steps": env.vector_steps,
                        "episode": total_episodes_with_rewards, "reward": reward,
                        "mean_reward_100": avg, "episode_length": int(length),
                        "last_loss": "" if last_loss is None else last_loss,
                    })
                    is_converged = (
                        total_episodes_with_rewards >= CONVERGENCE_WINDOW
                        and avg > CONVERGENCE_THRESHOLD)
                    if is_converged and convergence is None:
                        convergence = {"steps": env.total_steps,
                                       "vector_steps": env.vector_steps,
                                       "episodes": total_episodes_with_rewards,
                                       "mean_reward_100": avg}
                        print(f"PRIMEIRA CONVERGENCIA: {convergence}", flush=True)
                        if not args.stop_on_convergence:
                            save_checkpoint("first_convergence", env.total_steps)
                    if (total_episodes_with_rewards == 1
                            or total_episodes_with_rewards % args.report_every == 0):
                        print(f"steps={env.total_steps:,} | jogos={total_episodes_with_rewards} "
                              f"| media100={avg:.2f} | reward={reward:.1f}", flush=True)
                    if args.stop_on_convergence and is_converged:
                        stop_reason = "convergence"
                        break
                if stop_reason == "convergence":
                    break

                if len(batch) >= ref["BATCH_SIZE"]:
                    states_t, actions_t, targets_t = ref["common"].unpack_batch(
                        batch, net, device=device,
                        gamma=ref["GAMMA"], reward_steps=ref["REWARD_STEPS"])
                    batch.clear()
                    optimizer.zero_grad()
                    logits_t, value_t = net(states_t)
                    values_t = value_t.squeeze(-1)
                    loss_value_t = F.mse_loss(values_t, targets_t)
                    log_probs = F.log_softmax(logits_t, dim=1)
                    advantages = targets_t - values_t.detach()  # corrigido: [batch], nao [batch,batch]
                    selected = log_probs[torch.arange(len(actions_t), device=device), actions_t]
                    loss_policy_t = -(advantages * selected).mean()
                    probs = F.softmax(logits_t, dim=1)
                    entropy = -(probs * log_probs).sum(dim=1).mean()
                    loss_entropy_t = -ref["ENTROPY_BETA"] * entropy
                    loss_policy_t.backward(retain_graph=True)
                    updates += 1
                    if updates % args.grad_metrics_every == 0:
                        grad_rms, grad_max, grad_var = gradient_metrics(net.parameters())
                    else:
                        grad_rms = grad_max = grad_var = ""
                    (loss_value_t + loss_entropy_t).backward()
                    grad_norm_before = nn_utils.clip_grad_norm_(net.parameters(), ref["CLIP_GRAD"])
                    optimizer.step()
                    last_loss = float((loss_value_t + loss_policy_t + loss_entropy_t).item())
                    update_writer.writerow({
                        "steps": env.total_steps, "vector_steps": env.vector_steps,
                        "update": updates, "experience_samples": experience_samples,
                        "loss_policy": float(loss_policy_t.item()),
                        "loss_value": float(loss_value_t.item()),
                        "loss_entropy": float(loss_entropy_t.item()),
                        "loss_total": last_loss,
                        "advantage_mean": float(advantages.mean().item()),
                        "advantage_std": float(advantages.std(unbiased=False).item()),
                        "value_mean": float(values_t.mean().item()),
                        "value_target_mean": float(targets_t.mean().item()),
                        "entropy": float(entropy.item()),
                        "policy_grad_rms": grad_rms, "policy_grad_max": grad_max,
                        "policy_grad_var_components": grad_var,
                        "grad_norm_before_clip": float(grad_norm_before.item()),
                    })
                    if updates % 100 == 0:
                        f.flush()

                # Fonte vetorizada PTAN fornece um grupo de NUM_ENVS experiencias
                # por passo vetorizado (sync ou async). Terminar apenas apos consumir o grupo.
                at_group_end = (experience_samples % args.num_envs == 0)
                if at_group_end:
                    if not args.stop_on_convergence:
                        while next_milestone < args.steps and env.total_steps >= next_milestone:
                            if not any(r["training_steps"] == env.total_steps
                                       for r in checkpoint_rows):
                                save_checkpoint("periodic", env.total_steps)
                            next_milestone += args.eval_every
                    if env.total_steps >= args.steps:
                        break
                    if env.total_steps - prev_steps >= args.log_every_steps:
                        now = time.perf_counter()
                        fps = (env.total_steps - prev_steps) / max(now - prev_log, 1e-9)
                        prev_steps, prev_log = env.total_steps, now
                        print(f"steps={env.total_steps:,} | updates={updates:,} | "
                              f"jogos={total_episodes_with_rewards:,} | throughput={fps:.1f} interacoes/s",
                              flush=True)
    finally:
        env.close()

    training_wall_seconds = time.perf_counter() - start
    # Checkpoint final inclusive se o treino parou pela convergencia.
    final_evaluation, final_rows = save_checkpoint("final", env.total_steps)
    write_csv(run_dir / "episodes.csv", EPISODE_COLUMNS, episode_rows)
    write_csv(run_dir / "checkpoints.csv", CHECKPOINT_COLUMNS, checkpoint_rows)
    write_csv(run_dir / "checkpoint_evaluations.csv", CHECKPOINT_EVAL_COLUMNS,
              checkpoint_eval_rows)
    write_csv(run_dir / "evaluation.csv", EVAL_COLUMNS, final_rows)
    elapsed_total = time.perf_counter() - start
    summary = {
        **config, "run_id": run_id, "stop_reason": stop_reason,
        "converged": convergence is not None, "convergence": convergence,
        "training_steps": env.total_steps, "vector_steps": env.vector_steps,
        "experience_samples": experience_samples,
        "episodes_completed": total_episodes_with_rewards,
        "gradient_updates": updates,
        "training_reward_mean_last_100": (
            float(np.mean(training_rewards[-100:])) if training_rewards else None),
        "training_elapsed_seconds_excluding_evaluations": round(max(0., training_wall_seconds - eval_seconds), 3),
        "checkpoint_and_evaluation_elapsed_seconds": round(eval_seconds, 3),
        "total_elapsed_seconds": round(elapsed_total, 3),
        "wrapped_interactions_per_training_second": round(
            env.total_steps / max(training_wall_seconds - eval_seconds, 1e-9), 2),
        "evaluation": final_evaluation,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"FINAL E06 {args.vector_mode} | steps={env.total_steps:,} | jogos={total_episodes_with_rewards} | "
          f"convergiu={convergence is not None} | aval={final_evaluation['reward_mean']:.2f}",
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="E06 — A2C Pong: SyncVectorEnv vs AsyncVectorEnv")
    parser.add_argument("--check-env", action="store_true", help="Verifica ambiente Atari e modelo; nao treina")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--steps", type=int, default=5_000_000,
                        help="Orcamento de interacoes dos subambientes (nao frames ALE)")
    parser.add_argument("--vector-mode", choices=("sync", "async"), default="async",
                        help="Modo de coleta dos ambientes; padrao async")
    parser.add_argument("--mp-context", choices=("spawn", "forkserver"), default="spawn",
                        help="Processos AsyncVectorEnv; spawn recomendado no macOS com MPS")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Ambientes paralelos; padrao 50 do livro")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--max-eval-steps", type=int, default=20_000)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--report-every", type=int, default=20, help="Jogos completos por log")
    parser.add_argument("--log-every-steps", type=int, default=100_000)
    parser.add_argument("--grad-metrics-every", type=int, default=10,
                        help="Mede gradiente da politica a cada N atualizacoes")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--stop-on-convergence", dest="stop_on_convergence", action="store_true")
    flags.add_argument("--no-stop-on-convergence", dest="stop_on_convergence", action="store_false")
    parser.set_defaults(stop_on_convergence=True)
    args = parser.parse_args()
    ref = reference()
    if args.num_envs is None:
        args.num_envs = int(ref["NUM_ENVS"])
    if min(args.steps, args.num_envs, args.eval_episodes, args.eval_every,
           args.report_every, args.log_every_steps, args.grad_metrics_every,
           args.max_eval_steps) <= 0:
        parser.error("Todos os parametros numericos de contagem devem ser > 0")
    device = choose_device(args.device)
    if args.check_env:
        try:
            check_env(args.num_envs, device, args.vector_mode, args.mp_context)
        except Exception as exc:
            raise SystemExit(
                f"Falha no Pong/Atari: {exc}\nVerifique gymnasium[atari]==0.29.1, "
                "ALE-Py e os ROMs via AutoROM no .venv do livro.") from exc
        return
    for seed in args.seeds:
        run_one(seed, args, ref, device)


if __name__ == "__main__":
    main()

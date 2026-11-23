#!/usr/bin/env python3
"""E07 — A3C-Grad no Pong (Chapter12/03_a3c_grad.py).

Quatro processos, cada um com 8 ambientes SyncVectorEnv (configuracao do livro),
calculam gradientes de uma rede ator-critico e enviam ao processo principal;
o principal soma TRAIN_BATCH=2 pacotes e aplica Adam na rede CPU compartilhada.

Mudancas instrumentais documentadas vs livro:
- vantagem com shape [batch], e nao broadcasting [batch, batch];
- copia local dos pesos compartilhados, sob lock, antes de cada calculo de gradiente,
  evitando escrita concorrente em gradientes do modelo compartilhado;
- parada por orcamento de interacoes com ambientes, nao por pacotes de gradiente;
- checkpoints, avaliacoes, diagnosticos de gradiente, sementes e erro de worker.

O modelo compartilhado exige CPU. Nao se deve comparar sua vazao diretamente
com E04 treinado em MPS como se somente o algoritmo tivesse mudado.

Requer e04_pong.py no mesmo diretorio (wrappers, avaliacao e classes auxiliares).
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import torch.multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

import gymnasium as gym
import ptan

import e04_pong as base

# No macOS, multiprocessing usa spawn: cada worker precisa importar novamente
# a classe lib.common.AtariA2C durante a deserializacao da rede compartilhada.
# base.reference() adiciona Chapter12 ao sys.path apenas temporariamente;
# mantemos o caminho durante toda a vida do processo e o herdamos no spawn.
CHAPTER12_DIR = str(base.CHAPTER_DIR)
if CHAPTER12_DIR not in sys.path:
    sys.path.insert(0, CHAPTER12_DIR)

ROOT = Path(__file__).resolve().parent
RESULTS_BASE = ROOT / "results" / "e07" / "completo"
ENV_ID = "PongNoFrameskip-v4"
WINDOW, THRESHOLD = 100, 18.0

EPISODE_COLUMNS = ("steps", "episode", "reward", "mean_reward_100", "worker_id", "episode_length")
UPDATE_COLUMNS = (
    "steps", "update", "gradient_messages", "worker_id_last", "loss_policy_mean",
    "loss_value_mean", "loss_entropy_mean", "loss_total_mean", "advantage_mean",
    "entropy_mean", "worker_grad_rms_mean", "aggregated_grad_rms",
    "aggregated_grad_max", "grad_norm_before_clip",
)
CHECKPOINT_COLUMNS = (
    "checkpoint_id", "checkpoint_type", "training_steps", "gradient_updates",
    "model_path", "evaluation_reward_mean", "evaluation_reward_std",
    "evaluation_reward_min", "evaluation_reward_max",
)
EVAL_COLUMNS = ("episode", "seed", "reward", "steps", "terminated", "truncated")


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_model():
    ref = base.reference()
    env = base.make_env()
    try:
        net = ref["common"].AtariA2C(env.observation_space.shape, env.action_space.n)
    finally:
        env.close()
    return net


def make_worker_env(num_envs: int):
    return base.CountedVectorEnv(gym.vector.SyncVectorEnv(
        [functools.partial(base.make_env) for _ in range(num_envs)]))


def worker_main(wid: int, seed: int, shared_net, weights_lock, stop_event,
                results_queue, num_envs: int, batch_size: int):
    """Gradientes localmente calculados; pesos globais sao somente leitura para workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    env = None
    last_reported_steps = 0
    pending_episodes = []
    try:
        worker_seed = seed + wid * 100003
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        ref = base.reference()
        env = make_worker_env(num_envs)
        net = ref["common"].AtariA2C(env.single_observation_space.shape,
                                     env.single_action_space.n)
        with weights_lock:
            net.load_state_dict(shared_net.state_dict())
        net.train()
        agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device="cpu", apply_softmax=True)
        source = ptan.experience.VectorExperienceSourceFirstLast(
            env, agent, gamma=ref["GAMMA"], steps_count=ref["REWARD_STEPS"],
            env_seed=worker_seed)
        batch = []
        for exp in source:
            batch.append(exp)
            for reward, length in source.pop_rewards_steps():
                pending_episodes.append((float(reward), int(length), env.total_steps))
            if stop_event.is_set():
                break
            if len(batch) < batch_size:
                continue

            # Um snapshot de pesos e utilizado para TODOS os termos desse gradiente.
            with weights_lock:
                net.load_state_dict(shared_net.state_dict())
            with torch.no_grad():
                states_t, actions_t, targets_t = ref["common"].unpack_batch(
                    batch, net, device=torch.device("cpu"), gamma=ref["GAMMA"],
                    reward_steps=ref["REWARD_STEPS"])
            batch.clear()
            net.zero_grad(set_to_none=True)
            logits, values = net(states_t)
            v = values.squeeze(-1)
            advantage = targets_t - v.detach()  # CORRECAO: [batch], nao [batch,batch]
            lp = F.log_softmax(logits, dim=1)
            selected = lp[torch.arange(len(actions_t)), actions_t]
            loss_p = -(advantage * selected).mean()
            loss_v = F.mse_loss(v, targets_t)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * lp).sum(dim=1).mean()
            loss_e = -ref["ENTROPY_BETA"] * entropy
            loss_total = loss_p + loss_v + loss_e
            loss_total.backward()
            nn_utils.clip_grad_norm_(net.parameters(), ref["CLIP_GRAD"])

            grads = [p.grad.detach().numpy().copy() if p.grad is not None
                     else np.zeros_like(p.detach().numpy()) for p in net.parameters()]
            rms = np.sqrt(sum(float(np.square(g, dtype=np.float64).sum()) for g in grads)
                          / max(sum(g.size for g in grads), 1))
            metrics = {
                "loss_policy": float(loss_p.item()), "loss_value": float(loss_v.item()),
                "loss_entropy": float(loss_e.item()), "loss_total": float(loss_total.item()),
                "advantage_mean": float(advantage.mean().item()),
                "entropy": float(entropy.item()), "grad_rms": float(rms),
            }
            delta = env.total_steps - last_reported_steps
            results_queue.put(("gradient", wid, delta, pending_episodes, metrics, grads))
            last_reported_steps = env.total_steps
            pending_episodes = []
    except Exception:
        results_queue.put(("error", wid, traceback.format_exc()))
    finally:
        if env is not None:
            try:
                remaining = env.total_steps - last_reported_steps
                results_queue.put(("finished", wid, remaining, pending_episodes))
            except Exception:
                pass
            env.close()
        else:
            results_queue.put(("finished", wid, 0, pending_episodes))


def spawn_model_probe(net):
    """Valida a deserializacao do modelo original em um subprocesso spawn."""
    from lib.common import AtariA2C
    if not isinstance(net, AtariA2C):
        raise RuntimeError(f"Modelo inesperado no subprocesso: {type(net)!r}")
    net.eval()
    with torch.no_grad():
        logits, v = net(torch.zeros((1, 4, 84, 84), dtype=torch.float32))
    if logits.shape[0] != 1 or v.shape != (1, 1):
        raise RuntimeError(f"Saida da rede invalida: {logits.shape}, {v.shape}")


def check_env(procs: int, num_envs: int):
    ref = base.reference()
    vec = make_worker_env(num_envs)
    try:
        obs, _ = vec.reset(seed=42)
        actions = vec.action_space.sample()
        vec.step(actions)
        net = ref["common"].AtariA2C(vec.single_observation_space.shape,
                                       vec.single_action_space.n).cpu()
        net.share_memory()
        with torch.no_grad():
            logits, v = net(torch.from_numpy(obs[:1]).float())
        print(f"OK {ENV_ID} | workers={procs} | ambientes/worker={num_envs} "
              f"| ambientes totais={procs * num_envs} | obs={tuple(obs.shape)} "
              f"| logits={tuple(logits.shape)} | valor={tuple(v.shape)} "
              f"| shared_model=cpu", flush=True)
        # O teste anterior verificava apenas a rede local. Este teste realmente
        # inicia um processo spawn e passa a rede, reproduzindo a etapa que falhou.
        ctx = mp.get_context("spawn")
        probe = ctx.Process(target=spawn_model_probe, args=(net,), name="e07-spawn-probe")
        probe.start()
        probe.join(timeout=60)
        if probe.is_alive():
            probe.terminate()
            probe.join(timeout=5)
            raise RuntimeError("Preflight spawn excedeu 60 segundos")
        if probe.exitcode != 0:
            raise RuntimeError(f"Preflight spawn falhou com exitcode={probe.exitcode}")
        print("OK subprocesso spawn | modulo lib.common importado | modelo compartilhado recebido", flush=True)
    finally:
        vec.close()


def run_one(seed: int, args, ref: dict):
    base.seed_everything(seed)
    torch.set_num_threads(1)
    mode = "stop_on_convergence" if args.stop_on_convergence else "no_stop_on_convergence"
    tag = "stop" if args.stop_on_convergence else "nostop"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_id = (f"a3c_grad_seed{seed}_steps{args.steps}_procs{args.processes}"
              f"_envs{args.num_envs}_{tag}_{stamp}")
    out = RESULTS_BASE / mode / run_id
    out.mkdir(parents=True, exist_ok=False)
    # A rede e as constantes comuns sao carregadas de 02_pong_a2c.py.
    # GRAD_BATCH e TRAIN_BATCH pertencem ao exemplo 03_a3c_grad.py; no E07,
    # sao parametros da CLI, com os mesmos valores padrao do livro (64 e 2).
    constants = {k.lower(): ref[k] for k in (
        "GAMMA", "LEARNING_RATE", "ENTROPY_BETA", "REWARD_STEPS", "CLIP_GRAD"
    )}
    constants["grad_batch"] = args.grad_batch
    constants["train_batch"] = args.train_batch
    config = {
        "experiment": "E07", "algorithm": "a3c_grad", "environment": ENV_ID,
        "reference": "Chapter12/03_a3c_grad.py", "seed": seed,
        "max_training_steps": args.steps, "stop_on_convergence": args.stop_on_convergence,
        "num_processes": args.processes, "num_envs_per_process": args.num_envs,
        "num_envs_total": args.processes * args.num_envs,
        "gradient_batch": args.grad_batch, "train_batch": args.train_batch,
        "evaluation_interval_steps": None if args.stop_on_convergence else args.eval_every,
        "evaluation_episodes": args.eval_episodes, "evaluation_policy": "stochastic",
        "evaluation_seed_base": base.EVAL_SEED_BASE,
        "evaluation_max_episode_steps": args.max_eval_steps,
        "convergence_window": WINDOW, "convergence_mean_reward_strict_greater_than": THRESHOLD,
        "device": "cpu", "worker_start_method": "spawn", "gradients_computed_in_workers": True,
        "optimizer_updated_in_main_process": True,
        "worker_weights_snapshot": "shared_cpu_weights_copied_before_each_gradient_batch",
        "batch_advantage_fix": "squeeze(-1)",
        "step_definition": "sum of wrapped env.step calls across workers",
        "step_timestamp_note": "episode timestamps approximate at arrival of worker gradient batch",
        "train_params": constants,
        "note": "Original CPU shared memory; E04 trained on MPS must not be used for isolated throughput speedup claims",
    }
    write_json(out / "config.json", config)
    print(f"\nE07 | seed={seed} | workers={args.processes} | ambientes/worker={args.num_envs} "
          f"| budget={args.steps} | modo={mode} | device=cpu", flush=True)
    print(f"Resultados: {out}", flush=True)

    net = ref["common"].AtariA2C(args.observation_shape, args.num_actions).cpu()
    net.share_memory()
    opt = optim.Adam(net.parameters(), lr=ref["LEARNING_RATE"], eps=1e-3)
    ctx = mp.get_context("spawn")
    weights_lock = ctx.Lock()
    stop = ctx.Event()
    q = ctx.Queue(maxsize=max(args.processes * 2, 4))
    workers = [ctx.Process(target=worker_main,
                           args=(wid, seed, net, weights_lock, stop, q,
                                 args.num_envs, args.grad_batch),
                           name=f"e07-worker-{wid}") for wid in range(args.processes)]

    episodes = []
    checkpoint_rows, checkpoint_eval_rows = [], []
    update_rows = []
    episode_rewards = []
    worker_stats = {wid: {"gradient_messages": 0, "environment_interactions": 0,
                          "completed_games": 0} for wid in range(args.processes)}
    count_steps = 0
    update_count = 0
    gradient_messages = 0
    episode_count = 0
    converged = None
    last_loss = None
    grad_buffer = None
    metrics_buffer = []
    last_worker_id = -1
    done_workers = set()
    stopping = False
    next_checkpoint = args.eval_every
    prev_log_steps = 0
    begin = time.perf_counter()
    last_log = begin
    evaluation_seconds = 0.0
    stop_reason = "step_budget"

    def checkpoint(kind):
        nonlocal evaluation_seconds
        t0 = time.perf_counter()
        checkpoint_id = f"{kind}_steps{count_steps}"
        rel = Path("checkpoints") / f"{checkpoint_id}.pt"
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": {k: v.detach().cpu().clone()
                                         for k, v in net.state_dict().items()},
                    "experiment": "E07", "seed": seed, "algorithm": "a3c_grad",
                    "training_steps": count_steps, "gradient_updates": update_count,
                    "note": "Pesos apenas; nao permite retomada integral"}, target)
        stats, rows = base.evaluate(net, torch.device("cpu"), args.eval_episodes,
                                    args.max_eval_steps)
        checkpoint_rows.append({
            "checkpoint_id": checkpoint_id, "checkpoint_type": kind,
            "training_steps": count_steps, "gradient_updates": update_count,
            "model_path": str(rel),
            "evaluation_reward_mean": stats["reward_mean"],
            "evaluation_reward_std": stats["reward_std"],
            "evaluation_reward_min": stats["reward_min"],
            "evaluation_reward_max": stats["reward_max"],
        })
        checkpoint_eval_rows.extend({"checkpoint_id": checkpoint_id,
                                     "checkpoint_type": kind,
                                     "training_steps": count_steps, **r} for r in rows)
        evaluation_seconds += time.perf_counter() - t0
        print(f"Checkpoint {checkpoint_id} | avaliacao={stats['reward_mean']:.2f} "
              f"+/- {stats['reward_std']:.2f}", flush=True)
        return stats, rows

    def add_progress(wid, delta, completed_episodes):
        nonlocal count_steps, episode_count, converged, stop_reason, stopping
        count_steps += delta
        worker_stats[wid]["environment_interactions"] += delta
        for reward, length, local_steps in completed_episodes:
            episode_count += 1
            worker_stats[wid]["completed_games"] += 1
            episode_rewards.append(reward)
            avg = float(np.mean(episode_rewards[-WINDOW:]))
            episodes.append({"steps": count_steps, "episode": episode_count,
                             "reward": reward, "mean_reward_100": avg,
                             "worker_id": wid, "episode_length": length})
            if episode_count >= WINDOW and avg > THRESHOLD and converged is None:
                converged = {"steps": count_steps, "episodes": episode_count,
                             "mean_reward_100": avg}
                print(f"PRIMEIRA CONVERGENCIA: {converged}", flush=True)
                if args.stop_on_convergence:
                    stop_reason = "convergence"
                    stopping = True
                    stop.set()
                else:
                    checkpoint("first_convergence")
            if episode_count == 1 or episode_count % args.report_every == 0:
                print(f"steps={count_steps:,} | jogos={episode_count:,} | "
                      f"media100={avg:.2f} | reward={reward:.0f}", flush=True)

    for p in workers:
        p.start()
    try:
        while len(done_workers) < args.processes:
            try:
                item = q.get(timeout=10)
            except queue.Empty:
                dead = [(i, p.name, p.exitcode) for i, p in enumerate(workers)
                        if not p.is_alive() and i not in done_workers]
                if dead:
                    raise RuntimeError(f"Workers encerraram sem comunicar resultado; estados={dead}")
                continue
            kind, wid, *rest = item
            if kind == "error":
                raise RuntimeError(f"Falha do worker {wid}:\n{rest[0]}")
            if kind == "finished":
                delta, games = rest
                add_progress(wid, delta, games)
                done_workers.add(wid)
                continue
            if kind != "gradient":
                raise RuntimeError(f"Mensagem inesperada: {kind}")
            delta, games, metrics, gradients = rest
            add_progress(wid, delta, games)
            worker_stats[wid]["gradient_messages"] += 1
            gradient_messages += 1
            last_worker_id = wid
            if not stopping:
                if grad_buffer is None:
                    grad_buffer = [g.copy() for g in gradients]
                else:
                    for agg, g in zip(grad_buffer, gradients):
                        agg += g
                metrics_buffer.append(metrics)
                if len(metrics_buffer) >= args.train_batch:
                    opt.zero_grad(set_to_none=True)
                    for p, g in zip(net.parameters(), grad_buffer):
                        p.grad = torch.from_numpy(g)
                    grad_rms, grad_max, _ = base.gradient_metrics(net.parameters())
                    before = nn_utils.clip_grad_norm_(net.parameters(), ref["CLIP_GRAD"])
                    with weights_lock:
                        opt.step()
                    update_count += 1
                    last_loss = float(np.mean([v["loss_total"] for v in metrics_buffer]))
                    update_rows.append({
                        "steps": count_steps, "update": update_count,
                        "gradient_messages": gradient_messages, "worker_id_last": wid,
                        "loss_policy_mean": float(np.mean([v["loss_policy"] for v in metrics_buffer])),
                        "loss_value_mean": float(np.mean([v["loss_value"] for v in metrics_buffer])),
                        "loss_entropy_mean": float(np.mean([v["loss_entropy"] for v in metrics_buffer])),
                        "loss_total_mean": last_loss,
                        "advantage_mean": float(np.mean([v["advantage_mean"] for v in metrics_buffer])),
                        "entropy_mean": float(np.mean([v["entropy"] for v in metrics_buffer])),
                        "worker_grad_rms_mean": float(np.mean([v["grad_rms"] for v in metrics_buffer])),
                        "aggregated_grad_rms": grad_rms,
                        "aggregated_grad_max": grad_max,
                        "grad_norm_before_clip": float(before.item()),
                    })
                    grad_buffer = None
                    metrics_buffer = []
                if not args.stop_on_convergence:
                    while next_checkpoint < args.steps and count_steps >= next_checkpoint:
                        checkpoint("periodic")
                        next_checkpoint += args.eval_every
                if count_steps >= args.steps:
                    stop_reason = "step_budget"
                    stopping = True
                    stop.set()
            if count_steps - prev_log_steps >= args.log_every_steps:
                now = time.perf_counter()
                # Exclui avaliacoes do denominador acumulado, inclusive milestones.
                elapsed_training = now - begin - evaluation_seconds
                total_throughput = count_steps / max(elapsed_training, 1e-9)
                print(f"steps={count_steps:,} | updates={update_count:,} | "
                      f"jogos={episode_count:,} | throughput_total={total_throughput:.1f} int/s",
                      flush=True)
                prev_log_steps = count_steps
                last_log = now
        if not stopping:
            raise RuntimeError("Workers encerraram sem atingir orcamento/convergencia")
    except BaseException:
        stop.set()
        raise
    finally:
        stop.set()
        for p in workers:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        q.close()
        q.join_thread()

    wall = time.perf_counter() - begin
    training_seconds = max(0.0, wall - evaluation_seconds)
    final_evaluation, evaluation_rows = checkpoint("final")
    # Estrutura de saida compativel com a analise do E04.
    write_csv(out / "episodes.csv", EPISODE_COLUMNS, episodes)
    write_csv(out / "updates.csv", UPDATE_COLUMNS, update_rows)
    write_csv(out / "checkpoints.csv", CHECKPOINT_COLUMNS, checkpoint_rows)
    write_csv(out / "checkpoint_evaluations.csv",
              ("checkpoint_id", "checkpoint_type", "training_steps", *EVAL_COLUMNS),
              checkpoint_eval_rows)
    write_csv(out / "evaluation.csv", EVAL_COLUMNS, evaluation_rows)
    write_csv(out / "worker_stats.csv",
              ("worker_id", "gradient_messages", "environment_interactions", "completed_games"),
              [{"worker_id": wid, **data} for wid, data in worker_stats.items()])
    summary = {
        "experiment": "E07", "algorithm": "a3c_grad", "seed": seed,
        "training_steps": count_steps, "gradient_updates": update_count,
        "gradient_messages": gradient_messages, "episodes_completed": episode_count,
        "converged": converged is not None, "first_convergence": converged,
        "first_convergence_steps": converged["steps"] if converged else None,
        "train_mean_reward_last100": (float(np.mean(episode_rewards[-WINDOW:]))
                                      if episode_rewards else None),
        "final_evaluation_reward_mean": final_evaluation["reward_mean"],
        "final_evaluation_reward_std": final_evaluation["reward_std"],
        "training_elapsed_seconds_excluding_evaluations": training_seconds,
        "wrapped_interactions_per_training_second": count_steps / max(training_seconds, 1e-9),
        "checkpoint_and_evaluation_seconds": evaluation_seconds,
        "device": "cpu", "stop_reason": stop_reason, "worker_stats": worker_stats,
    }
    write_json(out / "summary.json", summary)
    print(f"FINAL E07 | steps={count_steps:,} | jogos={episode_count:,} | "
          f"convergiu={converged is not None} | aval={final_evaluation['reward_mean']:.2f}",
          flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description="E07 — A3C-grad Pong com multiprocessing")
    p.add_argument("--check-env", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--processes", type=int, default=4,
                   help="Processos calculando gradientes; padrao do livro: 4")
    p.add_argument("--num-envs", type=int, default=8,
                   help="Ambientes por processo; padrao do livro: 8")
    p.add_argument("--grad-batch", type=int, default=64,
                   help="Experiencias por pacote de gradientes; padrao do livro: 64")
    p.add_argument("--train-batch", type=int, default=2,
                   help="Pacotes por atualizacao; padrao do livro: 2")
    p.add_argument("--device", choices=["cpu"], default="cpu",
                   help="A rede compartilhada do A3C-grad original exige CPU")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500_000)
    p.add_argument("--max-eval-steps", type=int, default=20_000)
    p.add_argument("--report-every", type=int, default=20)
    p.add_argument("--log-every-steps", type=int, default=100_000)
    flag = p.add_mutually_exclusive_group()
    flag.add_argument("--stop-on-convergence", dest="stop_on_convergence", action="store_true")
    flag.add_argument("--no-stop-on-convergence", dest="stop_on_convergence", action="store_false")
    p.set_defaults(stop_on_convergence=True)
    args = p.parse_args()
    if min(args.steps, args.processes, args.num_envs, args.grad_batch,
           args.train_batch, args.eval_episodes, args.eval_every) <= 0:
        p.error("Todos os valores numericos de contagem devem ser positivos")
    if args.check_env:
        check_env(args.processes, args.num_envs)
        return
    ref = base.reference()
    if args.processes == 1:
        print("Aviso: E07 com um unico worker nao avalia paralelismo de gradientes")
    env = base.make_env()
    try:
        args.observation_shape = env.observation_space.shape
        args.num_actions = env.action_space.n
    finally:
        env.close()
    for seed in args.seeds:
        run_one(seed, args, ref)


if __name__ == "__main__":
    mp.freeze_support()
    main()

#!/usr/bin/env bash
# E04 (aguarda e analisa) -> E05 (PG) -> E06 (A2C async) -> E07 (A3C-grad).
# Para E05-E07, executa cada semente sequencialmente e nao repete execucoes completas.
# Executar no macOS via nohup + caffeinate; nao fechar a tampa do MacBook.
set -Eeuo pipefail

UNIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$UNIT_DIR/.venv/bin/python"
LOG_DIR="$UNIT_DIR/logs/lote_e05_e07"
STATUS_FILE="$LOG_DIR/status.txt"
LOCK_DIR="$LOG_DIR/.em_execucao"
STEPS=5000000
EVAL_EPISODES=10
EVAL_EVERY=500000
LOG_EVERY=100000
SEEDS=(42 123 2026)
WAIT_INTERVAL=120
MAX_WAIT_E04_SECONDS=43200  # 12h; evita esperar indefinidamente se E04 falhar.

mkdir -p "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "ERRO: outro lote pode estar em execucao. Confira $LOCK_DIR." >&2
    exit 1
fi

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
status() { printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" > "$STATUS_FILE"; }
finalizar() {
    local rc="$1"
    trap - EXIT
    if (( rc == 0 )); then
        status "CONCLUIDO | E04 analisado, E05-E07 treinados e analisados"
        log "LOTE CONCLUIDO COM SUCESSO."
    else
        status "FALHOU | codigo=$rc | consulte o log"
        log "LOTE INTERROMPIDO/FALHOU (codigo=$rc). Verifique o log."
    fi
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap 'finalizar "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "$PYTHON" ]]; then
    log "ERRO: ambiente nao encontrado: $PYTHON"
    exit 2
fi
for file in e05_pong.py e06_pong.py e07_pong.py analise_e04.py analise_e05.py analise_e06.py analise_e07.py; do
    if [[ ! -f "$UNIT_DIR/$file" ]]; then
        log "ERRO: arquivo ausente: $UNIT_DIR/$file"
        exit 2
    fi
done

# Retorna sucesso (0) e imprime o caminho de uma execucao completa compativel.
# Somente resultados completos sao reaproveitados; um treino interrompido recomeca do inicio.
resultado_completo() {
    local exp="$1" seed="$2"
    "$PYTHON" - "$UNIT_DIR" "$exp" "$seed" "$STEPS" "$EVAL_EPISODES" "$EVAL_EVERY" <<'PY'
import json
import sys
from pathlib import Path

unit, exp, seed, steps, eval_n, every = sys.argv[1:]
unit = Path(unit)
seed, steps, eval_n, every = map(int, (seed, steps, eval_n, every))
root = unit / 'results' / exp / 'completo' / 'no_stop_on_convergence'
if exp == 'e06':
    root = root / 'async'
expected = {
    'e04': {'experiment': 'E04', 'algorithm': 'a2c', 'num_envs': 50, 'vectorization': 'sync'},
    'e05': {'experiment': 'E05', 'algorithm': 'pg', 'num_envs': 32, 'vectorization': 'ptan_list_of_envs'},
    'e06': {'experiment': 'E06', 'algorithm': 'a2c', 'num_envs': 50, 'vectorization': 'async'},
    'e07': {'experiment': 'E07', 'algorithm': 'a3c_grad', 'num_processes': 4,
            'num_envs_per_process': 8, 'gradient_batch': 64, 'train_batch': 2},
}[exp]
mandatory = {'episodes.csv', 'evaluation.csv', 'checkpoints.csv', 'updates.csv'}
if exp in ('e04', 'e05', 'e06'):
    mandatory.add('checkpoint_evaluations.csv')
if exp == 'e07':
    mandatory.add('worker_stats.csv')

matches = []
if root.is_dir():
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            config = json.loads((directory / 'config.json').read_text())
            summary = json.loads((directory / 'summary.json').read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if any(config.get(k) != v for k, v in expected.items()):
            continue
        if (config.get('seed') != seed
                or config.get('max_training_steps') != steps
                or config.get('stop_on_convergence') is not False
                or config.get('evaluation_episodes') != eval_n
                or config.get('evaluation_interval_steps') != every):
            continue
        if int(summary.get('training_steps', 0)) < steps:
            continue
        if any(not (directory / name).is_file() for name in mandatory):
            continue
        matches.append(directory)

if len(matches) > 1:
    print(f'ERRO: duplicatas completas para {exp} semente {seed}: {matches}', file=sys.stderr)
    sys.exit(2)
if matches:
    print(matches[0])
    sys.exit(0)
sys.exit(1)
PY
}

log "Inicio do lote. Python: $PYTHON"
log "Orcamento por semente: $STEPS; avaliacoes: $EVAL_EPISODES; intervalos: $EVAL_EVERY."
status "AGUARDANDO_E04"
start_wait="$(date +%s)"
while :; do
    faltam=()
    for seed in "${SEEDS[@]}"; do
        if resultado_completo e04 "$seed" >/dev/null; then
            :
        else
            rc=$?
            if (( rc != 1 )); then
                log "ERRO: falha ao verificar E04 semente $seed (codigo=$rc)."
                exit "$rc"
            fi
            faltam+=("$seed")
        fi
    done
    if (( ${#faltam[@]} == 0 )); then
        log "E04 completo nas 3 sementes."
        break
    fi
    log "Aguardando E04 (sementes pendentes: ${faltam[*]}). Nao iniciando outro treinamento."
    elapsed=$(( $(date +%s) - start_wait ))
    if (( elapsed >= MAX_WAIT_E04_SECONDS )); then
        log "ERRO: E04 nao concluiu dentro do limite de espera de 12 horas."
        exit 3
    fi
    sleep "$WAIT_INTERVAL"
done

log "Gerando analise E04 (3 sementes)."
status "ANALISANDO_E04"
"$PYTHON" "$UNIT_DIR/analise_e04.py" \
    --seeds "${SEEDS[@]}" --steps "$STEPS" --num-envs 50 \
    --eval-episodes "$EVAL_EPISODES" --no-stop-on-convergence

rodar_treinamento() {
    local exp="$1" seed="$2"
    local existente=""
    if existente="$(resultado_completo "$exp" "$seed")"; then
        log "$exp semente $seed: resultado completo encontrado; pulando: $existente"
        return 0
    else
        rc=$?
        if (( rc != 1 )); then
            log "ERRO: falha ao verificar $exp semente $seed (codigo=$rc)."
            exit "$rc"
        fi
    fi
    log "$exp semente $seed: INICIANDO treinamento."
    status "TREINANDO_${exp}_SEED_${seed}"
    case "$exp" in
        e05)
            "$PYTHON" "$UNIT_DIR/e05_pong.py" \
                --seeds "$seed" --steps "$STEPS" --num-envs 32 \
                --eval-episodes "$EVAL_EPISODES" --eval-every "$EVAL_EVERY" \
                --log-every-steps "$LOG_EVERY" --no-stop-on-convergence
            ;;
        e06)
            "$PYTHON" "$UNIT_DIR/e06_pong.py" \
                --seeds "$seed" --steps "$STEPS" --num-envs 50 --vector-mode async \
                --eval-episodes "$EVAL_EPISODES" --eval-every "$EVAL_EVERY" \
                --log-every-steps "$LOG_EVERY" --no-stop-on-convergence
            ;;
        e07)
            "$PYTHON" "$UNIT_DIR/e07_pong.py" \
                --seeds "$seed" --steps "$STEPS" --processes 4 --num-envs 8 \
                --eval-episodes "$EVAL_EPISODES" --eval-every "$EVAL_EVERY" \
                --log-every-steps "$LOG_EVERY" --no-stop-on-convergence
            ;;
        *) log "ERRO: experimento desconhecido: $exp"; exit 4 ;;
    esac
    if ! resultado_completo "$exp" "$seed" >/dev/null; then
        log "ERRO: $exp semente $seed terminou sem gerar resultado completo compativel."
        exit 5
    fi
    log "$exp semente $seed: CONCLUIDO."
}

for seed in "${SEEDS[@]}"; do rodar_treinamento e05 "$seed"; done
log "Gerando analise E05."
status "ANALISANDO_E05"
"$PYTHON" "$UNIT_DIR/analise_e05.py" \
    --steps "$STEPS" --seeds "${SEEDS[@]}" --pg-num-envs 32 --a2c-num-envs 50 \
    --eval-episodes "$EVAL_EPISODES" --no-stop-on-convergence --require-all

for seed in "${SEEDS[@]}"; do rodar_treinamento e06 "$seed"; done
log "Gerando analise E06."
status "ANALISANDO_E06"
"$PYTHON" "$UNIT_DIR/analise_e06.py" \
    --steps "$STEPS" --seeds "${SEEDS[@]}" --num-envs 50 \
    --eval-episodes "$EVAL_EPISODES" --no-stop-on-convergence --require-all

for seed in "${SEEDS[@]}"; do rodar_treinamento e07 "$seed"; done
log "Gerando analise E07."
status "ANALISANDO_E07"
"$PYTHON" "$UNIT_DIR/analise_e07.py" \
    --steps "$STEPS" --seeds "${SEEDS[@]}" --processes 4 --num-envs 8 \
    --a2c-num-envs 50 --eval-episodes "$EVAL_EPISODES" \
    --no-stop-on-convergence --require-all

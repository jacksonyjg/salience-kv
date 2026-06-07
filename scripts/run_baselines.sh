#!/bin/bash
# scripts/run_baselines.sh
# =========================
# 실험 1-A: 6개 베이스라인 + 제안 방법 비교 (Qwen3-4B 주 실험)
#
# 사용법:
#   bash scripts/run_baselines.sh
#   bash scripts/run_baselines.sh --num_samples 20   # 빠른 테스트
#   bash scripts/run_baselines.sh --model phi-3-mini --mode cross

set -e

MODEL="qwen3-4b"
MODE="full"
NUM_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";       shift 2 ;;
        --mode)        MODE="$2";        shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/baselines_${MODEL}_${TIMESTAMP}.log"

ARGS="--model $MODEL --mode $MODE"
[ -n "$NUM_SAMPLES" ] && ARGS="$ARGS --num_samples $NUM_SAMPLES"

echo "[$(date '+%H:%M:%S')] START: Baselines | model=$MODEL mode=$MODE" | tee "$LOG_FILE"

cd "$PROJECT_ROOT"
python experiments/exp1_main_results.py $ARGS 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%H:%M:%S')] DONE. Log: $LOG_FILE"

#!/bin/bash
# scripts/run_ablation.sh
# ========================
# 실험 2 + 실험 3: Ablation Study 전체
#   - 실험 2: Hybrid Score 4개 성분 기여도
#   - 실험 3: 레이어별 예산 할당 전략 비교
#
# 사용법:
#   bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh --num_samples 20
#   bash scripts/run_ablation.sh --exp 2          # 실험 2만
#   bash scripts/run_ablation.sh --exp 3          # 실험 3만

set -e

MODEL="qwen3-4b"
BUDGET="0.20"
NUM_SAMPLES=""
EXP="all"   # all | 2 | 3

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";       shift 2 ;;
        --budget)      BUDGET="$2";      shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --exp)         EXP="$2";         shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ablation_${MODEL}_${TIMESTAMP}.log"

COMMON="--model $MODEL --budget $BUDGET"
[ -n "$NUM_SAMPLES" ] && COMMON="$COMMON --num_samples $NUM_SAMPLES"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

log "START: Ablation Study | model=$MODEL budget=$BUDGET"
cd "$PROJECT_ROOT"

# 실험 2: Score 성분 기여도
if [ "$EXP" = "all" ] || [ "$EXP" = "2" ]; then
    log "--- Exp2: Ablation Score Components ---"
    python experiments/exp2_ablation_score.py $COMMON 2>&1 | tee -a "$LOG_FILE"
    log "Exp2 DONE"
fi

# 실험 3: 레이어 할당 전략
if [ "$EXP" = "all" ] || [ "$EXP" = "3" ]; then
    log "--- Exp3: Ablation Allocation Strategy ---"
    python experiments/exp3_ablation_allocation.py $COMMON 2>&1 | tee -a "$LOG_FILE"
    log "Exp3 DONE"
fi

log "Ablation COMPLETE. Log: $LOG_FILE"

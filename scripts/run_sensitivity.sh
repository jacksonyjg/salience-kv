#!/bin/bash
# scripts/run_sensitivity.sh
# ===========================
# 실험 4 + 실험 5: 민감도 분석
#   - 실험 4: 예산 비율 민감도 (10%~50%)
#   - 실험 5: 하이퍼파라미터 민감도 (λ, 가중치)
#
# 사용법:
#   bash scripts/run_sensitivity.sh
#   bash scripts/run_sensitivity.sh --num_samples 20
#   bash scripts/run_sensitivity.sh --exp 4   # 실험 4만
#   bash scripts/run_sensitivity.sh --exp 5   # 실험 5만

set -e

MODEL="qwen3-4b"
NUM_SAMPLES=""
EXP="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";       shift 2 ;;
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
LOG_FILE="$LOG_DIR/sensitivity_${MODEL}_${TIMESTAMP}.log"

NS_ARG=""
[ -n "$NUM_SAMPLES" ] && NS_ARG="--num_samples $NUM_SAMPLES"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

log "START: Sensitivity Analysis | model=$MODEL"
cd "$PROJECT_ROOT"

# 실험 4: 예산 민감도
if [ "$EXP" = "all" ] || [ "$EXP" = "4" ]; then
    log "--- Exp4: Budget Sensitivity ---"
    python experiments/exp4_budget_sensitivity.py \
        --model "$MODEL" $NS_ARG 2>&1 | tee -a "$LOG_FILE"
    log "Exp4 DONE"
fi

# 실험 5: 하이퍼파라미터 민감도
if [ "$EXP" = "all" ] || [ "$EXP" = "5" ]; then
    log "--- Exp5: Hyperparameter Sensitivity ---"
    python experiments/exp5_hyperparam_sensitivity.py \
        --model "$MODEL" --mode all $NS_ARG 2>&1 | tee -a "$LOG_FILE"
    log "Exp5 DONE"
fi

log "Sensitivity COMPLETE. Log: $LOG_FILE"

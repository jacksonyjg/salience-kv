#!/bin/bash
# scripts/run_all.sh
# ===================
# 논문 전체 실험 순차 실행 (실험계획서 §8 순서 준수)
#
# RunPod tmux 사용법:
#   tmux new -s kvcache
#   bash scripts/run_all.sh
#   Ctrl+B → D  (분리)
#   tmux attach -t kvcache  (재접속)
#
# 사용법:
#   bash scripts/run_all.sh                  # 전체 샘플
#   bash scripts/run_all.sh --quick          # 샘플 20개 (빠른 검증)
#   bash scripts/run_all.sh --model phi-3-mini  # 특정 모델만

set -e  # 오류 발생 시 즉시 중단

# ── 인자 파싱 ──────────────────────────────
MODEL="qwen3-4b"
NUM_SAMPLES=""
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";       shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --quick)       QUICK=true;       shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$QUICK" = true ]; then
    NUM_SAMPLES=20
fi

# ── 경로 설정 ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/run_all_${MODEL}_${TIMESTAMP}.log"

# ── 공통 인자 ──────────────────────────────
COMMON_ARGS="--model $MODEL"
if [ -n "$NUM_SAMPLES" ]; then
    COMMON_ARGS="$COMMON_ARGS --num_samples $NUM_SAMPLES"
fi

# ── 로깅 함수 ──────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$MASTER_LOG"; }
run_exp() {
    local desc="$1"; shift
    log "START: $desc"
    python "$@" 2>&1 | tee -a "$MASTER_LOG"
    log "DONE : $desc"
    echo "" | tee -a "$MASTER_LOG"
}

# ── 실험 시작 ──────────────────────────────
log "=============================="
log "KV Cache Experiment START"
log "Model      : $MODEL"
log "Num samples: ${NUM_SAMPLES:-all}"
log "Quick mode : $QUICK"
log "Log        : $MASTER_LOG"
log "=============================="

cd "$PROJECT_ROOT"

# 0. Sanity check
run_exp "Sanity Check" \
    experiments/sanity_check.py --model "$MODEL"

# 1-A. 주 실험 (전체 베이스라인)
run_exp "Exp1-A: Main Results (${MODEL})" \
    experiments/exp1_main_results.py $COMMON_ARGS --mode full

# 1-B. 교차 아키텍처 Phi-3
if [ "$MODEL" = "qwen3-4b" ]; then
    run_exp "Exp1-B: Cross-Arch Phi-3-mini" \
        experiments/exp1_main_results.py \
        --model phi-3-mini --mode cross \
        ${NUM_SAMPLES:+--num_samples $NUM_SAMPLES}

    # 1-C. 교차 아키텍처 Gemma-2
    run_exp "Exp1-C: Cross-Arch Gemma-2-2B" \
        experiments/exp1_main_results.py \
        --model gemma-2-2b --mode cross \
        ${NUM_SAMPLES:+--num_samples $NUM_SAMPLES}
fi

# 2. Ablation - Score 성분
run_exp "Exp2: Ablation Score Components" \
    experiments/exp2_ablation_score.py $COMMON_ARGS

# 3. Ablation - 레이어 할당
run_exp "Exp3: Ablation Allocation Strategy" \
    experiments/exp3_ablation_allocation.py $COMMON_ARGS

# 4. 예산 민감도
run_exp "Exp4: Budget Sensitivity" \
    experiments/exp4_budget_sensitivity.py --model "$MODEL" \
    ${NUM_SAMPLES:+--num_samples $NUM_SAMPLES}

# 5. 하이퍼파라미터 민감도
run_exp "Exp5: Hyperparameter Sensitivity" \
    experiments/exp5_hyperparam_sensitivity.py $COMMON_ARGS --mode all

# 6. 계산 오버헤드
run_exp "Exp6: Overhead Analysis" \
    experiments/exp6_overhead.py --model "$MODEL"

# ── 완료 ───────────────────────────────────
log "=============================="
log "ALL EXPERIMENTS COMPLETED"
log "Results: $PROJECT_ROOT/results/"
log "=============================="

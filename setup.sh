#!/bin/bash
# setup.sh
# =========
# RunPod 환경 초기 설정 스크립트
# 처음 Pod 생성 후 한 번만 실행
#
# 사용법:
#   bash setup.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

# ── 1. Python 패키지 설치 ──────────────────
echo ""
echo "[1/4] Installing Python packages..."
pip install -r "$PROJECT_ROOT/requirements.txt" -q

# ── 1.5 환경변수 설정 ──────────────────────
echo ""
echo "[1.5/4] Setting environment variables..."
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
source ~/.bashrc
echo "  PYTORCH_CUDA_ALLOC_CONF set."


# ── 2. 디렉토리 생성 ───────────────────────
echo ""
echo "[2/4] Creating directories..."
mkdir -p "$PROJECT_ROOT"/{logs,results/{longbench,latency,memory},figures}

# ── 3. HuggingFace 토큰 설정 (비공개 모델 접근 시) ──
echo ""
echo "[3/4] HuggingFace token check..."
if [ -z "$HF_TOKEN" ]; then
    echo "  HF_TOKEN not set."
    echo "  Qwen3/Gemma 등 일부 모델은 토큰 필요 시:"
    echo "    export HF_TOKEN=hf_xxxx"
    echo "    huggingface-cli login --token \$HF_TOKEN"
else
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    echo "  HuggingFace login done."
fi

# ── 4. 모델/데이터셋 다운로드 (선택) ─────────
echo ""
echo "[4/4] Model & Dataset download (optional)..."
echo "  현재: HuggingFace Hub에서 실험 실행 시 자동 다운로드"
echo ""
echo "  LOCAL 전환 시 아래 명령어로 미리 다운로드:"
echo ""
echo "  # 모델 다운로드"
echo "  # huggingface-cli download Qwen/Qwen3-4B \\"
echo "  #     --local-dir /workspace/models/Qwen3-4B"
echo "  # huggingface-cli download microsoft/Phi-3-mini-128k-instruct \\"
echo "  #     --local-dir /workspace/models/Phi-3-mini-128k-instruct"
echo "  # huggingface-cli download google/gemma-2-2b-it \\"
echo "  #     --local-dir /workspace/models/gemma-2-2b-it"
echo ""
echo "  # 데이터셋 다운로드"
echo "  # huggingface-cli download THUDM/LongBench \\"
echo "  #     --repo-type dataset \\"
echo "  #     --local-dir /workspace/datasets/longbench"

# ── 완료 ───────────────────────────────────
echo ""
echo "=============================="
echo "Setup complete!"
echo ""
echo "다음 단계:"
echo "  # 환경 확인"
echo "  python3 experiments/sanity_check.py --model qwen3-4b --full_check"
echo ""
echo "  # 개별 TABLE 재현 (예: TABLE VI)"
echo "  python3 experiments/exp_table6_sink_intervention.py \\"
echo "      --budget 0.20 --tasks qmsum gov_report --num_samples 30 --invert_norm"
echo ""
echo "  # 전체 TABLE 목록 및 정확한 재현 명령은 README.md의"
echo "  # Manuscript-to-Code Mapping 섹션 참고 (tmux 세션 권장)"
echo "=============================="


#!/bin/bash
# setup.sh
# =========
# Initial environment setup (run once after creating a new pod / machine).
#
# Usage:
#   bash setup.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

# ── 1. Python packages ─────────────────────
echo ""
echo "[1/4] Installing Python packages..."
pip install -r "$PROJECT_ROOT/requirements.txt" -q
echo "  Note: transformers==5.0.0 is required (see README)."

# ── 1.5 Environment variables ──────────────
echo ""
echo "[1.5/4] Setting environment variables..."
if ! grep -q "PYTORCH_CUDA_ALLOC_CONF" ~/.bashrc 2>/dev/null; then
    echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "  PYTORCH_CUDA_ALLOC_CONF set."

# ── 2. Directories ─────────────────────────
echo ""
echo "[2/4] Creating directories..."
mkdir -p "$PROJECT_ROOT"/{logs,results/{longbench,latency,memory},figures}

# ── 3. HuggingFace token (for gated models) ─
echo ""
echo "[3/4] HuggingFace token check..."
if [ -z "$HF_TOKEN" ]; then
    echo "  HF_TOKEN not set."
    echo "  Some models (e.g. Gemma) require accepting a license first:"
    echo "    export HF_TOKEN=<your token>"
    echo "    hf auth login --token \$HF_TOKEN"
else
    hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
        || huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
        || true
    echo "  HuggingFace login done."
fi

# ── 4. Optional pre-download ───────────────
echo ""
echo "[4/4] Model & dataset download (optional)..."
echo "  Models and LongBench are fetched from the HuggingFace Hub on first run."
echo ""
echo "  To pre-download:"
echo "  # hf download Qwen/Qwen3-4B --local-dir /workspace/models/Qwen3-4B"
echo "  # hf download microsoft/Phi-3-mini-128k-instruct \\"
echo "  #     --local-dir /workspace/models/Phi-3-mini-128k-instruct"
echo "  # hf download THUDM/LongBench --repo-type dataset \\"
echo "  #     --revision 5e628be450b7e67fb7ae6e201bd6d8f7056f7672 \\"
echo "  #     --local-dir /workspace/datasets/longbench"

# ── Done ───────────────────────────────────
echo ""
echo "=============================="
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  # environment check"
echo "  python3 experiments/sanity_check.py --model qwen3-4b --full_check"
echo ""
echo "  # reproduce one table (e.g. TABLE 7, sink-size intervention)"
echo "  python3 -u experiments/exp_table6_sink_intervention.py \\"
echo "      --model qwen3-4b --budget 0.20 --tasks qmsum gov_report \\"
echo "      --num_samples 30 --seed 42 --invert_norm"
echo ""
echo "  Full table list and exact commands: see the"
echo "  'Manuscript-to-Code Mapping' section in README.md."
echo "  Long runs are best executed under tmux."
echo "=============================="

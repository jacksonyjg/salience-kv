# SalienceKV: KV-Cache Compression and Generation Robustness

Research code and experimental artifacts for the manuscript
**"An Empirical Study on KV Cache Compression in Small Language Models: Signal Composition, Repetition Collapse, and Early-Position Retention"** (prepared for submission to *IEEE Access*).

---

## Overview

This project studies how KV-cache compression affects both **task quality** and **generation stability** under aggressive memory budgets.

The study addresses three research questions:

1. At SLM scale, how do the composition and weighting of the importance signal affect task quality and generation stability, and are these two affected in the same way?
2. Can aggregate task score mask repetition instability in compressed generation, and under what evaluated conditions does that instability emerge or diminish?
3. What role does early-position retention play in generation stability, and what does a controlled removal-and-rescue intervention on that retention show?

The main evaluation uses **Qwen3-4B**, with cross-architecture validation on **Phi-3-mini**.

---

## Scope of the Study

### Models

- **Qwen3-4B** — primary model
- **Phi-3-mini-128k** — cross-architecture validation

Gemma-2-2b was explored during development but is **not** included in the final quantitative comparison: the evaluation harness does not implement the `HybridCache` semantics required by its mixed sliding-window/global-attention design. This reflects a limitation of the harness, not of the architecture.

### Long-context tasks

Seven LongBench tasks: NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, GovReport, QMSum.

The LongBench dataset revision is pinned in `core/dataset_loader.py`.

---

## Evaluated Configurations

FullKV · StreamingLLM · H2O-adapted · SnapKV-adapted · PyramidKV-adapted · AdaKV-adapted · SalienceKV (w/o sink) · SalienceKV-Sink-4

### Important note on adapted baselines

H2O, SnapKV, PyramidKV, and AdaKV are implemented as **adapted proxy baselines under a unified cache-compression interface**, not as exact reproductions of the published algorithms.

Under this interface all four replace attention-based importance computation with a common key-norm proxy, and all operate under a uniform per-layer cache length. Results characterize **these adapted configurations**, not the original published methods.

---

## Implementation Notes

### 1. Key-norm direction

Canonical runs prioritize **low L2-norm keys** and use:

```text
invert_norm=True
```

Legacy development runs using the opposite direction are retained for traceability only.

### 2. Fixed KV-cache budget

Explicitly preserved early positions are counted **inside** the fixed total budget, not added on top of it. Under a 20% retention budget, reserving four early positions reduces the score-selected budget accordingly.

### 3. Prompt templates

| Task | Template |
|---|---|
| QMSum | **Official LongBench** query-focused instruction (Transcript/Query/Answer) within the model-specific chat wrapper |
| GovReport | Separate query-free summarization instruction written for this study |
| QA tasks (5) | Common reading-comprehension instruction written for this study |

⚠️ Because only QMSum uses the official LongBench instruction, and because F1/ROUGE-L are computed by `core/metrics.py` rather than the official LongBench evaluator, **absolute scores are not directly comparable to published LongBench results.** All comparisons in the manuscript are within this evaluation setting, where every configuration sees identical prompts and scoring.

### 4. Generation settings

```text
seed              42
do_sample         False   (deterministic)
enable_thinking   False   (Qwen3)
max_new_tokens    512
input cap         16,000 tokens (head+tail truncation: first 500 tokens + tail)
```

Generation termination honours **all** end-of-sequence identifiers declared in each model's generation configuration.

### 5. Repetition-collapse criterion

Implemented in `core/collapse_metrics.py`:

```text
collapse  =  word 3-gram repetition ratio > 0.3
          OR char 5-gram repetition ratio > 0.7
```

The ratio follows the sequence-level repetition measure `rep-n = 1 − U_n/T_n` used in the neural text degeneration literature. **The two cutoffs are study-specific operational thresholds**, not externally validated boundaries.

---

## Environment

```text
Python:          3.12.3
PyTorch:         2.8.0 (CUDA 12.8)
Transformers:    5.0.0
Datasets:        2.21.0
Primary GPU:     NVIDIA A40 (46 GB), RunPod
Seed:            42
LongBench rev.:  5e628be450b7e67fb7ae6e201bd6d8f7056f7672
```

Install:

```bash
pip install -r requirements.txt
```

⚠️ `transformers==5.0.0` is required. The cache implementation relies on the
`cache.layers[i].keys / .values` structure introduced in Transformers 5.x and
will not run on 4.x.

### Model revisions are not pinned

Model checkpoints are loaded with `from_pretrained(model_name)` without a pinned
revision. Re-running with a later checkpoint published under the same model name
may not reproduce the reported values exactly. This limitation is stated in the
manuscript (Appendix B).

---

## Quick Start

```bash
git clone https://github.com/jacksonyjg/salience-kv.git
cd salience-kv
pip install -r requirements.txt

# environment check
python3 experiments/sanity_check.py --model qwen3-4b --full_check
```

---

## Example Reproduction Commands

```bash
# Main comparison (TABLE 5 / TABLE 6)
python3 -u experiments/exp1_main_results.py \
  --model qwen3-4b --mode full --budget 0.20 \
  --num_samples 30 --seed 42 --invert_norm

# Sink-size intervention at 20% retention (TABLE 7)
python3 -u experiments/exp_table6_sink_intervention.py \
  --model qwen3-4b --budget 0.20 --tasks qmsum gov_report \
  --num_samples 30 --seed 42 --invert_norm

# Sink-size intervention at 80% retention (TABLE 7)
python3 -u experiments/exp_table6_sink_intervention.py \
  --model qwen3-4b --budget 0.80 --tasks qmsum gov_report \
  --num_samples 30 --seed 42 --invert_norm
```

Long runs are best executed under `tmux` or with `nohup ... &`.

---

## Manuscript-to-Code Mapping

Experiment script filenames reflect development history and **do not match the
final manuscript table numbers.** Use this table, not the internal number printed
by a script. `PAPER_TABLE_MAPPING.md` gives the full artifact-level detail
including SHA-256 digests.

| Manuscript | Experiment | Script | Canonical artifact(s) in `results/final/` |
|---|---|---|---|
| **TABLE 5** | Main task results | `experiments/exp1_main_results.py` | `exp1_qwen3-4b_full_20260820_112755.json` (6 tasks) + `exp1_qwen3-4b_full_20260822_220917.json` (QMSum) |
| **TABLE 6** | Repetition-collapse survey | same run as TABLE 5 | same as above |
| **TABLE 7** | Sink-size intervention (20% / 80%) | `experiments/exp_table6_sink_intervention.py` | `exp6_sink_intervention_qwen3-4b_budget20_20260820_143453.json` · `..._budget80_20260820_200418.json` + QMSum `..._budget20_20260822_224940.json` · `..._budget80_20260823_011727.json` |
| **TABLE 8** | Signal ablation (a)(b) | `experiments/exp_table7_signal_ablation.py` + `exp_table7_extra_signals.py` | `exp7_signal_ablation_qwen3-4b_sink0_20260819_130428.json` · `..._sink4_20260819_202543.json` · `exp7_extra_signals_qwen3-4b_20260819_151144.json` · `..._20260819_223806.json` + QMSum `..._20260823_043358 / 050944 / 054206 / 055407.json` |
| **TABLE 9** | Position/content controlled validation | `experiments/exp_table9_position_content.py` | `table13_position_content_20260821_152211_merged_v2.json` + QMSum `table13_position_content_20260823_101113.json` |
| **TABLE 10(a)** | Phi-3 compression results | `experiments/exp_table10_cross_arch_sink.py` | `exp10_crossarch_phi3_20260822_055401.json` + QMSum `..._20260823_111747.json` |
| **TABLE 10(b)** | Phi-3 removal-and-rescue intervention | `experiments/exp_table10b_removal_rescue.py` | `phi3_h2o_causal_test.json` |
| **TABLE 11** | Weight sensitivity | `experiments/exp_table12_weight_sensitivity.py` | `exp12_weight_sensitivity_qwen3-4b_20260822_015451.json` + QMSum `..._20260823_092617.json` |
| **TABLE 12** | Budget sensitivity | `experiments/exp_table8_budget_sensitivity.py` | `exp8_budget_sensitivity_qwen3-4b_20260821_152826.json` + QMSum `..._20260823_060549.json` |
| **TABLE 13** | Efficiency benchmark | `experiments/exp_table7_efficiency_v2.py` | `table7_v2_efficiency_20260822_141505.json` |

### Note on the two-file composition

Most tables combine a **base run** (six or seven tasks) with a **corrected QMSum
rerun**. The QMSum prompt originally dropped the query for this query-focused
task; after the fix, QMSum alone was re-run and merged with the earlier results
for the remaining tasks. Both files are listed above; the merge is arithmetic
(per-task substitution of the QMSum column).

---

## Artifact Hierarchy

| Path | Status |
|---|---|
| `results/final/`, `logs/final/` | **Canonical.** Only the files listed in the Manuscript-to-Code Mapping back reported values; the directory also holds intermediate verified runs |
| `results/superseded/`, `logs/superseded/` | Earlier verified runs, replaced by `final/` |
| `results/pre_correction/`, `logs/pre_correction/` | Produced **before** the evaluation-pipeline correction described in Appendix B of the manuscript. Retained for traceability. **These do not correspond to any reported value** |
| `experiments/development/` | Verification and superseded scripts. **Not manuscript results.** See `experiments/development/README.md` |
| `legacy/` | Historical V1 pipeline and superseded experiment scripts. See `legacy/README.md` |

Internal script table numbers (e.g. "Table VI" in a log line) are **development-era
numbers** and must not be used to infer the final manuscript table.

## Repository Structure

```text
salience-kv/
├── core/
│   ├── model_loader.py          # model loading, prompt construction, tokenization
│   ├── dataset_loader.py        # LongBench loading, pinned dataset revision
│   ├── evaluator_v2.py          # evaluation and generation pipeline
│   ├── kv_cache_hook.py         # KV-cache selection / compression
│   ├── collapse_metrics.py      # repetition-collapse criterion
│   ├── metrics.py               # F1 / ROUGE-L
│   └── results_manager.py       # CSV / JSON serialization
│
├── experiments/                 # canonical per-table scripts
│   └── development/             # verification & superseded scripts (not canonical)
│
├── scripts/                     # run wrappers
├── results/                     # final/ · superseded/ · pre_correction/
├── logs/                        # final/ · superseded/ · pre_correction/
├── legacy/                      # historical V1 pipeline
│
├── setup.sh
├── requirements.txt
├── PAPER_TABLE_MAPPING.md
├── LICENSE
└── README.md
```

---

## Deprecated — Do Not Use for Reproduction

| Script | Reason |
|---|---|
| `experiments/exp_table7_efficiency.py` (v1) | Repeated a single prompt; superseded by `exp_table7_efficiency_v2.py` (30 distinct prompts) |
| `experiments/development/superseded/exp_table9_efficiency_UNUSED.py` | No `torch.cuda.synchronize()`, no real generation |
| `experiments/development/superseded/exp6_overhead.py` | Does not pass `invert_norm` / `sink_size`, no real generation, no CUDA sync |
| `experiments/development/verification_architecture/table10_cross_arch_check.py`, `table10_gemma_eos_fix_check.py` | N=2 diagnostics, no `invert_norm` support |

---

## Efficiency Benchmark Design

Context lengths 2,048 / 4,096 / 8,192 tokens. **30 distinct long-context prompts**
(10 each from GovReport, QMSum, NarrativeQA), reused across lengths for paired
scaling comparison. Method order is rotated across prompts; CUDA synchronization
is applied around each timed region.

---

## Citation

This work is currently under review. If you use this code, please cite:

```bibtex
@unpublished{yi2026saliencekv,
  title  = {An Empirical Study on KV Cache Compression in Small Language
            Models: Signal Composition, Repetition Collapse, and
            Early-Position Retention},
  author = {Yi, Jaegyun},
  year   = {2026},
  note   = {Manuscript under review}
}
```

The citation will be updated upon publication.

## License

MIT — see [LICENSE](LICENSE).

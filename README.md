# SalienceKV: KV-Cache Compression and Generation Robustness

Research code and experimental artifacts for a study of generation robustness under aggressive KV-cache compression in long-context language models.

> **Repository status:** Manuscript in preparation.  
> This repository is currently under active development and will be made public after manuscript submission.  
> The public release will include a submission-tagged snapshot of the code and canonical experimental artifacts used in the manuscript.

---

## Overview

This project studies how KV-cache compression affects both **task quality** and **generation stability** under aggressive memory budgets.

The study focuses on three questions:

1. Can conventional aggregate task scores hide severe generation failures such as repetition collapse?
2. How do different KV-cache selection strategies behave when the cache budget is strongly constrained?
3. Does preserving or implicitly retaining early-position tokens improve generation robustness?

The experiments distinguish **task correctness** from **generation robustness** and investigate early-position anchoring as a strong robustness factor under aggressive KV-cache compression.

The main empirical evaluation uses **Qwen3-4B**, with additional cross-architecture robustness validation on **Phi-3 Mini**.

---

## Scope of the Study

### Models

- **Qwen3-4B** — primary model
- **Phi-3 Mini** — cross-architecture validation

Gemma-2 was explored during development but is not included in the final quantitative compression comparison because the evaluation path used in this study does not implement the HybridCache semantics required for its mixed sliding-window/global-attention architecture at long sequence lengths.

### Long-context tasks

The Qwen3-4B experiments use seven LongBench tasks:

- NarrativeQA
- Qasper
- MultiFieldQA-en
- HotpotQA
- 2WikiMQA
- GovReport
- QMSum

The LongBench dataset revision is pinned in `core/dataset_loader.py` for reproducibility.

---

## Evaluated KV-Cache Methods

The evaluation framework includes:

- FullKV
- StreamingLLM
- H2O-adapted
- SnapKV-adapted
- PyramidKV-adapted
- AdaKV-adapted
- SalienceKV without explicit sink preservation
- SalienceKV with explicit early-token preservation

### Important note on adapted baselines

H2O, SnapKV, PyramidKV, and AdaKV are implemented as **adapted proxy baselines under a unified cache-compression interface**.

They are used for controlled comparative evaluation in the experimental framework and should **not** be interpreted as exact reproductions of the authors' original repositories or every implementation detail of the corresponding papers.

The exact method settings used for each manuscript experiment are defined in the corresponding scripts under `experiments/`.

---

## Important Implementation Notes

### 1. Key-norm direction

The corrected experiments prioritize **low L2-norm keys** when key norm is used as an importance signal.

Canonical runs use:

```text
invert_norm=True
```

Legacy development runs that used the opposite direction are retained only for traceability and are not used as canonical manuscript results.

### 2. Fixed KV-cache budget

Explicitly preserved early tokens are counted **inside the fixed total KV-cache budget**.

They are not added on top of the nominal retention budget.

For example, under a 20% retention budget, allocating four early-position tokens reduces the remaining score-selected token budget accordingly.

### 3. QMSum prompt handling

Canonical QMSum experiments use the official LongBench query-focused instruction template within the model-specific chat wrapper.

Pre-freeze development artifacts using earlier prompt configurations are retained only for traceability and are non-canonical.

GovReport remains a query-free summarization task and follows its separate summarization path.

### 4. EOS handling

Generation termination follows all end-of-sequence identifiers specified in each model's generation configuration.

This is especially important for models such as Qwen3 and Phi-3 that define multiple valid EOS identifiers.

### 5. Repetition-collapse evaluation

Generation robustness is evaluated using a predefined repetition-collapse criterion implemented in:

```text
core/collapse_metrics.py
```

Reported collapse rates refer to this predefined detector and should not be interpreted as a universal measure of all possible forms of generation degeneration.

Qualitative inspection is used only as supporting analysis where explicitly stated.

---

## Repository Structure

```text
salience-kv/
├── core/
│   ├── model_loader.py          # model loading, prompt construction, tokenization
│   ├── dataset_loader.py        # LongBench loading and pinned dataset revision
│   ├── evaluator_v2.py          # evaluation and generation pipeline
│   ├── kv_cache_hook.py         # KV-cache selection/compression implementation
│   ├── collapse_metrics.py      # repetition-collapse detection
│   ├── metrics.py               # task-quality metrics
│   └── results_manager.py       # CSV / JSON result serialization
│
├── experiments/
│   ├── exp1_main_results.py
│   ├── exp_table6_sink_intervention.py
│   ├── exp_table7_signal_ablation.py
│   ├── exp_table7_extra_signals.py
│   ├── exp_table8_budget_sensitivity.py
│   ├── exp_table7_efficiency_v2.py
│   ├── exp_table10_cross_arch_sink.py
│   ├── exp_table12_weight_sensitivity.py
│   ├── sanity_check.py
│   ├── debug_0822/
│   ├── deprecated/
│   └── debug_*/
│
├── scripts/
│   └── diagnostics/
│       └── table13_position_content_validation.py
│
├── results/
│   └── v3_verified/             # verified experimental artifacts
│
├── logs/
│   └── v3_verified/             # execution logs for verified runs
│
├── legacy/                      # historical development artifacts; non-canonical
├── configs/
├── setup.sh
├── requirements.txt
└── README.md
```

Files under `legacy/`, `experiments/deprecated/`, `experiments/debug_*`, intermediate sanity runs, and superseded result files are retained for research traceability.

They are **not** canonical manuscript results unless explicitly listed in the manuscript-to-code mapping below.

---

## Environment

The experiments were run on NVIDIA GPUs using PyTorch and Hugging Face Transformers.

Core dependencies are listed in:

```bash
requirements.txt
```

Install with:

```bash
pip install -r requirements.txt
```

Before public release, the exact environment used for the canonical manuscript experiments should be frozen and recorded here:

```text
Python:          [TO FILL AT SUBMISSION]
PyTorch:         [TO FILL AT SUBMISSION]
Transformers:    [TO FILL AT SUBMISSION]
Datasets:        2.21.0
CUDA:            [TO FILL AT SUBMISSION]
Primary GPU:     NVIDIA A40
Seed:            42
LongBench rev.:  [PINNED IN core/dataset_loader.py]
```

A frozen dependency snapshot such as `requirements-lock.txt` or an equivalent environment file should be added before public release.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/jacksonyjg/salience-kv.git
cd salience-kv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a basic environment check:

```bash
python3 experiments/sanity_check.py --model qwen3-4b --full_check
```

---

## Example Reproduction Commands

### Main comparison — Qwen3-4B

```bash
python3 -u experiments/exp1_main_results.py \
  --model qwen3-4b \
  --mode full \
  --budget 0.20 \
  --num_samples 30 \
  --seed 42 \
  --invert_norm
```

### Sink-size intervention — 20% retention

```bash
python3 -u experiments/exp_table6_sink_intervention.py \
  --model qwen3-4b \
  --budget 0.20 \
  --tasks qmsum gov_report \
  --num_samples 30 \
  --seed 42 \
  --invert_norm
```

### Sink-size intervention — 80% retention

```bash
python3 -u experiments/exp_table6_sink_intervention.py \
  --model qwen3-4b \
  --budget 0.80 \
  --tasks qmsum gov_report \
  --num_samples 30 \
  --seed 42 \
  --invert_norm
```

The exact commands used for every reported manuscript table will be frozen in the submission release.

---

## Manuscript-to-Code Mapping

The experiment script numbering reflects the development history and does **not always match the final manuscript table number**.

Use the mapping below rather than the internal table number printed by a script.

| Manuscript Table | Experiment | Primary script(s) | Canonical artifact |
|---|---|---|---|
| Table II | Main task-quality results | `experiments/exp1_main_results.py` | `[FINAL MERGED ARTIFACT TO BE FROZEN]` |
| Table III | Repetition-collapse survey | `experiments/exp1_main_results.py` | `[FINAL MERGED ARTIFACT TO BE FROZEN]` |
| Table IV | Sink intervention at 20% and 80% retention | `experiments/exp_table6_sink_intervention.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table V | Signal ablation | `experiments/exp_table7_signal_ablation.py` + `experiments/exp_table7_extra_signals.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table VI | Budget sensitivity | `experiments/exp_table8_budget_sensitivity.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table VII | Efficiency benchmark | `experiments/exp_table7_efficiency_v2.py` | `results/v3_verified/table7_v2_efficiency_20260822_141505.json` |
| Table VIII-A | Phi-3 compression results | `experiments/exp_table10_cross_arch_sink.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table VIII-B | Controlled Phi-3 early-anchor intervention | `experiments/debug_0822/verify_h2o_causal_intervention.py` | `results/v3_verified/phi3_h2o_causal_test.json` |
| Table IX | Weight sensitivity | `experiments/exp_table12_weight_sensitivity.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table X | Position/content intervention | `scripts/diagnostics/table13_position_content_validation.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |

### Current Table II–III source composition

The current Table II–III manuscript values combine:

```text
Base six-task run:
results/v3_verified/exp1_qwen3-4b_full_20260820_112755.json

Corrected QMSum targeted rerun:
results/v3_verified/exp1_qwen3-4b_full_20260822_220917.json
```

Before public release, these should preferably be merged into a single canonical manuscript artifact so that users do not need to manually reconstruct the final table.

---

## Canonical Results Policy

Only files explicitly listed in the **Manuscript-to-Code Mapping** section are considered canonical manuscript results.

The repository contains additional development artifacts because the study included multiple validation, ablation, and debugging stages.

These files are preserved to maintain traceability, but they must not be mixed with the final manuscript results.

In particular:

- `results/v3_verified/` may contain both final and intermediate verified runs.
- `legacy/` contains historical experiments.
- debug and sanity artifacts are not manuscript results unless explicitly mapped above.
- superseded prompt configurations are non-canonical.
- superseded efficiency experiments are non-canonical.
- internal script table numbers should not be used to infer the final manuscript table number.

---

## Deprecated / Non-Canonical Experiments

The following scripts are retained only for development traceability and must **not** be used to reproduce manuscript results.

### Superseded efficiency benchmark

```text
experiments/exp_table7_efficiency.py
```

This was the v1 efficiency benchmark based on repeated measurements of a single prompt and was replaced by:

```text
experiments/exp_table7_efficiency_v2.py
```

The v2 benchmark uses **30 distinct LongBench-derived prompts** (10 each from GovReport, QMSum, and NarrativeQA) and reuses the same document set across controlled context lengths.

### Deprecated timing script

```text
experiments/deprecated/exp_table9_efficiency_UNUSED.py
```

This script does not implement the final timing methodology and is non-canonical.

### Legacy overhead script

```text
experiments/exp6_overhead.py
```

This script does not implement the final evaluation protocol and must not be used for manuscript efficiency results.

### Diagnostic-only cross-architecture scripts

```text
scripts/diagnostics/table10_cross_arch_check.py
scripts/diagnostics/table10_gemma_eos_fix_check.py
```

These are diagnostic scripts rather than quantitative manuscript experiments.

The canonical Phi-3 quantitative experiment is:

```text
experiments/exp_table10_cross_arch_sink.py
```

with the controlled early-anchor intervention implemented separately in:

```text
experiments/debug_0822/verify_h2o_causal_intervention.py
```

---

## Efficiency Benchmark Design

The final efficiency benchmark uses controlled context lengths of:

- 2,048 tokens
- 4,096 tokens
- 8,192 tokens

The benchmark uses **30 distinct long-context prompts** drawn from LongBench:

- 10 GovReport prompts
- 10 QMSum prompts
- 10 NarrativeQA prompts

The same document set is reused across context lengths to enable paired scaling comparisons.

All methods are evaluated on identical prompts, and method order is rotated across prompts to reduce systematic time-dependent GPU effects.

At a 20% retention budget:

- FullKV retains the full KV cache.
- AdaKV-adapted retains 20% of the KV cache.
- SalienceKV-Sink4 retains 20% of the KV cache.

KV-cache footprint is measured directly from key/value tensor sizes.

Timing uses CUDA synchronization around the measured sections.

The final benchmark should be interpreted as a **system-level efficiency comparison**, not as an independent task-quality evaluation.

---

## Reproducibility and Traceability

For each final manuscript experiment, the public release should provide:

- exact script,
- command-line arguments,
- model revision,
- dataset revision,
- random seed,
- result JSON,
- corresponding execution log,
- manuscript table mapping.

At manuscript submission, the corresponding repository state should be tagged, for example:

```bash
git tag -a submission-v1.0 -m "Code and artifacts for submitted manuscript"
git push origin submission-v1.0
```

The submission tag should remain immutable so that later repository updates do not change the code corresponding to the submitted paper.

---

## Main Research Interpretation

This study distinguishes **task correctness** from **generation robustness**.

Under aggressive KV-cache compression, some score-based eviction methods exhibit substantial repetition collapse even when aggregate task scores alone do not make the instability obvious.

The experiments investigate early-position anchoring as a strong robustness factor.

On Qwen3-4B, explicit or implicit retention of early positions is associated with substantially lower repetition-collapse rates across several eviction strategies.

Controlled position manipulations further show that this effect cannot be reduced to a simple universal rule about specific token identity.

A separate controlled removal-and-rescue intervention on Phi-3 provides strong causal evidence, under the tested setting, that early-position anchoring contributes to generation robustness.

These findings should **not** be interpreted as evidence that early-token preservation is necessary, sufficient, or universally protective across all models and methods.

The specific manifestation of the effect remains architecture- and method-dependent.

---

## Cross-Architecture Validation

Phi-3 Mini is used as an independent architectural setting for robustness validation.

The quantitative Phi-3 analysis contains two components:

### Panel A — Compression comparison

The corrected evaluation pipeline is used to compare multiple KV-cache compression configurations on QMSum and GovReport.

### Panel B — Controlled early-anchor intervention

Under an otherwise fixed H2O-adapted compression setting on GovReport:

- Natural early-position retention: `0/60` collapse
- Blocking positions 0–3: `15/60` collapse
- Restoring position 0 while keeping positions 1–3 excluded: `1/60` collapse

The removal and rescue contrasts provide strong causal evidence under the tested setting that early-position anchoring contributes to generation robustness.

This result is interpreted as **cross-architecture evidence**, not as proof of a universal mechanism.

---

## Known Limitations

- The primary quantitative evaluation is centered on Qwen3-4B.
- Phi-3 Mini provides an independent cross-architecture validation setting, but the specific anchor mechanism remains architecture- and method-dependent.
- Gemma-2 is not included in the quantitative compression comparison because the evaluation harness used in this study does not implement the HybridCache semantics required for its mixed sliding-window/global-attention architecture at long sequence lengths.
- Experiments were executed on data-center GPUs rather than on physical 8GB on-device hardware.
- The study controls KV-cache retention in software but does not reproduce the full latency, power, thermal, and memory-management behavior of an actual on-device deployment.
- The repetition-collapse detector captures a predefined form of degeneration and does not cover every possible generation failure mode.
- Several baseline implementations are adapted proxies under a shared evaluation interface rather than exact reproductions of the original authors' repositories.

---

## Paper

**Title:** `[FINAL MANUSCRIPT TITLE TO BE INSERTED]`

**Target venue:** IEEE Access  
**Status:** Manuscript in preparation

After submission, update the status to:

```text
Status: Submitted to IEEE Access.
```

After acceptance/publication, add:

- final title,
- author list,
- DOI,
- IEEE bibliographic citation.

---

## Citation

Citation information will be finalized after manuscript submission/publication.

```bibtex
@article{saliencekv2026,
  title   = {[FINAL TITLE]},
  author  = {[AUTHOR LIST]},
  journal = {IEEE Access},
  year    = {2026}
}
```

---

## License

A repository license will be finalized before public release after confirming compatibility with all third-party baseline implementations and dependencies.

Third-party models, datasets, and code remain subject to their respective licenses and terms of use.

---

## Acknowledgments

This repository uses publicly available language models and the LongBench benchmark for research evaluation.

Please cite the original model, dataset, and baseline-method papers when reusing this code or reproducing the experiments.

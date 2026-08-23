# SalienceKV: KV-Cache Compression and Generation Robustness

Research code and experimental artifacts for a study of generation robustness under aggressive KV-cache compression in long-context language models.

> **Repository status:** Manuscript in preparation.  
> This repository is currently under active development and will be made public after manuscript submission.  
> The public release will include a submission-tagged snapshot of the code and canonical experimental artifacts.

## Overview

This project studies how KV-cache compression affects both **task quality** and **generation stability** under aggressive memory budgets.

The experiments focus on three questions:

1. Can conventional aggregate task scores hide severe generation failures such as repetition collapse?
2. How do different KV-cache selection strategies behave when the cache budget is strongly constrained?
3. Does preserving or implicitly retaining early-position tokens improve generation robustness?

The repository includes baseline implementations, controlled sink-token interventions, signal ablations, budget sensitivity experiments, efficiency measurements, cross-architecture evaluation, and position/content interventions.

The main empirical focus is on **Qwen3-4B**, with additional cross-architecture evaluation using **Phi-3 Mini**.

## Scope of the Study

### Models

- **Qwen3-4B** — primary model
- **Phi-3 Mini** — cross-architecture validation

Gemma-2 was explored during development but is not included in the final quantitative compression comparison because the current evaluation path does not implement the HybridCache semantics required by its mixed sliding-window/global-attention architecture at long sequence lengths.

### Long-context tasks

Experiments use seven tasks derived from LongBench:

- NarrativeQA
- Qasper
- MultiFieldQA-en
- HotpotQA
- 2WikiMQA
- GovReport
- QMSum

The dataset revision is pinned in `core/dataset_loader.py` for reproducibility.

## Methods

The evaluation framework includes:

- FullKV
- StreamingLLM
- H2O-adapted
- SnapKV-adapted
- PyramidKV-adapted
- AdaKV-adapted
- SalienceKV without explicit sink preservation
- SalienceKV with explicit early-token preservation

The exact method settings used for each manuscript experiment are defined in the corresponding scripts under `experiments/`.

## Important Implementation Notes

### Key-norm direction

The corrected experiments prioritize **low L2-norm keys** when key norm is used as an importance signal.

Runs used for the final manuscript use:

```text
invert_norm=True
```

Legacy development runs that used the opposite direction are retained only for traceability and are not used as canonical manuscript results.

### Fixed KV-cache budget

Explicitly preserved early tokens are counted **inside the fixed total KV-cache budget**. They are not added on top of the nominal cache budget.

### QMSum prompt handling

QMSum uses the official LongBench query-focused instruction template within the model-specific chat wrapper.

Earlier development runs used a generic summarization prompt that omitted the QMSum query. That issue was identified and corrected before the final manuscript results were frozen.

Legacy results generated before this correction are retained for traceability but are not used as canonical final results.

### EOS handling

Generation uses all EOS token IDs defined by each model's generation configuration when applicable.

### Repetition-collapse evaluation

Generation robustness is evaluated using a predefined repetition-collapse criterion implemented in the evaluation code. Reported collapse rates refer only to this predefined detector and should not be interpreted as a universal measure of all possible degeneration modes.

## Repository Structure

```text
salience-kv/
├── core/
│   ├── model_loader.py          # model loading, prompt construction, tokenization
│   ├── dataset_loader.py        # LongBench loading and pinned dataset revision
│   ├── evaluator_v2.py          # evaluation and generation pipeline
│   ├── kv_base.py               # KV-cache utilities
│   ├── kv_methods.py            # KV-cache compression / eviction methods
│   ├── metrics.py               # task-quality metrics
│   └── results_manager.py       # CSV / JSON result serialization
│
├── experiments/
│   ├── exp1_main_results.py
│   ├── exp_table6_sink_intervention.py
│   ├── exp_table7_signal_ablation.py
│   ├── exp_table8_budget_sensitivity.py
│   ├── exp_table7_efficiency_v2.py
│   ├── exp_table10_cross_arch_sink.py
│   ├── exp_table12_weight_sensitivity.py
│   ├── sanity_check.py
│   ├── deprecated/
│   └── debug_*/
│
├── results/
│   └── v3_verified/             # verified experimental artifacts
│
├── logs/
│   └── v3_verified/             # execution logs for verified runs
│
├── legacy/                      # historical development artifacts; non-canonical
├── configs/
├── scripts/
├── setup.sh
├── requirements.txt
└── README.md
```

> Files under `legacy/`, `experiments/deprecated/`, `experiments/debug_*`, intermediate sanity runs, and superseded result files are retained for research traceability. They are **not** canonical manuscript results unless explicitly listed in the table mapping below.

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

For the submission release, the exact software environment used for the canonical experiments should be recorded here:

```text
Python:          [TO FILL AT SUBMISSION]
PyTorch:         [TO FILL AT SUBMISSION]
Transformers:    [TO FILL AT SUBMISSION]
Datasets:        2.21.0
CUDA:            [TO FILL AT SUBMISSION]
Primary GPU:     NVIDIA A40
Seed:            42
LongBench rev.:  [pinned in core/dataset_loader.py]
```

A frozen dependency snapshot should be added before public release.

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/jacksonyjg/salience-kv.git
cd salience-kv
pip install -r requirements.txt
```

Run a basic environment check:

```bash
python3 experiments/sanity_check.py --model qwen3-4b --full_check
```

## Example Reproduction Commands

### Main comparison

```bash
python3 -u experiments/exp1_main_results.py \
  --model qwen3-4b \
  --mode full \
  --budget 0.20 \
  --num_samples 30 \
  --seed 42 \
  --invert_norm
```

### Sink-size intervention

20% KV-cache budget:

```bash
python3 -u experiments/exp_table6_sink_intervention.py \
  --model qwen3-4b \
  --budget 0.20 \
  --tasks qmsum gov_report \
  --num_samples 30 \
  --seed 42 \
  --invert_norm
```

80% KV-cache budget:

```bash
python3 -u experiments/exp_table6_sink_intervention.py \
  --model qwen3-4b \
  --budget 0.80 \
  --tasks qmsum gov_report \
  --num_samples 30 \
  --seed 42 \
  --invert_norm
```

The exact commands used for every reported table will be frozen and listed in the submission release.

## Manuscript-to-Code Mapping

The script numbering reflects the development history and does not always match the final paper table number.

| Manuscript Table | Experiment | Primary script | Canonical result |
|---|---|---|---|
| Table II–III | Main task-quality and collapse comparison | `experiments/exp1_main_results.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table IV | Sink-size intervention at 20% and 80% budgets | `experiments/exp_table6_sink_intervention.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table V | Signal ablation | `experiments/exp_table7_signal_ablation.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table VI | Budget sensitivity | `experiments/exp_table8_budget_sensitivity.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table VII | Efficiency benchmark | `experiments/exp_table7_efficiency_v2.py` | `results/v3_verified/table7_v2_efficiency_20260822_141505.json` |
| Table VIII | Cross-architecture robustness | `experiments/exp_table10_cross_arch_sink.py` + controlled Phi-3 intervention | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table IX | Weight sensitivity | `experiments/exp_table12_weight_sensitivity.py` | `[TO FILL AFTER QMSUM FINALIZATION]` |
| Table X | Position/content intervention | corresponding position/content experiment | `[TO FILL AFTER QMSUM FINALIZATION]` |

Before public release, every placeholder above should be replaced with the exact JSON artifact used to generate the submitted manuscript table.

## Canonical Results Policy

Only files explicitly listed in the **Manuscript-to-Code Mapping** section are considered canonical manuscript results.

The repository contains additional development artifacts because the study included multiple validation and debugging stages. These files are preserved to maintain traceability, but they must not be mixed with the final manuscript results.

In particular:

- `results/v3_verified/` contains both final and intermediate verified runs.
- `legacy/` contains historical experiments.
- debug and sanity artifacts are not manuscript results unless explicitly mapped above.
- superseded QMSum results generated before the query-prompt correction are non-canonical.

## Reproducibility and Traceability

For each final manuscript experiment, the public release should provide:

- exact script,
- command-line arguments,
- model and dataset revision,
- random seed,
- result JSON,
- corresponding execution log,
- manuscript table mapping.

At manuscript submission, the corresponding repository state should be tagged, for example:

```bash
git tag -a submission-v1.0 -m "Code and artifacts for submitted manuscript"
git push origin submission-v1.0
```

This tag should remain immutable so that later repository updates do not change the code corresponding to the submitted paper.

## Main Research Interpretation

The study distinguishes **task correctness** from **generation robustness**.

Under the tested aggressive-compression settings, some score-based eviction methods exhibit substantial repetition collapse even when aggregate task scores alone do not make the instability obvious.

The experiments investigate early-position anchoring as a strong robustness factor. Controlled interventions are used to test whether removing or restoring early-position retention changes collapse behavior.

These findings are intentionally bounded to the evaluated models, tasks, compression methods, and cache budgets; they should not be interpreted as evidence that early-token preservation is necessary, sufficient, or universally protective for all language models.

## Known Limitations

- The primary quantitative evaluation is centered on Qwen3-4B.
- Phi-3 Mini is used for cross-architecture validation, but cross-model behavior remains architecture-dependent.
- Gemma-2 is not included in the quantitative compression comparison because the current harness does not implement the HybridCache semantics required by its long-context attention architecture.
- Experiments were executed on data-center GPUs; deployment on physical on-device hardware remains future work.
- The repetition-collapse detector captures a predefined form of generation degeneration and does not cover all possible failure modes.

## Paper

**Title:** `[FINAL MANUSCRIPT TITLE TO BE INSERTED]`

**Venue:** IEEE Access  
**Status:** Manuscript in preparation

After submission, update only the status line:

```text
Status: Submitted to IEEE Access.
```

After publication, add the DOI and final bibliographic citation.

## Citation

Citation information will be added after manuscript submission/publication.

```bibtex
@article{saliencekv2026,
  title   = {[FINAL TITLE]},
  author  = {[AUTHOR LIST]},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

A repository license will be finalized before public release after confirming compatibility with all third-party baseline implementations and dependencies.

Third-party code, models, and datasets remain subject to their respective licenses and terms of use.

## Acknowledgments

This repository uses publicly available language models and the LongBench benchmark for research evaluation. Please cite the original model, dataset, and baseline-method papers when reusing this code.

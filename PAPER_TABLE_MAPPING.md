# Manuscript Table → Artifact Mapping

Experiment script filenames and log lines carry **development-era table numbers**
that do not match the final manuscript. This file is the authoritative mapping
from each manuscript table to the exact artifact files that back it, including
sample counts and SHA-256 digests.

**Manuscript:** *An Empirical Study on KV Cache Compression in Small Language
Models: Signal Composition, Repetition Collapse, and Early-Position Retention*

---

## 1. Numbering change

Table numbers were reassigned during manuscript preparation. Do **not** rely on
the number printed by a script or written in a log line.

| Development-era | Final manuscript | Content |
|---|---|---|
| Table II | **TABLE 5** | Main task results |
| Table III | **TABLE 6** | Repetition-collapse survey |
| Table IV | **TABLE 7** | Sink-size intervention |
| Table V | **TABLE 8** | Signal ablation (a)(b) |
| Table VI | **TABLE 12** | Budget sensitivity |
| Table VII | **TABLE 13** | Efficiency benchmark |
| Table VIII-A / VIII-B | **TABLE 10(a) / 10(b)** | Cross-architecture (Phi-3-mini) |
| Table IX | **TABLE 11** | Weight sensitivity |
| Table X | **TABLE 9** | Position and content controlled validation |

---

## 2. Table → script → artifact

Most tables combine a **base run** with a **corrected QMSum rerun** (see §4).
The merge is arithmetic: the QMSum column is substituted, other tasks unchanged.

| Manuscript | Script | Base artifact | Corrected QMSum artifact |
|---|---|---|---|
| **TABLE 5 / 6** | `experiments/exp1_main_results.py` | `exp1_qwen3-4b_full_20260820_112755.json` | `exp1_qwen3-4b_full_20260822_220917.json` |
| **TABLE 7** | `experiments/exp_table6_sink_intervention.py` | 20%: `exp6_sink_intervention_qwen3-4b_budget20_20260820_143453.json`<br>80%: `..._budget80_20260820_200418.json` | 20%: `..._budget20_20260822_224940.json`<br>80%: `..._budget80_20260823_011727.json` |
| **TABLE 8** | `experiments/exp_table7_signal_ablation.py`<br>`experiments/exp_table7_extra_signals.py` | sink0: `exp7_signal_ablation_qwen3-4b_sink0_20260819_130428.json`<br>sink4: `..._sink4_20260819_202543.json`<br>extra sink0: `exp7_extra_signals_qwen3-4b_20260819_151144.json`<br>extra sink4: `..._20260819_223806.json` | `..._sink0_20260823_043358.json`<br>`..._sink4_20260823_050944.json`<br>`exp7_extra_signals_qwen3-4b_20260823_054206.json`<br>`..._20260823_055407.json` |
| **TABLE 9** | `experiments/exp_table9_position_content.py` | `table13_position_content_20260821_152211_merged_v2.json` | `table13_position_content_20260823_101113.json` |
| **TABLE 10(a)** | `experiments/exp_table10_cross_arch_sink.py` | `exp10_crossarch_phi3_20260822_055401.json` | `..._20260823_111747.json` |
| **TABLE 10(b)** | `experiments/exp_table10b_removal_rescue.py` | `phi3_h2o_causal_test.json` | — (GovReport only) |
| **TABLE 11** | `experiments/exp_table12_weight_sensitivity.py` | `exp12_weight_sensitivity_qwen3-4b_20260822_015451.json` | `..._20260823_092617.json` |
| **TABLE 12** | `experiments/exp_table8_budget_sensitivity.py` | `exp8_budget_sensitivity_qwen3-4b_20260821_152826.json` | `..._20260823_060549.json` |
| **TABLE 13** | `experiments/exp_table7_efficiency_v2.py` | `table7_v2_efficiency_20260822_141505.json` | — (QMSum not involved) |

All paths are relative to `results/final/`.

---

## 3. Canonical artifacts — SHA-256

Filenames differ only by timestamp, so digests are given to make identification
unambiguous. Digests are the first 16 hex characters of the SHA-256.

| Manuscript | Role | File | N | Tasks | SHA-256 (first 16) |
|---|---|---|---:|---|---|
| 5 / 6 | base | `exp1_qwen3-4b_full_20260820_112755.json` | 30 | 7 tasks | `864cf25bb0939a68` |
| 5 / 6 | corrected | `exp1_qwen3-4b_full_20260822_220917.json` | 30 | qmsum | `6bb0805f4677d0cc` |
| 7 | base | `exp6_sink_intervention_qwen3-4b_budget20_20260820_143453.json` | 30 | qmsum, gov_report | `ea52331d18b5e3cf` |
| 7 | corrected | `exp6_sink_intervention_qwen3-4b_budget20_20260822_224940.json` | 30 | qmsum | `ef06098dfd91bb87` |
| 8 | base sink0 | `exp7_signal_ablation_qwen3-4b_sink0_20260819_130428.json` | 30 | 7 tasks | `aea78fa7b98dfadf` |
| 8 | corrected sink0 | `exp7_signal_ablation_qwen3-4b_sink0_20260823_043358.json` | 30 | qmsum | `bbd07b137a6aaa11` |
| 8 | base sink4 | `exp7_signal_ablation_qwen3-4b_sink4_20260819_202543.json` | 30 | 7 tasks | `7bedbc4320764070` |
| 8 | corrected sink4 | `exp7_signal_ablation_qwen3-4b_sink4_20260823_050944.json` | 30 | qmsum | `e0cecd0dddb37fdf` |
| 8 | base extra sink0 | `exp7_extra_signals_qwen3-4b_20260819_151144.json` | 30 | 7 tasks | `b4a799314b1b47aa` |
| 8 | corrected extra sink0 | `exp7_extra_signals_qwen3-4b_20260823_054206.json` | 30 | qmsum | `2889ab36a059f458` |
| 8 | base extra sink4 | `exp7_extra_signals_qwen3-4b_20260819_223806.json` | 30 | 7 tasks | `d6b4d5d80ef7b9fb` |
| 8 | corrected extra sink4 | `exp7_extra_signals_qwen3-4b_20260823_055407.json` | 30 | qmsum | `4356af7e72fa346c` |
| 9 | base | `table13_position_content_20260821_152211_merged_v2.json` | 30 | qmsum, gov_report | `e8cee8021e0a15e2` |
| 9 | corrected | `table13_position_content_20260823_101113.json` | 30 | qmsum | `b626646e0040cace` |
| 10(a) | base | `exp10_crossarch_phi3_20260822_055401.json` | 30 | qmsum, gov_report | `203c2ff39c6a2a60` |
| 10(a) | corrected | `exp10_crossarch_phi3_20260823_111747.json` | 30 | qmsum | `d6ac3d4bc6f41607` |
| 10(b) | single | `phi3_h2o_causal_test.json` | 60 records | gov_report | `e62b7a42fed831a1` |
| 11 | base | `exp12_weight_sensitivity_qwen3-4b_20260822_015451.json` | 30 | 7 tasks | `197b9470131f2f68` |
| 11 | corrected | `exp12_weight_sensitivity_qwen3-4b_20260823_092617.json` | 30 | qmsum (see §5) | `7180aebf1317c7bb` |
| 12 | base | `exp8_budget_sensitivity_qwen3-4b_20260821_152826.json` | 30 | qmsum, hotpotqa, gov_report | `9843cc7842b062dd` |
| 12 | corrected | `exp8_budget_sensitivity_qwen3-4b_20260823_060549.json` | 30 | qmsum | `df761935e8f138a8` |
| 13 | single | `table7_v2_efficiency_20260822_141505.json` | see §5 | — | `1bf14dc3cc0ff7a7` |

### Not canonical — do not use

The following are `num_samples=2` sanity checks with filenames that differ from
the canonical runs only by timestamp:

`exp7_signal_ablation_qwen3-4b_sink0_20260823_041811.json` ·
`exp7_signal_ablation_qwen3-4b_sink4_20260823_042054.json` ·
`exp7_extra_signals_qwen3-4b_20260823_042230.json` ·
`exp7_extra_signals_qwen3-4b_20260823_042322.json`

---

## 4. Corrections applied during development

**QMSum prompt.** `make_prompt()` in `core/model_loader.py` originally routed
QMSum through the summarization branch, which discarded the `question`
parameter. QMSum is a query-focused task, so the query was being dropped. The
fix adopts the official LongBench query-focused instruction and is applied only
when `question` is non-empty; GovReport is unaffected. All affected tables were
re-run for QMSum and merged, which is why most tables list two source files.

**Task-specific content offset.** `table13_position_content_validation.py`
applied a single content offset (14 tokens, derived from GovReport) to all
tasks. QMSum's corrected prompt has a different offset (30 tokens). The offset
is now a per-task dictionary; all 30 samples were re-verified with zero
mismatches.

---

## 5. Notes on JSON metadata

1. **Do not use the `tasks` field to infer experiment scope.**
   `exp12_weight_sensitivity_qwen3-4b_20260823_092617.json` lists all seven
   tasks in `tasks`, but `task_scores` and `sample_records` contain only
   `qmsum`. **Judge scope from the keys of `task_scores`.**

2. **TABLE 13 has no `n` field.** The "30 distinct long-context prompts" figure
   comes from the protocol in `experiments/exp_table7_efficiency_v2.py`, not
   from a numeric field in the JSON.

3. **Three source types.** `numeric` (a JSON holding the value) ·
   `protocol` (the script and commit that fix the experimental condition) ·
   `derived-statistic` (a post-processed value, with its script and correction
   method).

4. **TABLE 8 statistics are a derived statistic.** From the `sample_records` of
   the four N=30 ablation JSONs: exact McNemar paired by `(task, sample_idx)`
   over 210 pairs → Holm–Bonferroni across the eight variants (α = 0.05) →
   task-stratified cluster bootstrap (B = 10,000, seed 42). Recomputation
   reproduces the reported values exactly: 7 of 8 variants significant after
   correction, pooled 10.89% → 0.83%, 95% CI [8.33, 11.90] pp.

---

## 6. Excluded from the quantitative comparison

Gemma-2-2b is not part of the final comparison: the FullKV baseline is not
trustworthy because the evaluation harness does not implement the `HybridCache`
semantics its mixed sliding-window/global-attention design requires. This is a
limitation of the harness, not of the architecture.

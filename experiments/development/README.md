# Development and Verification Scripts

Scripts and outputs used while developing and verifying the implementation.

**None of the files in this directory produce a value reported in the
manuscript.** For the canonical scripts and artifacts behind each table, see
[`PAPER_TABLE_MAPPING.md`](../../PAPER_TABLE_MAPPING.md).

| Directory | Purpose |
|---|---|
| `verification_selection/` | Token-selection rule: sink/window overlap, smoothing behaviour of the score-based baselines |
| `verification_direction/` | Key-norm direction (`invert_norm`): whether low-norm or high-norm keys are prioritized |
| `verification_pipeline/` | Evaluation pipeline: EOS handling, prompt construction, truncation, sample counts, statistical reanalysis |
| `verification_architecture/` | Architecture-specific checks, including the Gemma-2-2b investigations behind the exclusion stated in the manuscript (Section IV-C and Appendix B) |
| `analysis/` | Post-hoc analysis helpers and plotting |
| `superseded/` | Earlier implementations replaced by the canonical scripts. Do not use for reproduction |

## Why these are kept

Two corrections applied during development are documented in the manuscript
(Appendix B) and in `PAPER_TABLE_MAPPING.md`:

- the QMSum prompt originally dropped the query for this query-focused task
- the evaluation pipeline originally advanced the decoding position from the
  compressed cache length rather than from the original input length

The scripts here are the checks that identified and confirmed those issues, and
the checks behind the Gemma-2-2b exclusion. They are published so that these
decisions can be inspected, not because they contribute to any reported number.

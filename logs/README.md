# logs/

| Directory | Status |
|---|---|
| `final/` | **Canonical.** Artifacts behind the reported values. The exact files for each table are listed in [`PAPER_TABLE_MAPPING.md`](../PAPER_TABLE_MAPPING.md); this directory also holds intermediate verified runs |
| `superseded/` | Earlier verified runs, replaced by `final/` |
| `pre_correction/` | Produced **before** the evaluation-pipeline correction described in Appendix B of the manuscript. Retained for traceability. **These do not correspond to any reported value** |

Filenames differ only by timestamp, so `PAPER_TABLE_MAPPING.md` gives SHA-256
digests for the canonical files.

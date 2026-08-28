#!/bin/bash
# REORG.sh — 공개용 저장소 재편 (전체)
#
# 목적
#   1) canonical 스크립트를 experiments/ 최상위로 승격
#   2) 날짜·debug 이름 제거 (debug_0819 → development/verification_*)
#   3) v2/v3 내부 버전 번호를 의미 있는 이름으로 (final / superseded / pre_correction)
#   4) 구식 figure 삭제
#
# 실행:  저장소 루트에서  bash REORG.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== 0. 사전 확인 =="
for d in experiments/debug_0819 experiments/debug_0822 results/v3_verified logs/v3_verified; do
  [ -d "$d" ] || { echo "  !! $d 없음 — 이미 재편되었거나 경로가 다릅니다."; exit 1; }
done
echo "  OK"

# ─────────────────────────────────────────────
echo ""
echo "== 1. canonical 스크립트 승격 =="

git mv experiments/debug_0822/verify_h2o_causal_intervention.py \
       experiments/exp_table10b_removal_rescue.py
echo "  → experiments/exp_table10b_removal_rescue.py        (TABLE 10b)"

git mv scripts/diagnostics/table13_position_content_validation.py \
       experiments/exp_table9_position_content.py
echo "  → experiments/exp_table9_position_content.py        (TABLE 9)"

# ─────────────────────────────────────────────
echo ""
echo "== 2. development/ 신설 및 이동 =="
mkdir -p experiments/development

git mv experiments/debug_0819                  experiments/development/verification_selection
git mv experiments/debug_0819_direction_verify experiments/development/verification_direction
git mv experiments/debug_0822                  experiments/development/verification_pipeline
git mv experiments/deprecated                  experiments/development/superseded
echo "  debug_0819                  → development/verification_selection"
echo "  debug_0819_direction_verify → development/verification_direction"
echo "  debug_0822                  → development/verification_pipeline"
echo "  deprecated                  → development/superseded"

# 아키텍처 검증(Gemma) — 논문 §IV-C · 부록 B.1 의 근거이므로 보존
mkdir -p experiments/development/verification_architecture
for f in diagnose_gemma_collapse.py gemma_short_seq_check.py \
         gemma_swa_boundary_check.py table10_gemma_eos_fix_check.py \
         table10_cross_arch_check.py; do
  if [ -f "scripts/diagnostics/$f" ]; then
    git mv "scripts/diagnostics/$f" "experiments/development/verification_architecture/$f"
  fi
done
echo "  scripts/diagnostics/{gemma*, table10_cross_arch_check} → development/verification_architecture"

# 나머지 diagnostics → development/analysis
mkdir -p experiments/development/analysis
for f in analyze_table6.py analyze_table8.py compare_position_fix.py \
         diagnose_position_bug.py final_smoke_test_all_methods.py \
         find_truncated_samples.py sanity_check_v2_pipeline.py \
         table2_sanity_log.py ttft_check.py verify_truncation_impact.py; do
  if [ -f "scripts/diagnostics/$f" ]; then
    git mv "scripts/diagnostics/$f" "experiments/development/analysis/$f"
  fi
done
echo "  scripts/diagnostics/* (나머지) → development/analysis"

# figure 재생성 스크립트 — 대응 그림이 논문에 없음
if [ -f scripts/figures/regenerate_recent500_plot.py ]; then
  git mv scripts/figures/regenerate_recent500_plot.py \
         experiments/development/analysis/regenerate_recent500_plot.py
fi
rmdir scripts/figures scripts/diagnostics 2>/dev/null || true
echo "  scripts/figures/ → development/analysis"

# ─────────────────────────────────────────────
echo ""
echo "== 3. 구식 figure 삭제 (논문 Fig.1~4 와 무관) =="
if [ -d results/figures ]; then
  git rm -r -q results/figures
  echo "  results/figures/ 삭제"
  echo "    fig1_pipeline / fig2_buggy_vs_fixed / fig3_scoring_sink_structure"
  echo "    fig4_budget_vs_score / fig5_sink_vs_collapse / recent500_vs_collapse_v3"
fi

# ─────────────────────────────────────────────
echo ""
echo "== 4. results / logs 디렉터리명 변경 =="
for base in results logs; do
  if [ -d "$base/v3_verified" ]; then
    git mv "$base/v3_verified" "$base/final";           echo "  $base/v3_verified    → $base/final"
  fi
  if [ -d "$base/v2_verified" ]; then
    git mv "$base/v2_verified" "$base/superseded";      echo "  $base/v2_verified    → $base/superseded"
  fi
  if [ -d "$base/legacy_pre_fix" ]; then
    git mv "$base/legacy_pre_fix" "$base/pre_correction"; echo "  $base/legacy_pre_fix → $base/pre_correction"
  fi
done

# ─────────────────────────────────────────────
echo ""
echo "== 5. README 생성 =="

cat > experiments/development/README.md << 'DOCEOF'
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
DOCEOF
echo "  experiments/development/README.md"

for base in results logs; do
cat > "$base/README.md" << DOCEOF
# $base/

| Directory | Status |
|---|---|
| \`final/\` | **Canonical.** Artifacts behind the reported values. The exact files for each table are listed in [\`PAPER_TABLE_MAPPING.md\`](../PAPER_TABLE_MAPPING.md); this directory also holds intermediate verified runs |
| \`superseded/\` | Earlier verified runs, replaced by \`final/\` |
| \`pre_correction/\` | Produced **before** the evaluation-pipeline correction described in Appendix B of the manuscript. Retained for traceability. **These do not correspond to any reported value** |

Filenames differ only by timestamp, so \`PAPER_TABLE_MAPPING.md\` gives SHA-256
digests for the canonical files.
DOCEOF
echo "  $base/README.md"
done

# ─────────────────────────────────────────────
echo ""
echo "== 6. 결과 =="
echo "  experiments/";             ls -1 experiments/             | sed 's/^/    /'
echo "  experiments/development/"; ls -1 experiments/development/ | sed 's/^/    /'
echo "  results/";                 ls -1 results/                 | sed 's/^/    /'
echo "  logs/";                    ls -1 logs/                    | sed 's/^/    /'

echo ""
echo "== 7. 다음 단계 =="
echo "  # README.md · PAPER_TABLE_MAPPING.md 를 갱신본으로 교체 (경로가 바뀌었음)"
echo "  git status"
echo "  git add -A"
echo "  git commit -m 'refactor: reorganize repository for public release'"
echo "  git push"

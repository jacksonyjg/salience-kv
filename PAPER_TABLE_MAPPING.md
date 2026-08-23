# PAPER_TABLE_MAPPING.md

**목적**: 실험 스크립트 파일명/로그의 "TABLE 번호"와 논문 실제 TABLE 번호가 다름 — 논문 반영 시 잘못된 CSV를 잘못된 표에 붙이는 사고 방지용 1장짜리 매핑표.
**최종 갱신**: 2026-08-23 — TABLE II~X 전체 완료 + QMSum corrected rerun 전체 완료(§22 참고).

## 최종 데이터 출처 (모두 corrected 병합 완료, 아래 파일이 canonical)

| 논문 TABLE | 최종 데이터 출처(legacy + corrected 병합에 쓰인 원본 파일) | 상태 |
|---|---|---|
| TABLE II/III (Main Results) | legacy: `exp1_qwen3-4b_full_20260820_112755.json` (6-task) + corrected: `exp1_qwen3-4b_full_20260822_220917.json` (qmsum만) | ✅ 병합 완료, §0 반영 |
| TABLE IV (Sink Intervention) | legacy 20%: `exp6_sink_intervention_qwen3-4b_budget20_20260820_143453.json`, legacy 80%: `..._budget80_20260820_200418.json` + corrected qmsum 20%: `..._budget20_20260822_224940.json`, corrected qmsum 80%: `..._budget80_20260823_011727.json` | ✅ 병합 완료, §0 반영 |
| TABLE V(a)/(b) (Signal Ablation) | legacy sink0: `exp7_signal_ablation_qwen3-4b_sink0_20260819_130428.json`, sink4: `..._sink4_20260819_202543.json` + corrected qmsum sink0: `..._sink0_20260823_043358.json`, sink4: `..._sink4_20260823_050944.json` | ✅ 병합+통계(McNemar/Holm/stratified bootstrap) 완료, §0/§5 반영. **⚠️ 2026-08-23 정정: corrected 출처가 N=2 sanity check 파일로 잘못 기재돼 있었음 → N=30 본실행 파일로 교체(아래 canonical artifacts 참고)** |
| TABLE V extra (V_only/P_only) | legacy sink0: `exp7_extra_signals_qwen3-4b_20260819_151144.json`, sink4: `..._20260819_223806.json` + corrected qmsum sink0: `..._20260823_054206.json`, sink4: `..._20260823_055407.json` | ✅ 병합 완료, §0 반영. **⚠️ 2026-08-23 정정: 위와 동일 사유로 corrected 출처 교체** |
| TABLE VI (Budget Sensitivity) | legacy: `exp8_budget_sensitivity_qwen3-4b_20260821_152826.json` (qmsum+hotpotqa+gov_report) + corrected qmsum: `..._20260823_060549.json` | ✅ 병합 완료, §0 반영 |
| TABLE VII (Efficiency) | `table7_v2_efficiency_20260822_141505.json`(v2, 30개 독립 프롬프트) | ✅ 완료(QMSum 무관, §0 반영) |
| TABLE VIII Panel A | legacy: `exp10_crossarch_phi3_20260822_055401.json` + corrected qmsum: `..._20260823_111747.json` | ✅ 병합 완료, §0 반영 |
| TABLE VIII Panel B | `phi3_h2o_causal_test.json`(N=60, gov_report only) | ✅ 완료(QMSum 무관) |
| TABLE IX (Weight Sensitivity) | legacy: `exp12_weight_sensitivity_qwen3-4b_20260822_015451.json` (7-task) + corrected qmsum: `..._20260823_092617.json` | ✅ 병합 완료, §0 반영 |
| TABLE X (Position/Content) | legacy: `table13_position_content_20260821_152211_merged_v2.json` (12조건, gov_report+qmsum) + corrected qmsum: `..._20260823_101113.json`(태스크별 content_offset 버그 수정 후) | ✅ 병합+통계 재계산 완료, §0/§15 반영 |

## 실험 스크립트 파일 (전부 `--tasks` CLI 옵션 지원, 오늘 추가)

| 논문 TABLE | 스크립트 파일 | 비고 |
|---|---|---|
| TABLE II/III | `experiments/exp1_main_results.py` | `--tasks qmsum` 이미 지원(기존) |
| TABLE IV | `experiments/exp_table6_sink_intervention.py` | `--tasks qmsum` 이미 지원(기존) |
| TABLE V | `experiments/exp_table7_signal_ablation.py` + `exp_table7_extra_signals.py` | `--tasks qmsum` 이미 지원(기존) |
| TABLE VI | `experiments/exp_table8_budget_sensitivity.py` | `--tasks qmsum` 이미 지원(기존) |
| **TABLE VII** | **`experiments/exp_table7_efficiency_v2.py`**(v1은 폐기) | QMSum 무관 |
| **TABLE VIII Panel A** | `experiments/exp_table10_cross_arch_sink.py` | **오늘 `--tasks` 옵션 신규 추가**(커밋 `1b591ec`) |
| TABLE VIII Panel B | `experiments/debug_0822/verify_h2o_causal_intervention.py` | QMSum 무관 |
| **TABLE IX** | `experiments/exp_table12_weight_sensitivity.py` | **오늘 `--tasks` 옵션 신규 추가**(커밋 `f9b10d5`) |
| **TABLE X** | `scripts/diagnostics/table13_position_content_validation.py` | **오늘 `--tasks` 옵션 신규 추가 + content_offset 태스크별 분리 버그 수정**(커밋 `a4e42c3`) |

## ⚠️ 폐기된 파일 — 절대 사용하지 말 것

| 파일 | 상태 | 이유 |
|---|---|---|
| `experiments/exp_table7_efficiency.py`(v1) | 미사용, 폐기(v2로 완전 대체) | "같은 프롬프트 N회 반복"이 "N=30 sample 실험"과 방법론적으로 다름(GPT 지적) |
| `experiments/deprecated/exp_table9_efficiency_UNUSED.py` | 미사용 | `torch.cuda.synchronize()` 없음, 실제 generation 없음 |
| `experiments/deprecated/exp6_overhead.py` | 미사용, 사용 금지(2026-08-23 `experiments/deprecated/`로 이동) | invert_norm/sink_size 미전달, 실제 generation 없음, CUDA sync 없음 |
| `scripts/diagnostics/table10_cross_arch_check.py`, `table10_gemma_eos_fix_check.py` | 미사용, 진단용으로만 존재 | N=2 진단 스크립트, invert_norm 미지원 |

## 오늘(2026-08-22~23) 발견·수정된 핵심 버그 2건

1. **QMSum 쿼리 누락 버그**(`core/model_loader.py`의 `make_prompt()`): summarization 분기가 `question` 파라미터를 사용 안 함 → LongBench 공식 템플릿(Transcript/Query/Answer)으로 수정. TABLE II~X 전체 targeted rerun 완료.
2. **TABLE X content_offset 태스크별 미분리 버그**(`scripts/diagnostics/table13_position_content_validation.py`): gov_report 기준 offset(14토큰)을 QMSum corrected(실제 30토큰)에도 그대로 적용할 뻔함 → 태스크별 dict로 수정, N=30 전수 검증(mismatch 0) 완료.

## canonical artifacts — 파일명 + N + SHA-256 (2026-08-23 추가)

**추가 이유**: 파일명 패턴이 같고 타임스탬프만 달라 `N=2 sanity check`와 `N=30 본실행`을 구분하지 못한 사고가 실제로 발생했다(TABLE V). 앞으로 source를 적을 때는 **반드시 N과 SHA-256을 함께 기재**한다.

| 논문 TABLE | 구분 | 파일 | N | task 범위 | SHA-256(앞16) |
|---|---|---|---:|---|---|
| II/III | legacy | `exp1_qwen3-4b_full_20260820_112755.json` | 30 | 7 tasks | `864cf25bb0939a68` |
| II/III | corrected | `exp1_qwen3-4b_full_20260822_220917.json` | 30 | qmsum | `6bb0805f4677d0cc` |
| IV | legacy | `exp6_sink_intervention_qwen3-4b_budget20_20260820_143453.json` | 30 | qmsum,gov_report | `ea52331d18b5e3cf` |
| IV | corrected | `exp6_sink_intervention_qwen3-4b_budget20_20260822_224940.json` | 30 | qmsum | `ef06098dfd91bb87` |
| V sink0 main | legacy | `exp7_signal_ablation_qwen3-4b_sink0_20260819_130428.json` | 30 | 7 tasks | `aea78fa7b98dfadf` |
| **V sink0 main** | **corrected** | **`exp7_signal_ablation_qwen3-4b_sink0_20260823_043358.json`** | **30** | qmsum | `bbd07b137a6aaa11` |
| V sink4 main | legacy | `exp7_signal_ablation_qwen3-4b_sink4_20260819_202543.json` | 30 | 7 tasks | `7bedbc4320764070` |
| **V sink4 main** | **corrected** | **`exp7_signal_ablation_qwen3-4b_sink4_20260823_050944.json`** | **30** | qmsum | `e0cecd0dddb37fdf` |
| V sink0 extra | legacy | `exp7_extra_signals_qwen3-4b_20260819_151144.json` | 30 | 7 tasks | `b4a799314b1b47aa` |
| **V sink0 extra** | **corrected** | **`exp7_extra_signals_qwen3-4b_20260823_054206.json`** | **30** | qmsum | `2889ab36a059f458` |
| V sink4 extra | legacy | `exp7_extra_signals_qwen3-4b_20260819_223806.json` | 30 | 7 tasks | `d6b4d5d80ef7b9fb` |
| **V sink4 extra** | **corrected** | **`exp7_extra_signals_qwen3-4b_20260823_055407.json`** | **30** | qmsum | `4356af7e72fa346c` |
| VI | legacy | `exp8_budget_sensitivity_qwen3-4b_20260821_152826.json` | 30 | qmsum,hotpotqa,gov_report | `9843cc7842b062dd` |
| VI | corrected | `exp8_budget_sensitivity_qwen3-4b_20260823_060549.json` | 30 | qmsum | `df761935e8f138a8` |
| VII | single | `table7_v2_efficiency_20260822_141505.json` | (n 필드 없음) | — | `1bf14dc3cc0ff7a7` |
| VIII-A | legacy | `exp10_crossarch_phi3_20260822_055401.json` | 30 | qmsum,gov_report | `203c2ff39c6a2a60` |
| VIII-A | corrected | `exp10_crossarch_phi3_20260823_111747.json` | 30 | qmsum | `d6ac3d4bc6f41607` |
| VIII-B | single | `phi3_h2o_causal_test.json` | 60 (records) | gov_report | `e62b7a42fed831a1` |
| IX | legacy | `exp12_weight_sensitivity_qwen3-4b_20260822_015451.json` | 30 | 7 tasks | `197b9470131f2f68` |
| IX | corrected | `exp12_weight_sensitivity_qwen3-4b_20260823_092617.json` | 30 | ⚠️ 아래 참고 | `7180aebf1317c7bb` |
| X | legacy | `table13_position_content_20260821_152211_merged_v2.json` | 30 | qmsum,gov_report | `e8cee8021e0a15e2` |
| X | corrected | `table13_position_content_20260823_101113.json` | 30 | qmsum | `b626646e0040cace` |

**폐기 확인 (사용 금지)** — 아래는 전부 `num_samples=2` sanity check이며 canonical 아님:
`exp7_signal_ablation_qwen3-4b_sink0_20260823_041811.json`(14 KB) ·
`exp7_signal_ablation_qwen3-4b_sink4_20260823_042054.json`(10 KB) ·
`exp7_extra_signals_qwen3-4b_20260823_042230.json`(2.5 KB) ·
`exp7_extra_signals_qwen3-4b_20260823_042322.json`(2.8 KB)

## JSON 메타데이터 신뢰성 주의 (2026-08-23 추가)

1. **`tasks` 필드를 실험 범위의 근거로 쓰지 말 것.** `exp12_weight_sensitivity_qwen3-4b_20260823_092617.json`(TABLE IX corrected)은 `tasks` 필드에 7개 태스크를 전부 나열하지만, 실제 `task_scores`/`sample_records`에는 `qmsum`만 있다. **실제 범위는 `task_scores`의 키로 판단할 것.**
2. **TABLE VII의 `n=30`은 JSON에 없다.** 결과 객체에 `n` 필드가 없으므로 "30 distinct long-context prompts"는 numeric source가 아니라 **protocol source**(`experiments/exp_table7_efficiency_v2.py`)에 근거한다.
3. **source는 세 종류로 구분한다**: `numeric`(값이 든 JSON) / `protocol`(실험 조건 근거 스크립트·커밋) / `derived-statistic`(후처리 산출값 + 산출 스크립트 + 보정법).
4. **TABLE V 통계의 derived-statistic source**: 위 4개 N=30 JSON의 `sample_records` → exact McNemar(paired by `(task, sample_idx)`, 210쌍) → Holm-Bonferroni(m=8, α=0.05) → task-stratified cluster bootstrap(B=10,000, seed=42). 2026-08-23 재계산으로 §0 값 완전 재현(8/8 variant discordant 구성 일치, Holm 7/8 유의, pooled 10.89%→0.83%, 95% CI [8.33, 11.90]pp).

## 사용 시 주의

- 로그/코드에 찍히는 "Table VI/VII/VIII/IX/XII/XIII" 같은 표기는 **전부 스크립트 작성 당시의 구버전 번호**이며 논문 최종 번호가 아님 — 위 매핑표의 "논문 TABLE" 열만 참고
- Gemma-2-2B는 정량 평가 대상에서 계속 제외(FullKV 기준선이 `HybridCache` 미지원으로 신뢰 불가, §18/§21 참고)

## 저장소 구조 정리 (2026-08-23)

- `legacy/` → `legacy/core/`(옛 evaluator.py/kv_base.py/kv_methods.py 파이프라인) + `legacy/experiments/`(옛 exp1a/exp2/exp3/exp4/exp5/run_all_experiments.py, 전부 `exp_table*` 계열로 대체됨)로 분리. 상세 근거는 `legacy/README.md` 참고.
- `experiments/exp_invert_core.py`, `exp_keynorm_reversal_check.py` → `experiments/debug_0819_direction_verify/`로 이동(key-norm 방향 진단 스크립트)
- 오늘 루트에 흩어져 있던 QMSum/TABLE X 검증 스크립트 7개 → `experiments/debug_0822/`로 이동
- `results/SOURCE_OF_TRUTH.md`(2026-07-04, 극초기 문서) → `results/legacy_pre_fix/SOURCE_OF_TRUTH_20260704.md`로 archived 표시하여 이동

## 논문 반영 시 참고 문서(우선순위 순)

1. **`TABLE_V_정리_및_Contribution_재구성_0820.md`** — §0(Paper-Ready Tables, corrected 최종값) + §20~23(논문 전체 Abstract~Limitations 반영 지침, QMSum 작업 전체 기록) — **사실상 이 파일 하나가 모든 걸 담고 있음, 다른 별도 "논문반영" 파일은 이제 불필요(중복)**
2. GitHub 저장소(`jacksonyjg/salience-kv`) — 실제 원본 데이터 확인 필요 시. 최신 커밋: `0a8de8f`

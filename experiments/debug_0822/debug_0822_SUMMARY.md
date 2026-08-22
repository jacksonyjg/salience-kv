# 2026-08-22 디버깅/검증 세션 요약

이 폴더는 2026-08-22 하루 동안 진행된 검증·디버깅 스크립트 전부를 재현성 근거로 아카이브한 것.
정식 실험(TABLE II~X)과 별개로, 아래 발견들을 뒷받침하는 코드.

## 1. TABLE X 메커니즘 검증
- `verify_content_offset.py`: TABLE X 오프셋(문서 내용 시작 지점=14) 60개 샘플 전수 확인
- `verify_template_eviction.py`: front_real_content=8.3%가 챗 템플릿 손실 때문인지 검증(→ 아님, 노이즈로 확정, McNemar p=0.625)
- `verify_random_sink_mechanism.py`: random_real이 위치 0~3을 실제로 보호하는지 검증(→ 0/60, 전혀 안 함에도 90% 정상 — "protective하지만 필요조건 아님" 확정)
- `verify_snapkv_front_positions.py`: SnapKV의 위치 0,1 결정론적 보존(210/210) 검증

## 2. TABLE VI 관련
- `verify_salience_sink4_budget_reproducibility.py`: SalienceKV-Sink-4 40% budget 유일 collapse(gov_report idx=9) 재현성 확인(knife-edge 패턴 확정)
- `cluster_bootstrap_reanalysis.py`: TABLE V 통계 재분석(cluster bootstrap, 로컬 실행)

## 3. Phi-3 / multi-EOS 버그 발견 및 TABLE VIII
- `verify_phi3_corrected_sanity.py`, `verify_phi3_eos.py`, `verify_phi3_eos_fix.py`, `verify_phi3_full_text.py`: Phi-3의 multi-EOS 버그 발견 과정(Gemma-2와 동일 유형) — "LongRoPE 비호환" 가설 철회의 근거
- `verify_phi3_hf_generate_compare.py`: EvaluatorV2 커스텀 디코딩 루프 vs HF 표준 generate() 대조(FullKV gov_report idx=17)
- `verify_h2o_phi3_mechanism.py`, `verify_h2o_phi3_position_retention.py`: H2O implicit anchoring 메커니즘 규명(위치 1="Please", 31/32 레이어, 레이어별 콘텐츠/위치 혼합 효과)
- `verify_position_content_swap.py`: 위치1↔2 토큰 swap 인과검증(레이어별 winner 분석)
- `verify_h2o_causal_intervention.py`: H2O 위치1 배제 causal test(H2O-natural/no-pos1/no-pos1+pos0, 3-condition)

## 4. Qwen도 multi-EOS였다는 추가 발견 및 audit
- `verify_qwen_151643.py`: Qwen3-4B의 두 번째 EOS(151643=`<|endoftext|>`) 정체 확인
- `verify_qwen_regression_after_eos_patch.py`: multi-EOS patch의 Qwen 회귀 없음 확인(9쌍 exact match, 이후 151643 발견으로 불충분함이 재확인됨)
- `verify_qwen_eos_audit.py` + `audit_targets.json`: TABLE II H2O/PyramidKV/AdaKV collapse 샘플 45개 targeted re-run(151643 발생 여부, prediction 일치 여부 확인)

## 5. TABLE IX 가중치 설정 패치
- `table9_weight_settings_patch.diff`: exp_table12_weight_sensitivity.py에 V-centric/P-centric 추가(적용·커밋 완료됨, 여기 diff는 기록용 보관)

## 핵심 교훈
- `core/evaluator_v2.py`가 원래 단일 `tokenizer.eos_token_id`만 확인하던 버그가 Gemma-2/Phi-3/**Qwen3 전부**에 존재했음이 순차적으로 드러남 — "우리 모델은 단일값이라 안전하다"는 가정을 검증 없이 하면 안 된다는 교훈
- 상관관계(위치가 자주 선택됨)와 인과관계(그래서 안정적임)는 반드시 분리해서 검증해야 함 — H2O implicit anchoring 조사에서 반복적으로 확인
- 자동화된 collapse 지표에도 사각지대가 있음(Phi-3의 토큰쓰레기/유사문구반복 등 verbatim n-gram 반복이 아닌 퇴화 패턴) — 정성적 감사로 보완 필요

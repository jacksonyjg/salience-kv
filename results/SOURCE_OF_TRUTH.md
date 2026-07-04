# 신뢰 가능한 실험 결과 소스 (2026-07-04 확정)

## Exp1-A 50샘플 최종 (4-signal, 4-8k context, 구버전 — 패치 이전)
- results/exp1_qwen3-4b_full_20260628_033401.csv (batch1: narrativeqa,qasper,multifieldqa_en)
- results/exp1_qwen3-4b_full_20260628_063645.csv (batch2: hotpotqa,2wikimqa)
- results/exp1_qwen3-4b_full_20260628_080407.csv (batch3: gov_report,qmsum)
- 검증: 7태스크 가중평균이 기록된 최종값(OURS=17.8 등)과 완전히 일치함 (2026-07-04 확인)
- TTFT/Throughput도 이 3파일 기준이 신뢰 가능한 값 (OURS: TTFT +16.4%, Throughput +11.7%)

## Exp4 budget sensitivity (3태스크만, 7태스크 풀버전 아님 - 재실행 필요)
- (기존 파일명 확인 후 추가 예정)

## 2026-07-04 재실험 (3-signal, 16k context, eager->sdpa 패치 이후) - 정찰 단계, 확정 아님
- results/exp1_recon15_seed42_20260704_075321.log (15샘플, seed=42) - OURS 3위/7 (19.2점)
- results/exp1_recon15_seed123_20260704_095746.log (15샘플, seed=123) - 재현성 확인 중
- Throughput 지표에 이상 현상 있음 (FullKV가 압축방법보다 빠름 - 원인 미규명, memory #8 참조)

## 코드 패치 이력 (2026-07-04)
- core/model_loader.py: qwen3-4b attn_implementation eager->sdpa
- core/evaluator_v2.py: max_input_length=16000 명시
- core/dataset_loader.py: word-level 안전망 4000->40000 완화
- core/kv_cache_hook.py: OursHybridCache use_semantic 기본값 False

## 그 외 results/*.csv, *.log 파일들
- 대부분 디버깅 중 단일 태스크/소샘플 실행 흔적. 정식 결과 아님. 삭제하지 말고 참고용으로만 보관.

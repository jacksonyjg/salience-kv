# Legacy V1 Pipeline (실행하지 않음, 참고용으로만 보관)

이 디렉토리의 코드는 2026-08-11 이후 사용하지 않습니다.

- `evaluator.py`, `kv_methods.py`, `kv_base.py`: V1 평가 파이프라인.
  `core/evaluator_v2.py`와 동일 계열의 position 버그가 있음이 확인됨
  (`_run_decode()`에서 `position_ids = actual_past_len`로 압축캐시를
  "원본의 연속된 앞부분"으로 잘못 가정).
- `exp3_ablation_allocation.py`: 레이어별 동적 예산 "크기" 할당 전략
  비교 실험(H2 가설). 이 실험은 한 번도 실행된 적이 없음. HF `generate()`
  API가 레이어마다 다른 캐시 길이를 지원하지 않아(모든 레이어가 균일한
  캐시 길이를 가져야 함), 애초에 이 파이프라인 구조에서는 의미 있게
  테스트할 수 없다고 결론 내림 - AdaKV/PyramidKV의 레이어 적응형 설계가
  동일한 이유로 무력화된 것과 같은 아키텍처 제약. 논문 V5에 이미 이
  한계가 기록되어 있음. 코드는 참고용으로만 보관.

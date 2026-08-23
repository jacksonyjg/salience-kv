# Legacy V1 Pipeline (실행하지 않음, 참고용으로만 보관)

이 디렉토리의 코드는 2026-08-11 이후 사용하지 않습니다.

```text
legacy/
├── core/           # 옛 V1 평가 파이프라인 (core/evaluator_v2.py 이전 버전)
└── experiments/    # 옛 실험 스크립트 (exp_table* 계열로 대체됨)
```

## legacy/core/

- `evaluator.py`, `kv_methods.py`, `kv_base.py`: V1 평가 파이프라인.
  `core/evaluator_v2.py`와 동일 계열의 position 버그가 있음이 확인됨
  (`_run_decode()`에서 `position_ids = actual_past_len`로 압축캐시를
  "원본의 연속된 앞부분"으로 잘못 가정).

## legacy/experiments/

- `exp3_ablation_allocation.py`: 레이어별 동적 예산 "크기" 할당 전략
  비교 실험(H2 가설). 이 실험은 한 번도 실행된 적이 없음. HF `generate()`
  API가 레이어마다 다른 캐시 길이를 지원하지 않아(모든 레이어가 균일한
  캐시 길이를 가져야 함), 애초에 이 파이프라인 구조에서는 의미 있게
  테스트할 수 없다고 결론 내림 - AdaKV/PyramidKV의 레이어 적응형 설계가
  동일한 이유로 무력화된 것과 같은 아키텍처 제약. 논문 V5에 이미 이
  한계가 기록되어 있음. 코드는 참고용으로만 보관.

### 2026-08-23 추가 아카이브 (초기 evaluator_v2 기반 실험, 이후 `exp_table*` 계열로 완전히 대체됨)

- `exp1a_confirm.py`: Exp1-A 확인용 재실행. docstring에 "논문 미반영, 순수 확인 목적"이라
  명시돼 있음.
- `exp2_ablation_score.py`: Hybrid Score 구성요소 기여도 분석 초기 버전. 최종 canonical은
  `experiments/exp_table7_signal_ablation.py` + `exp_table7_extra_signals.py`(TABLE V).
- `exp4_budget_sensitivity.py`: Budget Sensitivity 분석 초기 버전. 최종 canonical은
  `experiments/exp_table8_budget_sensitivity.py`(TABLE VI).
- `exp5_hyperparam_sensitivity.py`: 하이퍼파라미터(λ, 가중치) 민감도 분석 초기 버전. 최종
  canonical은 `experiments/exp_table12_weight_sensitivity.py`(TABLE IX).
- `run_all_experiments.py`: 위 exp2/exp4/exp5 및 `experiments/deprecated/exp6_overhead.py`를
  순서대로 호출하던 구형 마스터 러너. 현재는 각 TABLE 스크립트를 sanity(N=2) → 본실행(N=30)
  순서로 개별 실행하는 방식으로 대체됨(README의 Manuscript-to-Code Mapping 참고).

이 5개 파일이 참조하던 `core/evaluator_v2.py`, `core/kv_cache_hook.py` 등 core 모듈 자체는
바뀌지 않았음(legacy/core/의 V1 파이프라인처럼 별도 버그가 있는 게 아님) - 단지 최종 논문
TABLE 구조와 맞지 않는 초기 실험 설계라 대체된 것.

## 참고: key-norm 방향 진단 스크립트는 이 디렉토리 대상이 아님

`exp_invert_core.py`, `exp_keynorm_reversal_check.py`는 key-norm 선택 방향(low-norm vs
high-norm 우선) 버그 조사용 진단 스크립트라, 이 legacy pipeline과는 성격이 달라 별도로
`experiments/debug_0819_direction_verify/`에 보관됨.

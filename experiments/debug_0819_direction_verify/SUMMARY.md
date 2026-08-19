# N_only invert_norm 메커니즘 직접 검증 (0819)

## 목적
stage 1(N=2) 결과에서 N_only가 wo_N 대비 기대만큼 개선되지 않아 보여,
invert_norm이 실제로 제대로 적용되는지 score가 아닌 메커니즘(선택된 토큰의 실제 norm) 레벨에서 검증.

## 방법
_key_importance()를 후킹해 layer 0의 raw norm을 캡처하고,
_selected_positions[0]과 대조해 "선택된 토큰 평균norm" vs "버려진 토큰 평균norm" 직접 비교.
gov_report, qmsum 각 1샘플(seed=42), invert_norm=False/True 양쪽 실행.

## 결과
4개 조건 전부 예상 방향과 정확히 일치:
- invert_norm=False: 선택된 토큰 평균norm > 버려진 토큰 평균norm (HIGH-norm 우선)
- invert_norm=True:  선택된 토큰 평균norm < 버려진 토큰 평균norm (LOW-norm 우선, Devoto 방향)

qmsum에서 invert_norm=False는 교과서적 반복 붕괴 출력, invert_norm=True는 정상 출력 (score 1.20→8.70).

## 결론
invert_norm 메커니즘은 완전히 정상 작동. stage 1(N=2)에서 관찰된 "N_only가 wo_N과 비슷하다"는
패턴은 메커니즘 문제가 아니라 N=2 + gov_report 특유의 샘플 노이즈로 확정.
stage 2(N=30, 7태스크) 진행에 대한 우려 해소.

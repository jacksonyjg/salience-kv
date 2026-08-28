# 0819 SnapKV 0% collapse 원인 규명 (디버깅 세션)

## 배경
invert_core 실험(20%/80% x inv=True/False)에서 SnapKV만 20% budget 전 태스크 collapse=0%.
다른 baseline(h2o/pyramidkv/adakv)은 norm 방향 수정(inv=True) 후에도 잔존 collapse 있음.

## 검증 순서 및 결과 (gov_report, 20%, inv=True, sink_size=0, N=15)
1. window 16→32만 변경: collapse 53.3%→46.7% (거의 무변화, window는 원인 아님)
2. kernel smoothing(avg_pool1d, k=5) 추가(window=16 유지): collapse 53.3%→0.0%
3. window=32+smoothing=5 동시 적용: 0.0% (smoothing 단독과 동일, window 추가 이득 없음)
4. sink-position(0~3) 겹침 분석:
   - smoothing 없음: 평균 1/4 겹침 (항상 위치 2)
   - smoothing=5: 평균 2/4 겹침 (항상 위치 0~1, 15/15 샘플 동일 — 구조적 아티팩트 의심)
5. 확정 실험: smoothing=5 + 위치0~3 강제 배제(score=-inf 마스킹): collapse 0.0%→66.7%

## 결론
SnapKV의 0% collapse는 독립적 "국소연속성" 메커니즘이 아니라,
avg_pool1d의 zero-padding 경계 효과로 sink 위치(0~3)를 우연히 재포착한 것이 지배적 원인.
sink(위치 보존)가 여전히 붕괴 해소의 핵심 메커니즘이라는 기존 결론을 재확인/강화함.

## 참고: PyramidKV≈AdaKV 수치 완전 일치 재확인
generate() API 레이어별 가변 캐시 길이 미지원 → 균일 budget 강제 → 두 방법 다
동일 key-norm scoring으로 수렴 (기존 문서화된 결론과 일치, 버그 아님).

## 코드 변경 없음
전부 monkeypatch 기반 디버그 스크립트(core/kv_cache_hook.py 원본 미수정).
다음 세션에서 corrected_v1 실험 설계 시 이 결론을 E0/E11 해석에 반영 필요.

## 추가 확인 (0820, TABLE IV corrected 본실행 sanity check 중 재발견)

TABLE IV(exp_table6_sink_intervention.py, corrected 방향, N=2 sanity check)에서 **SnapKV의
m0/m1/m2(sink_size=0/1/2) 점수가 소수점까지 완전히 동일**하게 나오는 현상 재관찰
(gov_report, N=2: 세 조건 모두 score=15.95).

위 결론(smoothing이 sink_size=0에서도 위치 0~1을 우연히 선점)이 맞다면, sink_size=1·2로
"강제 보존"해봤자 이미 자연 선택되어 있던 위치라 실제 선택 토큰 집합 자체가 안 바뀔 것이라는
예측이 나옴 → 직접 검증(`_selected_positions` 캡처, gov_report 1샘플, sink=0/1/2/4 비교):

```
sink=0  score=15.95  선택 토큰=1472개
sink=1  score=15.95  선택 토큰=1472개  (m0와 완전 동일 집합)
sink=2  score=15.95  선택 토큰=1472개  (m0와 완전 동일 집합)
sink=4  score=16.35  선택 토큰=1472개  (여기서 처음 집합이 달라짐)
```

**확정**: m0==m1==m2는 우연한 점수 근접이 아니라 **선택된 토큰 집합 자체가 진짜로 동일**해서
발생. sink=4부터 처음으로 실제 차이 발생. 버그 아님 — 위 0819 결론(smoothing의 우연한 sink
위치 재포착)의 직접적·필연적 귀결로 최종 확정. TABLE IV 본실행(N=30)에서도 SnapKV
m0=m1=m2가 동일하게 재현될 가능성 높음 — 발견 시 이 문서를 근거로 정상 처리할 것.

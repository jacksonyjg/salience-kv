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

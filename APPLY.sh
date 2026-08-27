#!/bin/bash
# 저장소 공개 전 정리 — 적용 스크립트
# RunPod 저장소 루트에서 실행:  bash APPLY.sh
set -e

echo "== 1. configs/ 삭제 (코드가 읽지 않는 사문화 파일) =="
# 사전 확인: 참조가 정말 없는지
if grep -rn "configs/\|import yaml\|yaml\.safe_load" --include="*.py" --include="*.sh" . \
     | grep -v "^\./configs/" | grep -v APPLY.sh ; then
  echo "  !! configs 참조가 발견되었습니다. 삭제를 중단합니다."
  exit 1
fi
git rm -r -q configs/ && echo "  configs/ 삭제됨"

echo ""
echo "== 2. core/kv_cache_hook.py 버전 주석 수정 =="
sed -i 's/Transformers 5\.10\.2/Transformers 5.0.0/g' core/kv_cache_hook.py
grep -n "Transformers 5" core/kv_cache_hook.py || true

echo ""
echo "== 3. core/collapse_metrics.py 헤더 문구 완화 =="
python3 - << 'PY'
import io,re
p='core/collapse_metrics.py'
s=io.open(p,encoding='utf-8').read()
old = """주의: 이 모듈은 원본 0808 세션 코드가 git에 커밋되지 않아 소실된 것을
2026-08-09 세션에서 정의 문서(메모리 기록)에 기반해 재구현한 것이다."""
new = """이 모듈은 위 정의에 따라 구현된 최종 판정기이며, 모든 canonical 실험에
동일하게 적용되었다. 임계값 0.3 / 0.7 은 본 연구의 운영적 컷오프이며
외부 검증을 거친 보편 기준이 아니다."""
if old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new))
    print("  헤더 수정됨")
else:
    print("  (해당 문구를 찾지 못함 — 수동 확인 필요)")
PY

echo ""
echo "== 4. 문서/설정 파일 교체 =="
echo "  README.md · requirements.txt · setup.sh · PAPER_TABLE_MAPPING.md · .gitignore · LICENSE"
echo "  → 위 6개 파일은 별도 제공본으로 저장소 루트에 복사하세요."

echo ""
echo "== 5. 공개 전 보안 점검 =="
echo "-- 현재 트리 --"
grep -rniE "hf_[A-Za-z0-9]{20,}|api[_-]?key\s*=|token\s*=\s*['\"]|password" \
     --include="*.py" --include="*.md" --include="*.sh" --include="*.json" . | head || echo "  (없음)"
echo "-- 커밋 히스토리 --"
git log -p --all 2>/dev/null | grep -niE "hf_[A-Za-z0-9]{20,}" | head || echo "  (없음)"

echo ""
echo "== 6. 다음 단계 =="
echo "  git status                      # 변경 확인"
echo "  git add -A && git commit -m 'chore: prepare repository for public release'"
echo "  git push"
echo "  git tag v9-submission && git push --tags     # 투고 직전"
echo "  # GitHub → Settings → Change visibility → Public"

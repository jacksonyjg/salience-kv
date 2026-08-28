#!/bin/bash
# CLEANUP3.sh — 작업 파일 제외 + 폐기 스크립트 이동
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== 1. 작업 파일 추적 해제 (로컬 파일은 유지) =="
git rm --cached -q REORG.sh repo_reorg.zip 2>/dev/null || true

echo "== 2. .gitignore 보강 =="
if ! grep -q "^REORG.sh$" .gitignore; then
cat >> .gitignore << 'GI'

# Repository-preparation working files (not part of the release)
APPLY.sh
REORG.sh
CLEANUP*.sh
repo_fix.zip
repo_reorg.zip
GI
fi

echo "== 3. 폐기된 efficiency v1 이동 =="
if [ -f experiments/exp_table7_efficiency.py ]; then
  git mv experiments/exp_table7_efficiency.py \
         experiments/development/superseded/exp_table7_efficiency_v1.py
  echo "  → experiments/development/superseded/exp_table7_efficiency_v1.py"
fi

echo ""
echo "== 4. 결과 =="
ls -1 experiments/ | sed 's/^/  /'
echo ""
git status --short

echo ""
echo "== 5. 다음 =="
echo "  # README.md 갱신본으로 교체"
echo "  git add -A"
echo "  git commit -m 'chore: move superseded script and exclude working files'"
echo "  git push"
echo "  rm REORG.sh repo_reorg.zip CLEANUP3.sh   # 작업 파일 로컬 삭제"

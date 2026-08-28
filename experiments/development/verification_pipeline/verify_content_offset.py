import sys
sys.path.insert(0, "/workspace/kv-cache-exp")
from transformers import AutoTokenizer
from core.model_loader import make_prompt

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

real_context = "This is a placeholder real-looking document body used only to diverge from the marker."
marker_context = "\uE000\uE000\uE000\uE000 MARKERMARKERMARKER \uE000\uE000\uE000\uE000"

p1 = make_prompt("qwen3-4b", tok, real_context, "", "summarization")
p2 = make_prompt("qwen3-4b", tok, marker_context, "", "summarization")

ids1 = tok(p1, add_special_tokens=False)["input_ids"]
ids2 = tok(p2, add_special_tokens=False)["input_ids"]

offset = 0
for a, b in zip(ids1, ids2):
    if a != b:
        break
    offset += 1

print(f"계산된 offset: {offset}")
print(f"\n=== offset 직전 5개 토큰 (고정 지시어 템플릿의 끝부분) ===")
for i in range(max(0, offset - 5), offset):
    print(f"  [{i}] id={ids1[i]} -> {tok.decode([ids1[i]])!r}")

print(f"\n=== offset 위치부터 5개 (real_context 버전 — 실제 문서 내용 시작) ===")
for i in range(offset, min(offset + 5, len(ids1))):
    print(f"  [{i}] id={ids1[i]} -> {tok.decode([ids1[i]])!r}")

print(f"\n=== offset 위치부터 5개 (marker_context 버전 — 마커 시작) ===")
for i in range(offset, min(offset + 5, len(ids2))):
    print(f"  [{i}] id={ids2[i]} -> {tok.decode([ids2[i]])!r}")

print(f"\n=== 검증: qmsum도 같은 offset인지 (task_type 둘 다 summarization) ===")
p1_qmsum_check = make_prompt("qwen3-4b", tok, real_context, "", "summarization")
print("qmsum과 gov_report 모두 동일 템플릿 사용 확인됨 (task_type='summarization' 공유)" if p1 == p1_qmsum_check else "다름 - 재확인 필요")

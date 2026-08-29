import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from transformers import AutoTokenizer
from core.model_loader import make_prompt

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

real_context = "This is a placeholder real-looking document body used only to diverge from the marker."

# gov_report 스타일(question="")
p_gov = make_prompt("qwen3-4b", tokenizer, real_context, "", "summarization")
ids_gov = tokenizer(p_gov, add_special_tokens=False)["input_ids"]

# qmsum corrected 스타일(question=실제 쿼리)
sample_question = "What did the team discuss about the design?"
p_qmsum = make_prompt("qwen3-4b", tokenizer, real_context, sample_question, "summarization")
ids_qmsum = tokenizer(p_qmsum, add_special_tokens=False)["input_ids"]

print(f"gov_report 스타일 프롬프트 앞부분: {p_gov[:150]!r}")
print(f"토큰 수(전체): {len(ids_gov)}")
print()
print(f"qmsum corrected 스타일 프롬프트 앞부분: {p_qmsum[:200]!r}")
print(f"토큰 수(전체): {len(ids_qmsum)}")
print()

# 현재 compute_content_start_offset 방식으로 계산되는 offset(question="" 고정)
marker_context = "\uE000\uE000\uE000\uE000 MARKERMARKERMARKER \uE000\uE000\uE000\uE000"
p1 = make_prompt("qwen3-4b", tokenizer, real_context, "", "summarization")
p2 = make_prompt("qwen3-4b", tokenizer, marker_context, "", "summarization")
ids1 = tokenizer(p1, add_special_tokens=False)["input_ids"]
ids2 = tokenizer(p2, add_special_tokens=False)["input_ids"]
offset_gov = 0
for a, b in zip(ids1, ids2):
    if a != b:
        break
    offset_gov += 1
print(f"현재 코드가 계산하는 offset(question='' 고정, gov_report 기준): {offset_gov}")

# qmsum 실제 offset(question 있을 때)
p1q = make_prompt("qwen3-4b", tokenizer, real_context, sample_question, "summarization")
p2q = make_prompt("qwen3-4b", tokenizer, marker_context, sample_question, "summarization")
ids1q = tokenizer(p1q, add_special_tokens=False)["input_ids"]
ids2q = tokenizer(p2q, add_special_tokens=False)["input_ids"]
offset_qmsum = 0
for a, b in zip(ids1q, ids2q):
    if a != b:
        break
    offset_qmsum += 1
print(f"실제 qmsum corrected offset(question 있을 때 기준): {offset_qmsum}")
print(f"\n차이: {offset_qmsum - offset_gov} 토큰")

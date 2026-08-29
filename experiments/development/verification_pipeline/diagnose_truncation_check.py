import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from transformers import AutoTokenizer
from core.model_loader import make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
samples = load_longbench_task("qmsum", num_samples=1, seed=42)
s = samples[0]

prompt = make_prompt(model_key="qwen3-4b", tokenizer=tokenizer, context=s["context"],
                      question=s["question"], task_type=s["task_type"])

inputs = tokenize_prompt(prompt, tokenizer, "qwen3-4b", max_input_length=2000, device="cpu")
ids = inputs["input_ids"][0].tolist()

decoded = tokenizer.decode(ids, skip_special_tokens=False)

print(f"QUERY: {s['question']!r}")
print(f"\n디코드된 truncated 프롬프트 마지막 400자:")
print(repr(decoded[-400:]))

print(f"\n쿼리 텍스트가 디코드된 결과에 있는가: {s['question'] in decoded}")

# 참고: 토큰 ID 직접 비교(원래 시도) - 실패 이유 확인용
query_ids_standalone = tokenizer(s["question"], add_special_tokens=False)["input_ids"]
print(f"\n쿼리 단독 토큰화: {query_ids_standalone[:10]}... (총 {len(query_ids_standalone)}개)")

# 프롬프트 전체(truncation 전)에서 쿼리 부분만 다시 잘라 토큰화해서 비교
full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
full_decoded = tokenizer.decode(full_ids, skip_special_tokens=False)
query_pos_in_text = full_decoded.find(s["question"])
print(f"전체 프롬프트(원본) 텍스트에서 쿼리 위치: {query_pos_in_text}")

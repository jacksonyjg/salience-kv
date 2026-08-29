import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task

_, tok, _ = load_model_and_tokenizer("qwen3-4b")

POOL_SIZE = 30
pool = load_longbench_task("gov_report", num_samples=POOL_SIZE, seed=42)
print(f"pool 크기: {len(pool)}")

s = pool[0]
prompt = make_prompt(model_key="qwen3-4b", tokenizer=tok, context=s["context"],
                      question=s["question"], task_type=s["task_type"])

# 방식 1: 사전 필터에서 쓴 방식
ids_v1 = tok(prompt, add_special_tokens=False)["input_ids"]
print(f"방식1(사전필터, list): len={len(ids_v1)}")

# 방식 2: tokenize_prompt 내부와 동일한 방식
encoded = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
ids_v2 = encoded["input_ids"]
print(f"방식2(tokenize_prompt 내부, tensor): shape={ids_v2.shape}")

# 방식 3: tokenize_prompt 직접 호출(8192 목표)
result = tokenize_prompt(prompt, tok, "qwen3-4b", max_input_length=8192, device="cpu")
print(f"방식3(tokenize_prompt 직접 호출, max=8192): shape={result['input_ids'].shape}")

print(f"\ncontext 문자 길이: {len(s['context'])}")
print(f"prompt 전체 문자 길이: {len(prompt)}")

import sys, os
sys.path.insert(0, os.getcwd())

from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task

MODEL_KEY = "gemma-2-2b"

model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
samples = load_longbench_task("qmsum", num_samples=1, seed=42)
sample = samples[0]

prompt = make_prompt(
    model_key=MODEL_KEY, tokenizer=tokenizer,
    context=sample["context"][:200] + "...(생략)...",
    question=sample.get("question", ""),
    task_type=sample["task_type"],
)

print("=== 생성된 프롬프트 (앞부분) ===")
print(repr(prompt[:500]))
print("\n=== 프롬프트 끝부분 ===")
print(repr(prompt[-300:]))

print(f"\n=== tokenizer 기본 정보 ===")
print(f"bos_token: {repr(tokenizer.bos_token)} (id={tokenizer.bos_token_id})")
print(f"eos_token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
print(f"pad_token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")

short_prompt = make_prompt(
    model_key=MODEL_KEY, tokenizer=tokenizer,
    context=sample["context"][:500],
    question=sample.get("question", ""),
    task_type=sample["task_type"],
)
inputs = tokenize_prompt(short_prompt, tokenizer, MODEL_KEY, max_input_length=7192, device=model.device)
input_ids = inputs["input_ids"]
print(f"\n=== 토큰화 결과 ===")
print(f"input_ids shape: {input_ids.shape}")
print(f"첫 5개 토큰: {input_ids[0, :5].tolist()}")
print(f"첫 5개 토큰 디코드: {[tokenizer.decode([t]) for t in input_ids[0, :5].tolist()]}")
print(f"bos_token_id가 첫 토큰인가? {input_ids[0, 0].item() == tokenizer.bos_token_id}")

print(f"\n=== 짧은 직접 generate() 테스트 (KV 압축 없이) ===")
import torch
with torch.no_grad():
    out = model.generate(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"],
        max_new_tokens=30,
        do_sample=False,
    )
gen_text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
print(f"생성 결과: {gen_text!r}")

print(f"\n=== model.generation_config ===")
print(model.generation_config)

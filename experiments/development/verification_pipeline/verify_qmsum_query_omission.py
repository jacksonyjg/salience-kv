"""
QMSum query 미사용 문제 CPU 전용 확인(GPT 20차 검토 제안).
GPU/모델 가중치 로드 불필요 - AutoTokenizer만 사용.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from transformers import AutoTokenizer
from core.model_loader import make_prompt
from core.dataset_loader import load_longbench_task

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

samples = load_longbench_task("qmsum", num_samples=3, seed=42)

for i, s in enumerate(samples):
    print(f"=== SAMPLE {i} ===")
    print(f"QUERY: {s['question'][:200]!r}")

    prompt = make_prompt(
        model_key="qwen3-4b",
        tokenizer=tokenizer,
        context=s["context"],
        question=s["question"],
        task_type=s["task_type"],
    )

    print(f"QUERY IN PROMPT: {s['question'] in prompt}")
    print(f"프롬프트 앞부분: {prompt[:300]!r}")
    print()

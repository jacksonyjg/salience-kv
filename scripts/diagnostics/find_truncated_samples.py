"""
TABLE VI에서 심각하게 잘린 두 샘플(41630, 51949 토큰)이
qmsum/gov_report 중 어느 태스크의 몇 번째 샘플인지 특정.
읽기 전용 조회 - 모델 로드 없이 토크나이저만 사용, 지금 도는 실험에 영향 없음.
"""
import sys
sys.path.insert(0, ".")

from transformers import AutoTokenizer
from core.dataset_loader import load_longbench_task
from core.model_loader import make_prompt, MODEL_CONFIGS

MODEL_KEY = "qwen3-4b"
TARGET_LENGTHS = {41630, 51949}
TOLERANCE = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[MODEL_KEY]["hf_name"])

for task in ["qmsum", "gov_report"]:
    samples = load_longbench_task(task, num_samples=30, seed=42)
    for idx, sample in enumerate(samples):
        prompt = make_prompt(
            MODEL_KEY, tokenizer,
            context=sample.get("context", sample.get("input", "")),
            question=sample.get("question", ""),
            task_type="summarization",
        )
        encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
        seq_len = len(encoded["input_ids"])

        for target in TARGET_LENGTHS:
            if abs(seq_len - target) <= TOLERANCE:
                print(f"[MATCH] task={task} idx={idx} seq_len={seq_len} (target={target})")

import sys, os
sys.path.insert(0, os.getcwd())

import torch
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from scripts.diagnostics.table10_gemma_eos_fix_check import EvaluatorV2FixedEOS

# Gemma-2-2b sliding_window = 4096. 이 경계 전후로 FullKV 정상 여부 확인.
TARGET_LENGTHS = [2000, 3000, 3800, 4000, 4200, 4500, 5000, 6000]


def make_synthetic_sample(context_text, question, max_new_tokens=48):
    return {
        "context": context_text, "question": question, "task_type": "qa",
        "answers": ["dummy"], "metric": "f1", "max_new_tokens": max_new_tokens,
        "task_name": "synthetic",
    }


def build_context_of_length(tokenizer, target_tokens):
    """반복 문장을 이어붙여 대략 target_tokens 길이의 컨텍스트를 만든다."""
    unit = (
        "The Industrial Designer presented a new remote control concept, focusing on "
        "ergonomics and a distinctive silhouette. The Project Manager reviewed the budget "
        "constraints, noting the target cost of twelve euros per unit. The Marketing "
        "representative proposed a fruit-and-vegetable theme for younger buyers. "
    )
    unit_len = len(tokenizer(unit)["input_ids"])
    repeats = max(1, target_tokens // unit_len + 1)
    text = unit * repeats
    # 토큰 수 맞춰서 자르기
    ids = tokenizer(text)["input_ids"][:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    MODEL_KEY = "gemma-2-2b"
    model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
    evaluator = EvaluatorV2FixedEOS(model, tokenizer, model_config)

    print(f"{'목표길이':>8}{'실제프롬프트길이':>14}  결과")
    print("-" * 70)

    for target_len in TARGET_LENGTHS:
        context = build_context_of_length(tokenizer, target_len)
        sample = make_synthetic_sample(context, "What did the team discuss?")

        prompt = make_prompt(
            model_key=MODEL_KEY, tokenizer=tokenizer,
            context=sample["context"], question=sample["question"],
            task_type=sample["task_type"],
        )
        n_tokens = len(tokenizer(prompt)["input_ids"])

        r = evaluator.evaluate_sample(sample, "fullkv", 0.20, measure_efficiency=False)
        pred_short = r["prediction"][:80].replace("\n", " ")
        boundary_mark = " <-- SWA 경계(4096)" if 3900 <= target_len <= 4300 else ""
        print(f"{target_len:>8}{n_tokens:>14}  {pred_short!r}{boundary_mark}")


if __name__ == "__main__":
    main()

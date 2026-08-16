import sys, os
sys.path.insert(0, os.getcwd())

import torch
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.evaluator_v2 import EvaluatorV2
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# table10_gemma_eos_fix_check.py와 동일한 EOS 수정 버전 재사용
from scripts.diagnostics.table10_gemma_eos_fix_check import EvaluatorV2FixedEOS


def make_synthetic_sample(context_text, question, task_type="qa", max_new_tokens=64):
    return {
        "context": context_text,
        "question": question,
        "task_type": task_type,
        "answers": ["dummy"],
        "metric": "rouge",
        "max_new_tokens": max_new_tokens,
        "task_name": "synthetic",
    }


def main():
    MODEL_KEY = "gemma-2-2b"
    model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
    print(f"model.generation_config.eos_token_id = {model.generation_config.eos_token_id}")

    evaluator = EvaluatorV2FixedEOS(model, tokenizer, model_config)

    # 매우 짧은 컨텍스트 (SWA 윈도우보다 확실히 짧음, 대략 200~400토큰 예상)
    short_context = (
        "The Industrial Designer presented a new remote control concept. "
        "The team discussed using a scroll wheel for channel navigation and "
        "a rubberized grip for comfort. The Project Manager asked about cost "
        "constraints, noting the budget was fixed at twelve euros per unit. "
        "The Marketing representative suggested a fruit-and-vegetable theme "
        "to appeal to younger buyers, which the team found unconventional but interesting."
    ) * 3  # 반복해서 조금 더 길게, 그래도 짧음

    sample = make_synthetic_sample(
        short_context,
        "What did the team discuss?",
        task_type="qa",
        max_new_tokens=64,
    )

    prompt = make_prompt(
        model_key=MODEL_KEY, tokenizer=tokenizer,
        context=sample["context"], question=sample["question"],
        task_type=sample["task_type"],
    )
    n_tokens = len(tokenizer(prompt)["input_ids"])
    print(f"\n프롬프트 길이: {n_tokens} 토큰")

    print("\n--- FullKV (짧은 시퀀스, EOS 수정판) ---")
    r = evaluator.evaluate_sample(sample, "fullkv", 0.20, measure_efficiency=False)
    print(f"score={r['score']:.2f} pred={r['prediction'][:200]!r}")


if __name__ == "__main__":
    main()

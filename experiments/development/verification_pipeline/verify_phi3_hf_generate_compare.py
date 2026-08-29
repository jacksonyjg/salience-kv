"""
FullKV gov_report idx=17에서, EvaluatorV2의 커스텀 step-by-step 디코딩 루프와
HuggingFace 표준 model.generate()가 동일한 결과를 내는지 직접 대조.
Case A: HF generate()도 똑같이 깨짐 -> Phi-3 자체의 특성, evaluator 버그 아님
Case B: HF generate()는 정상 -> EvaluatorV2의 커스텀 루프에 문제 있음
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed, word_repetition_ratio, char_repetition_ratio
import torch

model, tok, cfg = load_model_and_tokenizer("phi-3-mini")

samples = load_longbench_task("gov_report", num_samples=30, seed=42)
sample = samples[17]

print("=== 1) EvaluatorV2 커스텀 디코딩 루프 (기존 방식, 재확인) ===")
ev = EvaluatorV2(model, tok, cfg)
r_custom = ev.evaluate_sample(sample, "fullkv", 0.20, method_kwargs={})
pred_custom = r_custom["prediction"]
print(f"score={r_custom['score']:.2f}  word_rep={word_repetition_ratio(pred_custom):.4f}  "
      f"collapsed={is_collapsed(pred_custom)}")
print(f"뒷부분(마지막 300자): ...{pred_custom[-300:]!r}")

print("\n\n=== 2) HuggingFace 표준 model.generate() (동일 prompt, 동일 조건) ===")
prompt = make_prompt(model_key="phi-3-mini", tokenizer=tok, context=sample["context"],
                      question=sample["question"], task_type=sample["task_type"])
model_max_len = cfg.get("max_length", 16000)
safe_max_input = min(16000, model_max_len - 1000)
inputs = tokenize_prompt(prompt, tok, "phi-3-mini", max_input_length=safe_max_input, device=ev.device)

raw_eos = model.generation_config.eos_token_id
eos_ids = raw_eos if isinstance(raw_eos, (list, tuple)) else [raw_eos]
print(f"generate()에 전달할 eos_token_id: {eos_ids}")

with torch.no_grad():
    gen_out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=sample["max_new_tokens"],
        do_sample=False,  # EvaluatorV2와 동일하게 greedy
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
    )
new_tokens = gen_out[0][inputs["input_ids"].shape[1]:]
pred_hf = tok.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(f"score 계산 생략(비교 목적) | word_rep={word_repetition_ratio(pred_hf):.4f}  "
      f"collapsed={is_collapsed(pred_hf)}  num_new_tokens={len(new_tokens)}")
print(f"뒷부분(마지막 300자): ...{pred_hf[-300:]!r}")

print("\n\n=== 3) 두 출력이 동일한가? ===")
print(f"완전 동일: {pred_custom == pred_hf}")
if pred_custom != pred_hf:
    # 앞에서부터 몇 글자까지 일치하는지
    min_len = min(len(pred_custom), len(pred_hf))
    diverge_at = min_len
    for i in range(min_len):
        if pred_custom[i] != pred_hf[i]:
            diverge_at = i
            break
    print(f"몇 번째 글자부터 갈리는지: {diverge_at} (전체 길이: custom={len(pred_custom)}, hf={len(pred_hf)})")

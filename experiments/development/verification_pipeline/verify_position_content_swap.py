import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import torch
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.kv_cache_hook import _key_importance

model, tok, cfg = load_model_and_tokenizer("phi-3-mini")
device = next(model.parameters()).device

samples = load_longbench_task("gov_report", num_samples=5, seed=42)

print("=== 위치 1↔2 swap 실험 (N=5, gov_report) ===")
print("원래: [0]=<|user|> [1]=Please [2]=summar ...")
print("swap 후: [0]=<|user|> [1]=summar [2]=Please ...\n")

for i, sample in enumerate(samples):
    prompt = make_prompt(model_key="phi-3-mini", tokenizer=tok, context=sample["context"],
                          question=sample["question"], task_type=sample["task_type"])
    model_max_len = cfg.get("max_length", 16000)
    safe_max_input = min(16000, model_max_len - 1000)
    inputs = tokenize_prompt(prompt, tok, "phi-3-mini", max_input_length=safe_max_input, device=device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    tok1_orig = tok.decode([input_ids[0, 1].item()])
    tok2_orig = tok.decode([input_ids[0, 2].item()])

    # 원본으로 prefill (전체 레이어 키 확보)
    with torch.no_grad():
        out_orig = model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=True, return_dict=True, output_attentions=False)

    # 위치 1,2 스왑
    swapped_ids = input_ids.clone()
    swapped_ids[0, 1], swapped_ids[0, 2] = input_ids[0, 2].item(), input_ids[0, 1].item()
    tok1_swap = tok.decode([swapped_ids[0, 1].item()])
    tok2_swap = tok.decode([swapped_ids[0, 2].item()])

    with torch.no_grad():
        out_swap = model(input_ids=swapped_ids, attention_mask=attention_mask,
                          use_cache=True, return_dict=True, output_attentions=False)

    num_layers = len(out_orig.past_key_values.layers)
    please_follows_count = 0
    print(f"--- 샘플 {i} (레이어 수: {num_layers}) ---")
    print(f"  {'Layer':>6s} {'원본승자':>10s} {'swap승자':>10s} {'Please를 따라감':>16s}")
    for layer_idx in range(num_layers):
        k_orig = out_orig.past_key_values.layers[layer_idx].keys
        score_orig = -_key_importance(k_orig)
        k_swap = out_swap.past_key_values.layers[layer_idx].keys
        score_swap = -_key_importance(k_swap)

        orig_winner_pos = 1 if score_orig[1] > score_orig[2] else 2
        swap_winner_pos = 1 if score_swap[1] > score_swap[2] else 2
        # "Please"가 원본에서 위치1, swap에서 위치2에 있음
        orig_winner_is_please = (orig_winner_pos == 1)
        swap_winner_is_please = (swap_winner_pos == 2)
        follows = orig_winner_is_please and swap_winner_is_please
        if follows:
            please_follows_count += 1
        print(f"  {layer_idx:6d} {'pos'+str(orig_winner_pos):>10s} {'pos'+str(swap_winner_pos):>10s} "
              f"{'YES(content)' if follows else 'no(position?)':>16s}")
    print(f"  => {please_follows_count}/{num_layers} 레이어에서 'Please'를 따라 승자 이동\n")

print("판정 기준: swap 후에도 'Please' 위치를 따라 승자가 이동하면 -> content 효과")
print("           swap 후에도 여전히 position 1이 이기면 -> position 효과")

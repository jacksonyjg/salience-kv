"""
현재 evaluator_v2.py 방식(버그 있는 재forward) vs 올바른 방식(원본 위치 이어가는
수동 그리디 디코딩)을 소규모(3방법×5샘플)로 비교.
"""
import sys
sys.path.insert(0, '.')
import torch
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.kv_cache_hook import make_hook_cache
from core.metrics import compute_score
from core.evaluator_v2 import EvaluatorV2

model, tokenizer, model_config = load_model_and_tokenizer('qwen3-4b')
evaluator = EvaluatorV2(model, tokenizer, model_config)
samples = load_longbench_task('qmsum', num_samples=5, seed=42)

METHODS = ['h2o', 'adakv', 'ours']
BUDGET = 0.2


def correct_generate(sample):
    prompt = make_prompt('qwen3-4b', tokenizer, sample['context'], sample['question'], sample['task_type'])
    inputs = tokenize_prompt(prompt, tokenizer, 'qwen3-4b', max_input_length=16000, device=model.device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    N = input_ids.shape[1]

    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        prefill_kv = prefill_out.past_key_values
        last_logits = prefill_out.logits[:, -1, :]

    results = {}
    for method in METHODS:
        cache = make_hook_cache(method, budget_ratio=BUDGET, model_config=model_config)
        for i, layer in enumerate(prefill_kv.layers):
            if i >= len(cache.layers):
                cache.update(layer.keys, layer.values, i)
            else:
                cache.layers[i].keys = layer.keys.clone()
                cache.layers[i].values = layer.values.clone()
            cache.set_prefill_keys(i, layer.keys)
        cache.mark_prefill_done(N)
        cache.apply_compression_all_layers()
        cache_len = cache.layers[0].keys.shape[2]

        next_token = last_logits.argmax(dim=-1, keepdim=True)
        generated = [next_token.item()]

        max_new = sample['max_new_tokens']
        eos_id = tokenizer.eos_token_id
        cur_pos = N

        for step in range(max_new - 1):
            if generated[-1] == eos_id:
                break
            cache_position = torch.tensor([cur_pos], device=model.device)
            attn_mask = torch.ones(1, cache_len + step + 1, dtype=attention_mask.dtype, device=model.device)
            with torch.no_grad():
                out = model(
                    input_ids=next_token,
                    past_key_values=cache,
                    cache_position=cache_position,
                    attention_mask=attn_mask,
                    use_cache=True,
                    return_dict=True,
                )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
            cur_pos += 1

        pred_ids = torch.tensor(generated, device=model.device)
        prediction = tokenizer.decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prediction = evaluator._clean_prediction(prediction)
        score = compute_score(prediction, sample['answers'], sample['metric'])
        results[method] = {'score': score, 'prediction': prediction}

    return results


print("===== 비교 시작 =====\n")
for idx, sample in enumerate(samples):
    print(f"########## 샘플 {idx} ##########")
    correct = correct_generate(sample)
    for method in METHODS:
        buggy = evaluator.evaluate_sample(sample, method, budget_ratio=BUDGET, method_kwargs=None)
        c = correct[method]
        print(f"--- {method} ---")
        print(f"  [현재 방식]  score={buggy['score']:.2f}  pred={buggy['prediction'][:100]!r}")
        print(f"  [올바른 방식] score={c['score']:.2f}  pred={c['prediction'][:100]!r}")
    print()

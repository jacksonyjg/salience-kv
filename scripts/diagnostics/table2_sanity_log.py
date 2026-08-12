"""
TABLE II (Evaluation Pipeline Sanity Check) 데이터 수집 (수정본, 2026-08-12)
"""
import sys
sys.path.insert(0, '.')
import torch
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
import core.evaluator_v2 as ev2_module

MODEL_KEY = 'qwen3-4b'
METHODS = ['fullkv', 'ours', 'adakv', 'streaming']
BUDGET = 0.2

model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
evaluator = EvaluatorV2(model, tokenizer, model_config)
sample = load_longbench_task('qmsum', num_samples=1, seed=42)[0]

prompt = make_prompt(MODEL_KEY, tokenizer, sample['context'], sample['question'], sample['task_type'])
inputs = tokenize_prompt(prompt, tokenizer, MODEL_KEY, max_input_length=16000, device=model.device)
input_length = inputs['input_ids'].shape[1]

print(f"{'항목':28s} " + " ".join(f"{m:>14s}" for m in METHODS))

rows = {k: [] for k in [
    'input_length', 'cache_len_before', 'cache_len_after',
    'first_decode_input_len', 'attention_mask_len_step1', 'next_position',
    'selected_pos_min', 'selected_pos_max', 'prompt_reforward'
]}

orig_make_hook_cache = ev2_module.make_hook_cache

for method in METHODS:
    call_log = []
    captured_cache = {}

    def hooked_make_hook_cache(*a, **kw):
        hc = orig_make_hook_cache(*a, **kw)
        captured_cache['obj'] = hc
        return hc

    orig_forward = model.forward

    def hooked_forward(*a, **kw):
        ii = kw.get('input_ids', None)
        n = ii.shape[1] if ii is not None else -1
        am = kw.get('attention_mask', None)
        cp = kw.get('cache_position', None)
        call_log.append({
            'input_len': n,
            'attn_mask_len': am.shape[1] if am is not None else None,
            'cache_pos': cp.clone().cpu() if cp is not None else None,
        })
        return orig_forward(*a, **kw)

    ev2_module.make_hook_cache = hooked_make_hook_cache
    model.forward = hooked_forward
    kwargs = {} if method == 'fullkv' else {'sink_size': 4}
    try:
        r = evaluator.evaluate_sample(sample, method, budget_ratio=BUDGET, method_kwargs=kwargs)
    finally:
        model.forward = orig_forward
        ev2_module.make_hook_cache = orig_make_hook_cache

    cache_len_before = input_length
    step2 = call_log[1] if len(call_log) > 1 else call_log[0]
    first_decode_input_len = step2['input_len']
    attn_mask_len_step1 = step2['attn_mask_len']
    next_position = step2['cache_pos'].item() if step2['cache_pos'] is not None else None

    if method == 'fullkv':
        cache_len_after = input_length
        pos_min, pos_max = '-', '-'
    else:
        hc = captured_cache.get('obj')
        # 압축 직후 값은 첫 디코딩 스텝의 attention_mask 길이에서 역산
        cache_len_after = attn_mask_len_step1 - 1
        sel = getattr(hc, '_selected_positions', [None])[0] if hc is not None else None
        if sel is not None:
            pos_min, pos_max = int(sel.min().item()), int(sel.max().item())
        else:
            pos_min, pos_max = '-', '-'

    reforward = any(c['input_len'] > 10 for c in call_log[1:])

    rows['input_length'].append(input_length)
    rows['cache_len_before'].append(cache_len_before)
    rows['cache_len_after'].append(cache_len_after)
    rows['first_decode_input_len'].append(first_decode_input_len)
    rows['attention_mask_len_step1'].append(attn_mask_len_step1)
    rows['next_position'].append(next_position)
    rows['selected_pos_min'].append(pos_min)
    rows['selected_pos_max'].append(pos_max)
    rows['prompt_reforward'].append('Yes' if reforward else 'No')

for key, vals in rows.items():
    print(f"{key:28s} " + " ".join(f"{str(v):>14s}" for v in vals))

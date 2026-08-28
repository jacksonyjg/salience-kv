import sys, json
sys.path.insert(0, "/workspace/kv-cache-exp")
import torch
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
import core.kv_cache_hook as kvh
import core.evaluator_v2 as ev2

# 1) _key_importance 후킹 — layer 0의 raw norm array 캡처 (invert 적용 "전" 원본 norm)
_orig_key_importance = kvh._key_importance
captured_norms = {}
def patched_key_importance(key_states):
    result = _orig_key_importance(key_states)
    if "layer0" not in captured_norms:
        captured_norms["layer0"] = result.detach().cpu().clone()
    return result
kvh._key_importance = patched_key_importance

# 2) 캐시 인스턴스 캡처 (selected_positions 얻기 위해)
_orig_factory = kvh.make_hook_cache
captured = {}
def patched_factory(*args, **kwargs):
    cache = _orig_factory(*args, **kwargs)
    captured["cache"] = cache
    return cache
kvh.make_hook_cache = patched_factory
if hasattr(ev2, "make_hook_cache"):
    ev2.make_hook_cache = patched_factory

model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tok, cfg)

N_ONLY = dict(use_attention=True, use_entropy=False, use_position=False, use_semantic=False, sink_size=0)

for task in ["gov_report", "qmsum"]:
    data = load_longbench_task(task, num_samples=1, seed=42)
    sample = data[0]
    print(f"\n{'='*70}\nTASK: {task}\n{'='*70}")
    for inv in [False, True]:
        captured.clear()
        captured_norms.clear()
        kwargs = dict(N_ONLY, invert_norm=inv)
        r = ev.evaluate_sample(sample, "ours", 0.20, method_kwargs=kwargs)

        cache = captured.get("cache")
        summary = "캡처 실패"
        if cache is not None and cache._selected_positions[0] is not None and "layer0" in captured_norms:
            sel = cache._selected_positions[0]
            norms = captured_norms["layer0"]
            seq_len = norms.shape[0]
            sel_idx = list(set(sel.tolist()))
            dropped_idx = list(set(range(seq_len)) - set(sel_idx))
            sel_mean = norms[sel_idx].mean().item() if sel_idx else float('nan')
            dropped_mean = norms[dropped_idx].mean().item() if dropped_idx else float('nan')
            overall_mean = norms.mean().item()
            summary = (f"선택={len(sel_idx)}/{seq_len} | "
                       f"선택된 토큰 평균norm={sel_mean:.3f} | "
                       f"버려진 토큰 평균norm={dropped_mean:.3f} | "
                       f"전체평균={overall_mean:.3f}")
            verdict = "선택<버려진 (LOW-norm 우선, Devoto 방향 O)" if sel_mean < dropped_mean else "선택>버려진 (HIGH-norm 우선)"
            summary += f"\n  판정: {verdict}"

        print(f"invert_norm={inv!s:5s} -> score={r['score']:6.2f}")
        print(f"  {summary}")
        print(f"  pred[:150]={r['prediction'][:150]!r}")

kvh._key_importance = _orig_key_importance
kvh.make_hook_cache = _orig_factory

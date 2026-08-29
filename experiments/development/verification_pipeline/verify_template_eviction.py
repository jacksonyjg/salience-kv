import sys, json, importlib.util
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

spec = importlib.util.spec_from_file_location(
    "table13_mod", "/workspace/kv-cache-exp/scripts/diagnostics/table13_position_content_validation.py"
)
t13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t13)

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed
import core.kv_cache_hook as kvh
import core.evaluator_v2 as ev2

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

offset = t13.compute_content_start_offset(tok, "qwen3-4b", task_type="summarization")
print(f"content_offset = {offset}")

kvh._select_with_sink = t13.make_patched_select_with_sink("front", offset=offset)

samples = load_longbench_task("gov_report", num_samples=30, seed=42)
TEMPLATE_END = 14

rows = []
for i, sample in enumerate(samples):
    captured.clear()
    r = ev.evaluate_sample(sample, "ours", 0.20, method_kwargs=dict(sink_size=4, invert_norm=True))
    cache = captured.get("cache")
    sel = set(cache._selected_positions[0].tolist()) if (cache and cache._selected_positions[0] is not None) else set()
    template_kept = len(sel & set(range(TEMPLATE_END)))
    front4_kept = len(sel & {0, 1, 2, 3})
    collapsed = is_collapsed(r["prediction"])
    rows.append({"idx": i, "collapsed": collapsed, "template_kept": template_kept, "front4_kept": front4_kept})
    print(f"idx={i:2d}  collapsed={collapsed!s:5s}  템플릿(0~13) 생존={template_kept}/14  앞4개(0~3) 생존={front4_kept}/4")

kvh._select_with_sink = t13._ORIGINAL_SELECT_WITH_SINK

collapsed_rows = [r for r in rows if r["collapsed"]]
passed_rows = [r for r in rows if not r["collapsed"]]
print(f"\n=== collapsed({len(collapsed_rows)}개) vs non-collapsed({len(passed_rows)}개) 비교 ===")
if collapsed_rows:
    print(f"collapsed 평균 템플릿(0~13) 생존: {sum(r['template_kept'] for r in collapsed_rows)/len(collapsed_rows):.2f}/14")
    print(f"collapsed 평균 앞4개(0~3) 생존: {sum(r['front4_kept'] for r in collapsed_rows)/len(collapsed_rows):.2f}/4")
if passed_rows:
    print(f"non-collapsed 평균 템플릿(0~13) 생존: {sum(r['template_kept'] for r in passed_rows)/len(passed_rows):.2f}/14")
    print(f"non-collapsed 평균 앞4개(0~3) 생존: {sum(r['front4_kept'] for r in passed_rows)/len(passed_rows):.2f}/4")

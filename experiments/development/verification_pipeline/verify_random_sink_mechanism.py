import sys, importlib.util
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

spec = importlib.util.spec_from_file_location(
    "table13_mod", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "scripts", "diagnostics",
        "table13_position_content_validation.py")
)
t13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t13)

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed
import core.kv_cache_hook as kvh
import core.evaluator_v2 as ev2
import torch

_orig_factory = kvh.make_hook_cache
captured = {}
def patched_factory(*args, **kwargs):
    cache = _orig_factory(*args, **kwargs)
    captured["cache"] = cache
    return cache
kvh.make_hook_cache = patched_factory
if hasattr(ev2, "make_hook_cache"):
    ev2.make_hook_cache = patched_factory

torch.manual_seed(t13.SEED)
model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tok, cfg)

# random_real이 실제로 썼던 그 함수 그대로 재사용 (offset=0, 절대위치)
kvh._select_with_sink = t13.make_patched_select_with_sink("random", offset=0)

rows = []
for task in ["qmsum", "gov_report"]:
    samples = load_longbench_task(task, num_samples=30, seed=t13.SEED)
    for i, sample in enumerate(samples):
        captured.clear()
        r = ev.evaluate_sample(sample, "ours", 0.20, method_kwargs=dict(sink_size=4, invert_norm=True))
        cache = captured.get("cache")
        sel = set(cache._selected_positions[0].tolist()) if (cache and cache._selected_positions[0] is not None) else set()
        front4_kept = len(sel & {0, 1, 2, 3})
        collapsed = is_collapsed(r["prediction"])
        rows.append({"task": task, "idx": i, "collapsed": collapsed, "front4_kept": front4_kept})

kvh._select_with_sink = t13._ORIGINAL_SELECT_WITH_SINK

n = len(rows)
zero_front = sum(1 for r in rows if r["front4_kept"] == 0)
print(f"=== random_real, 60개 전체: 위치 0~3 중 몇 개나 우연히 포함됐는지 ===")
print(f"0~3 중 0개 포함(전혀 안 지켜짐): {zero_front}/{n} ({100*zero_front/n:.1f}%)")

collapsed_rows = [r for r in rows if r["collapsed"]]
passed_rows = [r for r in rows if not r["collapsed"]]
print(f"\ncollapsed({len(collapsed_rows)}개) 평균 0~3 생존: "
      f"{sum(r['front4_kept'] for r in collapsed_rows)/max(len(collapsed_rows),1):.2f}/4")
print(f"non-collapsed({len(passed_rows)}개) 평균 0~3 생존: "
      f"{sum(r['front4_kept'] for r in passed_rows)/max(len(passed_rows),1):.2f}/4")
print(f"\n이번 재실행 자체의 collapse rate: {100*len(collapsed_rows)/n:.1f}% (원본 N=30 결과와 유사하면 재현성 확인)")

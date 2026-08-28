import sys, json
sys.path.insert(0, "/workspace/kv-cache-exp")
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

TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "qmsum"]
FRONT_RANGE = list(range(10))  # 위치 0~9 개별 포함률 확인

rows = []
for task in TASKS:
    samples = load_longbench_task(task, num_samples=30, seed=42)
    for i, sample in enumerate(samples):
        captured.clear()
        r = ev.evaluate_sample(sample, "snapkv", 0.20, method_kwargs=dict(sink_size=0, invert_norm=True))
        cache = captured.get("cache")
        sel = None
        if cache is not None and cache._selected_positions[0] is not None:
            sel = set(cache._selected_positions[0].tolist())
        front_inclusion = {p: (p in sel if sel is not None else None) for p in FRONT_RANGE}
        collapsed = is_collapsed(r["prediction"])
        rows.append({"task": task, "idx": i, "front_inclusion": front_inclusion, "collapsed": collapsed})
        if (i + 1) % 10 == 0:
            print(f"[{task}] {i+1}/30 완료", flush=True)

json.dump(rows, open("/tmp/snapkv_front_position_check.json", "w"))

n = len(rows)
print(f"\n=== 전체 {n}개 샘플, 위치별 포함률 (SnapKV, sink_size=0, corrected) ===")
for p in FRONT_RANGE:
    included = [r["front_inclusion"][p] for r in rows if r["front_inclusion"][p] is not None]
    rate = sum(included) / len(included) * 100 if included else float("nan")
    print(f"  위치 {p:2d}: {rate:5.1f}% 샘플에서 포함 ({sum(included)}/{len(included)})")

collapsed_rows = [r for r in rows if r["collapsed"]]
passed_rows = [r for r in rows if not r["collapsed"]]
print(f"\ncollapsed({len(collapsed_rows)}개) vs non-collapsed({len(passed_rows)}개) 위치별 포함률 비교:")
for p in FRONT_RANGE:
    c_inc = [r["front_inclusion"][p] for r in collapsed_rows if r["front_inclusion"][p] is not None]
    n_inc = [r["front_inclusion"][p] for r in passed_rows if r["front_inclusion"][p] is not None]
    c_rate = sum(c_inc) / len(c_inc) * 100 if c_inc else float("nan")
    n_rate = sum(n_inc) / len(n_inc) * 100 if n_inc else float("nan")
    print(f"  위치 {p:2d}: collapsed={c_rate:5.1f}%  non-collapsed={n_rate:5.1f}%")

print(f"\ncollapsed 샘플 목록:")
for r in collapsed_rows:
    print(f"  task={r['task']} idx={r['idx']}")

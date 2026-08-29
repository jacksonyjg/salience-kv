"""
H2O(sink=0)가 Phi-3에서 위치 0~3을 암묵적으로 이미 보존하는지 확인
(GPT Case C 가설: implicit anchoring이 sink intervention의 대조 효과를 지웠을 가능성)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
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

model, tok, cfg = load_model_and_tokenizer("phi-3-mini")
ev = EvaluatorV2(model, tok, cfg)

samples = load_longbench_task("gov_report", num_samples=30, seed=42)

front4_counts = []
per_position = {0: 0, 1: 0, 2: 0, 3: 0}
for i, sample in enumerate(samples):
    captured.clear()
    r = ev.evaluate_sample(sample, "h2o", 0.20, method_kwargs=dict(sink_size=0, invert_norm=True))
    cache = captured.get("cache")
    sel = set(cache._selected_positions[0].tolist()) if (cache and cache._selected_positions[0] is not None) else set()
    front4 = len(sel & {0, 1, 2, 3})
    front4_counts.append(front4)
    for p in [0, 1, 2, 3]:
        if p in sel:
            per_position[p] += 1

n = len(front4_counts)
avg = sum(front4_counts) / n
all4 = sum(1 for c in front4_counts if c == 4)
none = sum(1 for c in front4_counts if c == 0)
print(f"=== H2O(sink=0), Phi-3, gov_report 30개: 위치 0~3 자연 보존 개수 ===")
print(f"평균: {avg:.2f}/4")
print(f"4개 다 보존: {all4}/{n}개 샘플")
print(f"0개 보존(전혀 안 됨): {none}/{n}개 샘플")
print(f"분포: {sorted(front4_counts)}")
print(f"\n=== 위치별 개별 보존율 (어느 위치가 30/30인지) ===")
for p in [0, 1, 2, 3]:
    print(f"  위치 {p}: {per_position[p]}/{n}")

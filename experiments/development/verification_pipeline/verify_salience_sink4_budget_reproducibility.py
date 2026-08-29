import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed

model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tok, cfg)

samples = load_longbench_task("gov_report", num_samples=30, seed=42)
sample = samples[9]

BUDGETS = [0.10, 0.20, 0.40, 0.60, 0.80]
print("=== SalienceKV-Sink-4, gov_report idx=9, 전체 budget 스윕 ===")
for b in BUDGETS:
    r = ev.evaluate_sample(sample, "ours", b, method_kwargs=dict(sink_size=4, invert_norm=True))
    c = is_collapsed(r["prediction"])
    print(f"  budget={int(b*100):3d}%  score={r['score']:6.2f}  collapsed={c}")
    if c:
        print(f"    원문 앞부분: {r['prediction'][:150]!r}")

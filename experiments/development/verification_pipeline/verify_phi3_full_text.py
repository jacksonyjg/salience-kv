import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed, word_repetition_ratio, char_repetition_ratio

MODEL_KEY = "phi-3-mini"
model, tok, cfg = load_model_and_tokenizer(MODEL_KEY)
ev = EvaluatorV2(model, tok, cfg)

samples = load_longbench_task("qmsum", num_samples=2, seed=42)

targets = [
    ("fullkv", {}),
    ("adakv", {"invert_norm": True}),
    ("ours",  {"sink_size": 4, "invert_norm": True}),
]

for method, kwargs in targets:
    print(f"\n{'='*70}\nmethod={method} kwargs={kwargs}\n{'='*70}")
    for i, sample in enumerate(samples):
        r = ev.evaluate_sample(sample, method, 0.20, method_kwargs=kwargs)
        pred = r["prediction"]
        wr = word_repetition_ratio(pred)
        cr = char_repetition_ratio(pred)
        c = is_collapsed(pred)
        print(f"\n[{i}] score={r['score']:.2f}  word_rep={wr:.3f}  char_rep={cr:.3f}  collapsed={c}  len={len(pred)}")
        print(f"전체 원문:\n{pred!r}")

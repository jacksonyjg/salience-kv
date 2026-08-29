import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed

MODEL_KEY = "phi-3-mini"  # 확인 완료(2026-08-22): core.model_loader.MODEL_CONFIGS의 실제 키

model, tok, cfg = load_model_and_tokenizer(MODEL_KEY)
ev = EvaluatorV2(model, tok, cfg)

samples = load_longbench_task("qmsum", num_samples=2, seed=42)

targets = [
    ("fullkv", {}),
    ("adakv", {"invert_norm": False}),   # legacy 방향 - 예전 "완전 붕괴" 재현 확인용
    ("adakv", {"invert_norm": True}),    # corrected 방향 - 이게 핵심 질문
    ("ours",  {"sink_size": 4, "invert_norm": True}),  # SalienceKV-Sink4 corrected
]

print("=== Phi-3-mini corrected sanity check (N=2, qmsum, budget=20%) ===")
for method, kwargs in targets:
    print(f"\n--- method={method} kwargs={kwargs} ---")
    for i, sample in enumerate(samples):
        try:
            r = ev.evaluate_sample(sample, method, 0.20, method_kwargs=kwargs)
            c = is_collapsed(r["prediction"])
            print(f"  [{i}] score={r['score']:6.2f}  collapsed={c}")
            print(f"      원문 앞부분: {r['prediction'][:120]!r}")
        except Exception as e:
            print(f"  [{i}] 예외 발생: {type(e).__name__}: {e}")

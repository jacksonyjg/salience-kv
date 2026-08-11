"""
Phase 1 초소규모 점검
목적: 1) 파이프라인 수정 후 점수가 정상 범위인지  2) sink4 방향성 유지되는지
"""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.WARNING)

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed, collapse_report

MODEL_KEY = 'qwen3-4b'
TASKS = ['qmsum', 'gov_report']
NUM_SAMPLES = 8
BUDGET = 0.2
CONFIGS = [
    ('fullkv', None),
    ('h2o', 0), ('h2o', 4),
    ('adakv', 0), ('adakv', 4),
    ('ours', 0), ('ours', 4),
]

print("모델 로딩 중...")
model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
evaluator = EvaluatorV2(model, tokenizer, model_config)

rows = []

for task in TASKS:
    samples = load_longbench_task(task, num_samples=NUM_SAMPLES, seed=42)
    print(f"\n{'='*70}\n태스크: {task} ({len(samples)}샘플)\n{'='*70}")

    for method, sink in CONFIGS:
        kwargs = {} if sink is None else {'sink_size': sink}
        scores, collapses = [], []
        for i, sample in enumerate(samples):
            try:
                r = evaluator.evaluate_sample(sample, method, budget_ratio=BUDGET, method_kwargs=kwargs)
            except Exception as e:
                print(f"  [{method} sink={sink}] 샘플{i} 예외 발생: {e}")
                scores.append(0.0)
                collapses.append(True)
                continue
            scores.append(r['score'])
            collapsed = is_collapsed(r['prediction'])
            collapses.append(collapsed)
            if i == 0:
                tag = f"{method}" + (f"+sink{sink}" if sink else "")
                print(f"  [{tag:16s}] score={r['score']:6.2f}  mem_red={r['memory_reduction_pct']:5.1f}%  "
                      f"collapsed={collapsed}  pred={r['prediction'][:80]!r}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        collapse_rate = sum(collapses) / len(collapses) if collapses else 0.0
        rows.append((task, method, sink, avg_score, collapse_rate))

print(f"\n\n{'='*70}\n요약 (평균 점수 / 붕괴율)\n{'='*70}")
print(f"{'태스크':10s} {'방법':10s} {'sink':6s} {'평균점수':>8s} {'붕괴율':>8s}")
for task, method, sink, avg_score, collapse_rate in rows:
    sink_str = '-' if sink is None else str(sink)
    print(f"{task:10s} {method:10s} {sink_str:6s} {avg_score:8.2f} {collapse_rate*100:7.1f}%")

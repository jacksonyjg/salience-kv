"""
Phase 0 최종 게이트: 7개 방법 전체 x 2태스크(QA+요약) 스모크 테스트
"""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.WARNING)

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed

MODEL_KEY = 'qwen3-4b'
TASKS = ['qasper', 'gov_report']
NUM_SAMPLES = 3
BUDGET = 0.2
METHODS = ['fullkv', 'streaming', 'h2o', 'snapkv', 'pyramidkv', 'adakv', 'ours']

print("모델 로딩 중...")
model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
evaluator = EvaluatorV2(model, tokenizer, model_config)

errors = []
rows = []

for task in TASKS:
    samples = load_longbench_task(task, num_samples=NUM_SAMPLES, seed=42)
    print(f"\n{'='*70}\n태스크: {task} ({len(samples)}샘플)\n{'='*70}")

    for method in METHODS:
        kwargs = {} if method == 'fullkv' else {'sink_size': 4}
        scores, collapses = [], []
        for i, sample in enumerate(samples):
            try:
                r = evaluator.evaluate_sample(sample, method, budget_ratio=BUDGET, method_kwargs=kwargs)
            except Exception as e:
                print(f"  [{method}] 샘플{i} 예외 발생: {type(e).__name__}: {e}")
                errors.append((task, method, i, str(e)))
                scores.append(0.0)
                collapses.append(True)
                continue
            scores.append(r['score'])
            collapsed = is_collapsed(r['prediction'])
            collapses.append(collapsed)
            if i == 0:
                tag = f"{method}"
                print(f"  [{tag:12s}] score={r['score']:6.2f}  collapsed={collapsed}  "
                      f"pred={r['prediction'][:70]!r}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        collapse_rate = sum(collapses) / len(collapses) if collapses else 0.0
        rows.append((task, method, avg_score, collapse_rate))

print(f"\n\n{'='*70}\n요약\n{'='*70}")
print(f"{'태스크':10s} {'방법':10s} {'평균점수':>8s} {'붕괴율':>8s}")
for task, method, avg_score, collapse_rate in rows:
    print(f"{task:10s} {method:10s} {avg_score:8.2f} {collapse_rate*100:7.1f}%")

print(f"\n{'='*70}")
if errors:
    print(f"⚠️  예외 발생 {len(errors)}건:")
    for task, method, i, msg in errors:
        print(f"  - {task}/{method}/샘플{i}: {msg}")
else:
    print("✅ 예외 없음 - 7개 방법 전체가 새 파이프라인에서 에러 없이 동작함")
print(f"{'='*70}")

"""
Exp1-A 확인용 재실행 (논문 미반영, 순수 확인 목적)
- GovReport + QMSum, 7개 방법, 15샘플씩, budget=20%
"""
import sys, csv, time, datetime
sys.path.insert(0, '.')
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2

model, tokenizer, model_config = load_model_and_tokenizer('qwen3-4b')
evaluator = EvaluatorV2(model, tokenizer, model_config)

tasks = ['gov_report', 'qmsum']
methods = ['fullkv', 'pyramidkv', 'adakv', 'streaming', 'ours', 'h2o', 'snapkv']

rows = []
t0 = time.time()
run_count = 0
total_runs = len(tasks) * len(methods) * 15

for task in tasks:
    samples = load_longbench_task(task, num_samples=15, seed=42)
    for method in methods:
        for i, s in enumerate(samples):
            r = evaluator.evaluate_sample(s, method, budget_ratio=0.2, method_kwargs=None)
            rows.append({'task': task, 'method': method, 'sample_idx': i, 'score': r['score']})
            run_count += 1
            if run_count % 15 == 0:
                elapsed = time.time() - t0
                print(f"  {run_count}/{total_runs} 완료 ({elapsed/60:.1f}분 경과) - 방금: {task}/{method}")

    with open('results/exp1a_confirm_progress.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
final_path = f'results/exp1a_confirm_{ts}.csv'
with open(final_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\n✓ 최종 저장: {final_path}")

print("\n===== 태스크별 평균 =====")
for task in tasks:
    print(f"\n--- {task} ---")
    task_avgs = {}
    for method in methods:
        sub = [r['score'] for r in rows if r['task']==task and r['method']==method]
        avg = sum(sub)/len(sub)
        task_avgs[method] = avg
    for method, avg in sorted(task_avgs.items(), key=lambda x: -x[1]):
        print(f"  {method}: {avg:.2f}")

print("\n===== 2개 태스크 통합 평균 (참고용) =====")
overall_avgs = {}
for method in methods:
    sub = [r['score'] for r in rows if r['method']==method]
    avg = sum(sub)/len(sub)
    overall_avgs[method] = avg
for rank, (method, avg) in enumerate(sorted(overall_avgs.items(), key=lambda x: -x[1]), 1):
    print(f"  {rank}. {method}: {avg:.2f}")

print("\n참고: Table I 조건A 원본 순위 - FullKV > PyramidKV > AdaKV > StreamingLLM > HSS > H2O > SnapKV")
print("(오늘 확인은 2개 태스크·15샘플만이라 참고용, 논문 미반영)")

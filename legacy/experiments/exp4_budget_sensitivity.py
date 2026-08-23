#!/usr/bin/env python3
"""
experiments/exp4_budget_sensitivity.py
Exp4: Budget Sensitivity Analysis (evaluator_v2 기반)
"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.results_manager import save_results_csv, save_results_json, get_timestamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"results/exp4_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TASK_ORDER = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "qmsum"]
BUDGET_RATIOS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
METHODS = ["fullkv", "adakv", "ours"]

def parse_args():
    parser = argparse.ArgumentParser(description="Exp4: Budget Sensitivity")
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--budgets", nargs="+", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def evaluate_at_budget(evaluator, method_name, tasks, budget_ratio, num_samples, seed):
    task_scores = {}
    for task_name in tasks:
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            scores = []
            for i, sample in enumerate(samples):
                r = evaluator.evaluate_sample(sample, method_name, budget_ratio, measure_efficiency=False)
                scores.append(r["score"])
                if (i+1) % 5 == 0:
                    logger.info(f"  [{method_name}@{budget_ratio:.0%}] {task_name} [{i+1}/{len(samples)}] avg={sum(scores)/len(scores):.2f}")
            avg = sum(scores) / len(scores) if scores else 0.0
            task_scores[task_name] = round(avg, 1)
            logger.info(f"  [{method_name}@{budget_ratio:.0%}] {task_name}: {avg:.2f}")
        except Exception as e:
            logger.error(f"  [{method_name}@{budget_ratio:.0%}] {task_name} failed: {e}", exc_info=True)
            task_scores[task_name] = 0.0
    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return round(avg_score, 1), task_scores

def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    timestamp = get_timestamp()
    tasks = args.tasks if args.tasks else TASK_ORDER
    budgets = args.budgets if args.budgets else BUDGET_RATIOS

    logger.info("Experiment 4: Budget Sensitivity Analysis")
    logger.info(f"  Model: {args.model} | Budgets: {[f'{b:.0%}' for b in budgets]}")

    model, tokenizer, model_config = load_model_and_tokenizer(args.model, device="cuda")
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    table_rows = []
    method_scores = {m: {} for m in METHODS}

    for budget in budgets:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Budget: {budget:.0%}")
        row = {"Budget": f"{budget:.0%}"}

        for method_name in METHODS:
            avg_score, task_scores = evaluate_at_budget(
                evaluator, method_name, tasks, budget, args.num_samples, args.seed
            )
            method_scores[method_name][budget] = avg_score
            row[method_name.upper()] = avg_score

            fullkv_score = method_scores["fullkv"].get(budget, 0)
            if method_name == "ours" and fullkv_score > 0:
                row["Pct_of_FullKV_%"] = round(avg_score / fullkv_score * 100, 1)
            if method_name == "ours":
                adakv_score = method_scores["adakv"].get(budget, 0)
                row["Δ_OURS_vs_AdaKV"] = round(avg_score - adakv_score, 1)

            logger.info(f"  {method_name.upper()}: {avg_score:.2f}")
            all_results.append({"method": method_name, "budget": budget, "avg_score": avg_score, "task_scores": task_scores})

        table_rows.append(row)

    # 결과 출력
    print("\n" + "="*70)
    print(f"  Exp4 Budget Sensitivity | {args.model}")
    print("="*70)
    header = "Budget | FULLKV | ADAKV | OURS  | %FullKV | Δ vs AdaKV"
    print(header)
    print("-"*70)
    for row in table_rows:
        print(f"{row['Budget']:>6} | {row.get('FULLKV',0):>6} | {row.get('ADAKV',0):>5} | {row.get('OURS',0):>5} | {row.get('Pct_of_FullKV_%',0):>7} | {row.get('Δ_OURS_vs_AdaKV',0):>10}")
    print("="*70)

    save_results_csv(table_rows, f"exp4_budget_sensitivity_{args.model}_{timestamp}.csv")
    save_results_json({"model": args.model, "results": all_results}, f"exp4_budget_sensitivity_{args.model}_{timestamp}.json")
    logger.info("Experiment 4 completed!")

if __name__ == "__main__":
    main()

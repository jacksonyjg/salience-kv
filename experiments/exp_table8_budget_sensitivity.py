import sys
import os
import argparse
import logging
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.results_manager import (
    save_results_csv, save_results_json, format_table_row,
    print_result_table, get_timestamp,
)

os.makedirs("logs/v2_verified", exist_ok=True)
os.makedirs("results/v2_verified", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/v2_verified/exp8_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

BUDGET_RATIOS = [0.10, 0.20, 0.40, 0.60, 0.80]

METHODS = [
    ("streamingllm", "StreamingLLM", {}),
    ("h2o", "H2O", {}),
    ("snapkv", "SnapKV", {}),
    ("pyramidkv", "PyramidKV-adapted", {}),
    ("adakv", "AdaKV-adapted", {}),
    ("ours", "SalienceKV (w/o sink)", {"sink_size": 0}),
    ("ours", "SalienceKV-Sink-4", {"sink_size": 4}),
]

DEFAULT_TASKS = ["qmsum", "hotpotqa"]


def run_method_at_budget(
    evaluator: EvaluatorV2,
    method_name: str,
    label: str,
    tasks: List[str],
    budget_ratio: float,
    num_samples: int,
    seed: int,
    method_kwargs: Dict = None,
) -> Dict:
    logger.info(f"\n{'─'*60}")
    logger.info(f"Method: {label} | Budget: {budget_ratio:.0%} | kwargs={method_kwargs}")
    logger.info(f"{'─'*60}")

    task_scores = {}
    for task_name in tasks:
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            scores = []
            for i, sample in enumerate(samples):
                r = evaluator.evaluate_sample(
                    sample, method_name, budget_ratio,
                    measure_efficiency=False, method_kwargs=method_kwargs,
                )
                scores.append(r["score"])
                if (i + 1) % 10 == 0:
                    logger.info(f"  [{label}@{budget_ratio:.0%}] {task_name} [{i+1}/{len(samples)}] "
                                f"avg={sum(scores)/len(scores):.2f}")
            avg = sum(scores) / len(scores) if scores else 0.0
            task_scores[task_name] = round(avg, 2)
            logger.info(f"  → {task_name}: {avg:.2f}")
        except Exception as e:
            logger.error(f"  {task_name} failed: {e}", exc_info=True)
            task_scores[task_name] = 0.0

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return {
        "method": label,
        "budget_ratio": budget_ratio,
        "task_scores": task_scores,
        "avg_score": round(avg_score, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Exp8: Budget Sensitivity (TABLE VIII)")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--budgets", nargs="+", type=float, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = args.tasks if args.tasks else DEFAULT_TASKS
    budgets = args.budgets if args.budgets else BUDGET_RATIOS
    timestamp = get_timestamp()

    logger.info("Experiment 8: Budget Sensitivity (TABLE VIII)")
    logger.info(f"  Model: {args.model} | Budgets: {[f'{b:.0%}' for b in budgets]} | Tasks: {tasks} | "
                f"Samples: {args.num_samples}")

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    for budget in budgets:
        for method_name, label, method_kwargs in METHODS:
            result = run_method_at_budget(
                evaluator=evaluator,
                method_name=method_name,
                label=label,
                tasks=tasks,
                budget_ratio=budget,
                num_samples=args.num_samples,
                seed=args.seed,
                method_kwargs=method_kwargs,
            )
            all_results.append(result)

            csv_filename = f"exp8_budget_sensitivity_{args.model}_{timestamp}.csv"
            json_filename = f"exp8_budget_sensitivity_{args.model}_{timestamp}.json"
            json_data_partial = {
                "experiment": "exp8_budget_sensitivity",
                "model": args.model,
                "tasks": tasks,
                "num_samples": args.num_samples,
                "seed": args.seed,
                "results": all_results,
                "completed_configs": len(all_results),
                "total_configs": len(budgets) * len(METHODS),
            }
            save_results_json(json_data_partial, json_filename)
            logger.info(f"  [중간 저장 완료] {len(all_results)}/{len(budgets) * len(METHODS)} 설정")

    save_results_csv(all_results, csv_filename)
    logger.info(f"\n완료: {len(all_results)}개 설정. 저장: results/v2_verified/{csv_filename}")


if __name__ == "__main__":
    main()

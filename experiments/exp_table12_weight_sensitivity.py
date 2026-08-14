import sys
import os
import argparse
import logging
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.results_manager import save_results_csv, save_results_json, get_timestamp

os.makedirs("logs/v2_verified", exist_ok=True)
os.makedirs("results/v2_verified", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/v2_verified/exp12_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
         "2wikimqa", "gov_report", "qmsum"]

BUDGET_RATIO = 0.20
SINK_SIZE = 4

WEIGHT_SETTINGS = [
    ("Default", 0.40, 0.20, 0.20),
    ("Uniform", 0.25, 0.25, 0.25),
    ("N-centric", 0.55, 0.15, 0.15),
]


def run_setting(evaluator, label, alpha, beta, delta, tasks, num_samples, seed):
    logger.info(f"\n{'='*60}")
    logger.info(f"Setting: {label} (α={alpha}, β={beta}, δ={delta})")
    logger.info(f"{'='*60}")

    method_kwargs = {
        "sink_size": SINK_SIZE, "alpha": alpha, "beta": beta, "delta": delta,
        "gamma": 0.0,
        "use_semantic": False,
    }

    task_scores = {}
    for task_name in tasks:
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            scores = []
            for i, sample in enumerate(samples):
                r = evaluator.evaluate_sample(
                    sample, "ours", BUDGET_RATIO,
                    measure_efficiency=False, method_kwargs=method_kwargs,
                )
                scores.append(r["score"])
                if (i + 1) % 10 == 0:
                    logger.info(f"  [{label}] {task_name} [{i+1}/{len(samples)}] "
                                f"avg={sum(scores)/len(scores):.2f}")
            avg = sum(scores) / len(scores) if scores else 0.0
            task_scores[task_name] = round(avg, 2)
            logger.info(f"  → {task_name}: {avg:.2f}")
        except Exception as e:
            logger.error(f"  {task_name} failed: {e}", exc_info=True)
            task_scores[task_name] = 0.0

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return {
        "setting": label, "alpha": alpha, "beta": beta, "delta": delta,
        "task_scores": task_scores, "avg_score": round(avg_score, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Exp12: Weight Sensitivity (TABLE XII)")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = get_timestamp()

    logger.info("Experiment 12: Weight Sensitivity (TABLE XII)")
    logger.info(f"  Model: {args.model} | Tasks: {TASKS} | Samples: {args.num_samples} | "
                f"Sink: {SINK_SIZE} | Budget: {BUDGET_RATIO:.0%}")

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    for label, alpha, beta, delta in WEIGHT_SETTINGS:
        result = run_setting(evaluator, label, alpha, beta, delta, TASKS, args.num_samples, args.seed)
        all_results.append(result)

        json_data = {
            "experiment": "exp12_weight_sensitivity",
            "model": args.model,
            "tasks": TASKS,
            "num_samples": args.num_samples,
            "budget_ratio": BUDGET_RATIO,
            "sink_size": SINK_SIZE,
            "results": all_results,
        }
        save_results_json(json_data, f"exp12_weight_sensitivity_{args.model}_{timestamp}.json")
        logger.info(f"[중간 저장 완료] {len(all_results)}/{len(WEIGHT_SETTINGS)} 설정")

    save_results_csv(all_results, f"exp12_weight_sensitivity_{args.model}_{timestamp}.csv")
    logger.info(f"\n완료: {len(all_results)}개 설정")
    for r in all_results:
        logger.info(f"  {r['setting']:<12} avg_score={r['avg_score']}")


if __name__ == "__main__":
    main()

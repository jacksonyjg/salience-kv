import sys
import os
import argparse
import logging
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed, word_repetition_ratio, char_repetition_ratio
from core.results_manager import save_results_csv, save_results_json, get_timestamp

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


def run_setting(evaluator, label, alpha, beta, delta, tasks, num_samples, seed, invert_norm=False):
    logger.info(f"\n{'='*60}")
    logger.info(f"Setting: {label} (α={alpha}, β={beta}, δ={delta}) | invert_norm={invert_norm}")
    logger.info(f"{'='*60}")

    method_kwargs = {
        "sink_size": SINK_SIZE, "alpha": alpha, "beta": beta, "delta": delta,
        "gamma": 0.0,
        "use_semantic": False,
        "invert_norm": invert_norm,
    }

    task_scores = {}
    all_sample_records = {}
    for task_name in tasks:
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            scores = []
            sample_records = []
            for i, sample in enumerate(samples):
                r = evaluator.evaluate_sample(
                    sample, "ours", BUDGET_RATIO,
                    measure_efficiency=False, method_kwargs=method_kwargs,
                )
                scores.append(r["score"])
                pred = r["prediction"]
                sample_records.append({
                    "sample_idx": i,
                    "score": r["score"],
                    "prediction": pred,
                    "word_rep": word_repetition_ratio(pred),
                    "char_rep": char_repetition_ratio(pred),
                    "collapsed": is_collapsed(pred),
                })
                if (i + 1) % 10 == 0:
                    logger.info(f"  [{label}] {task_name} [{i+1}/{len(samples)}] "
                                f"avg={sum(scores)/len(scores):.2f}")
            avg = sum(scores) / len(scores) if scores else 0.0
            task_scores[task_name] = round(avg, 2)
            all_sample_records[task_name] = sample_records
            logger.info(f"  → {task_name}: {avg:.2f}")
        except Exception as e:
            logger.error(f"  {task_name} failed: {e}", exc_info=True)
            task_scores[task_name] = 0.0

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return {
        "setting": label, "alpha": alpha, "beta": beta, "delta": delta,
        "task_scores": task_scores, "avg_score": round(avg_score, 2),
        "sample_records": all_sample_records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Exp12: Weight Sensitivity (TABLE XII)")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invert_norm", action="store_true",
                        help="key-norm 선택 방향을 corrected(low-norm 우선, Devoto et al. 방향)로 전환.")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = get_timestamp()

    log_dir = "logs/v3_verified"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/exp12_{timestamp}.log"),
        ],
        force=True,
    )

    import core.results_manager as rm
    rm.RESULTS_DIR = "results/v3_verified"
    os.makedirs(rm.RESULTS_DIR, exist_ok=True)

    logger.info("Experiment 12: Weight Sensitivity (TABLE XII)")
    logger.info(f"  Model: {args.model} | Tasks: {TASKS} | Samples: {args.num_samples} | "
                f"Sink: {SINK_SIZE} | Budget: {BUDGET_RATIO:.0%}")
    logger.info(f"  Key-norm direction: {'CORRECTED' if args.invert_norm else 'legacy'} | Results dir: {rm.RESULTS_DIR} | Log dir: {log_dir}")

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    for label, alpha, beta, delta in WEIGHT_SETTINGS:
        result = run_setting(evaluator, label, alpha, beta, delta, TASKS, args.num_samples, args.seed,
                              invert_norm=args.invert_norm)
        all_results.append(result)

        json_data = {
            "experiment": "exp12_weight_sensitivity",
            "model": args.model,
            "tasks": TASKS,
            "num_samples": args.num_samples,
            "budget_ratio": BUDGET_RATIO,
            "sink_size": SINK_SIZE,
            "invert_norm": args.invert_norm,
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

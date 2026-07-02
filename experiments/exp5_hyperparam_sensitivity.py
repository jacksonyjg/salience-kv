#!/usr/bin/env python3
"""
experiments/exp5_hyperparam_sensitivity.py
Exp5: 하이퍼파라미터 민감도 분석 (evaluator_v2 기반)
5-A: 위치 감쇠율 λ 민감도
5-B: 가중치 계수 민감도
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
        logging.FileHandler(f"results/exp5_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TASK_ORDER = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "qmsum"]

LAMBDA_CONFIGS = [
    {"lambda_pos": 0.5,  "label": "λ=0.5",         "alpha": 0.40, "beta": 0.20, "gamma": 0.20, "delta": 0.20},
    {"lambda_pos": 1.0,  "label": "λ=1.0 (기본값)", "alpha": 0.40, "beta": 0.20, "gamma": 0.20, "delta": 0.20},
    {"lambda_pos": 2.0,  "label": "λ=2.0",          "alpha": 0.40, "beta": 0.20, "gamma": 0.20, "delta": 0.20},
]

WEIGHT_CONFIGS = [
    {"label": "Default (0.40,0.20,0.20,0.20)",    "alpha": 0.40, "beta": 0.20, "gamma": 0.20, "delta": 0.20, "lambda_pos": 1.0},
    {"label": "Uniform (0.25,0.25,0.25,0.25)",    "alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25, "lambda_pos": 1.0},
    {"label": "Attn-Heavy (0.55,0.15,0.15,0.15)", "alpha": 0.55, "beta": 0.15, "gamma": 0.15, "delta": 0.15, "lambda_pos": 1.0},
]

def parse_args():
    parser = argparse.ArgumentParser(description="Exp5: Hyperparameter Sensitivity")
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--mode", default="all", choices=["lambda", "weights", "all"])
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def run_configs(evaluator, configs, tasks, budget, num_samples, seed, title):
    logger.info(f"\n{'='*60}")
    logger.info(f"{title}")
    table_rows = []
    baseline_avg = None

    for cfg in configs:
        label = cfg["label"]
        method_kwargs = {k: v for k, v in cfg.items() if k not in ("label",)}
        logger.info(f"\n--- {label} ---")

        task_scores = {}
        for task_name in tasks:
            try:
                samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
                scores = []
                for i, sample in enumerate(samples):
                    r = evaluator.evaluate_sample(sample, "ours", budget, method_kwargs=method_kwargs)
                    scores.append(r["score"])
                avg = sum(scores) / len(scores) if scores else 0.0
                task_scores[task_name] = round(avg, 1)
                logger.info(f"  {task_name}: {avg:.2f}")
            except Exception as e:
                logger.error(f"  {task_name} failed: {e}", exc_info=True)
                task_scores[task_name] = 0.0

        avg_score = sum(task_scores[t] for t in tasks) / len(tasks)
        if baseline_avg is None:
            baseline_avg = avg_score
        delta = avg_score - baseline_avg

        row = {"Config": label}
        for t in tasks:
            row[t] = task_scores.get(t, 0.0)
        row["Avg"] = round(avg_score, 1)
        row["Δ vs Default"] = round(delta, 1)
        table_rows.append(row)
        logger.info(f"  → Avg={avg_score:.2f}, Δ={delta:+.2f}")

    # 출력
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)
    print(f"{'Config':<35} | {'Avg':>6} | {'Δ vs Default':>12}")
    print("-"*80)
    for row in table_rows:
        print(f"{row['Config']:<35} | {row['Avg']:>6} | {row['Δ vs Default']:>+12}")
    print("="*80)
    return table_rows

def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    timestamp = get_timestamp()
    tasks = args.tasks if args.tasks else TASK_ORDER

    logger.info("Experiment 5: Hyperparameter Sensitivity")
    logger.info(f"  Model: {args.model} | Budget: {args.budget:.0%} | Mode: {args.mode}")

    model, tokenizer, model_config = load_model_and_tokenizer(args.model, device="cuda")
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_rows = []

    if args.mode in ("lambda", "all"):
        rows = run_configs(evaluator, LAMBDA_CONFIGS, tasks, args.budget, args.num_samples, args.seed, "5-A: λ 민감도")
        all_rows.extend(rows)

    if args.mode in ("weights", "all"):
        rows = run_configs(evaluator, WEIGHT_CONFIGS, tasks, args.budget, args.num_samples, args.seed, "5-B: 가중치 민감도")
        all_rows.extend(rows)

    save_results_csv(all_rows, f"exp5_hyperparam_{args.model}_{timestamp}.csv")
    save_results_json({"model": args.model, "budget": args.budget, "results": all_rows}, f"exp5_hyperparam_{args.model}_{timestamp}.json")
    logger.info("Experiment 5 completed!")

if __name__ == "__main__":
    main()

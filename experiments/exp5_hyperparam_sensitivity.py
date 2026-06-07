#!/usr/bin/env python3
"""
experiments/exp5_hyperparam_sensitivity.py
===========================================
실험 5: 하이퍼파라미터 민감도 분석

5-A: 위치 감쇠율 λ 민감도 (0.5, 1.0, 2.0)
5-B: 가중치 계수 (α, β, γ, δ) 민감도

실행:
    python experiments/exp5_hyperparam_sensitivity.py --model qwen3-4b --mode lambda
    python experiments/exp5_hyperparam_sensitivity.py --model qwen3-4b --mode weights
    python experiments/exp5_hyperparam_sensitivity.py --model qwen3-4b --mode all
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.kv_methods import OursHybrid
from core.evaluator import Evaluator
from core.results_manager import (
    save_results_csv, save_results_json, print_result_table, get_timestamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"results/exp5_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TASK_ORDER = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "gov_report", "qmsum",
]

# 5-A: λ 민감도
LAMBDA_CONFIGS = [
    {"lambda_pos": 0.5, "label": "λ=0.5", "interpretation": "최근성 편향 약함"},
    {"lambda_pos": 1.0, "label": "λ=1.0 (기본값)", "interpretation": "최적 균형점"},
    {"lambda_pos": 2.0, "label": "λ=2.0", "interpretation": "최근성 편향 강함"},
]

# 5-B: 가중치 계수 민감도
WEIGHT_CONFIGS = [
    {
        "label": "Default (0.40, 0.20, 0.20, 0.20)",
        "alpha": 0.40, "beta": 0.20, "gamma": 0.20, "delta": 0.20,
    },
    {
        "label": "Uniform (0.25, 0.25, 0.25, 0.25)",
        "alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25,
    },
    {
        "label": "Attn-Heavy (0.55, 0.15, 0.15, 0.15)",
        "alpha": 0.55, "beta": 0.15, "gamma": 0.15, "delta": 0.15,
    },
    {
        "label": "Sem-Heavy (0.30, 0.15, 0.40, 0.15)",
        "alpha": 0.30, "beta": 0.15, "gamma": 0.40, "delta": 0.15,
    },
]


def evaluate_config(evaluator, method_kwargs, tasks, budget_ratio, num_samples, seed):
    """단일 설정에 대해 모든 태스크 평가."""
    kv_method = OursHybrid(evaluator.cfg, **method_kwargs)
    task_scores = {}
    
    for task_name in tasks:
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            result = evaluator.evaluate_task(samples, kv_method, budget_ratio)
            task_scores[task_name] = result["avg_score"]
        except Exception as e:
            logger.error(f"  {task_name} failed: {e}")
            task_scores[task_name] = 0.0
    
    avg = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return avg, task_scores


def parse_args():
    parser = argparse.ArgumentParser(description="Exp5: Hyperparameter Sensitivity")
    parser.add_argument("--model", default="qwen3-4b",
                        choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--mode", default="all", choices=["lambda", "weights", "all"])
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_5a_lambda(evaluator, tasks, args, timestamp):
    """5-A: λ 민감도 실험."""
    logger.info("\n--- 5-A: Lambda Sensitivity ---")
    
    table_rows = []
    baseline_avg = None
    all_results = []
    
    for cfg in LAMBDA_CONFIGS:
        avg, task_scores = evaluate_config(
            evaluator,
            {"lambda_pos": cfg["lambda_pos"]},
            tasks, args.budget, args.num_samples, args.seed,
        )
        
        if baseline_avg is None:
            baseline_avg = avg
        
        row = {
            "Lambda": cfg["label"],
            "Avg_Score": round(avg, 1),
            "Δ vs λ=1.0": round(avg - LAMBDA_CONFIGS[1]["lambda_pos"], 1),  # vs 기본값
            "Interpretation": cfg["interpretation"],
        }
        # 실제 delta는 기본값(인덱스1) 기준으로 나중에 계산
        table_rows.append(row)
        all_results.append({"lambda": cfg["lambda_pos"], "avg_score": avg})
        logger.info(f"  λ={cfg['lambda_pos']}: avg={avg:.2f}")
    
    # delta 재계산 (λ=1.0 기준)
    base_score = all_results[1]["avg_score"]
    for i, row in enumerate(table_rows):
        row["Δ vs λ=1.0"] = round(all_results[i]["avg_score"] - base_score, 1)
    
    print_result_table(table_rows, title=f"Exp5-A Lambda Sensitivity | {args.model}")
    save_results_csv(table_rows, f"exp5a_lambda_{args.model}_{timestamp}.csv")
    return all_results


def run_5b_weights(evaluator, tasks, args, timestamp):
    """5-B: 가중치 계수 민감도 실험."""
    logger.info("\n--- 5-B: Weight Sensitivity ---")
    
    table_rows = []
    baseline_avg = None
    all_results = []
    
    for cfg in WEIGHT_CONFIGS:
        kwargs = {
            "alpha": cfg["alpha"],
            "beta": cfg["beta"],
            "gamma": cfg["gamma"],
            "delta": cfg["delta"],
        }
        avg, task_scores = evaluate_config(
            evaluator, kwargs, tasks, args.budget, args.num_samples, args.seed
        )
        
        if baseline_avg is None:
            baseline_avg = avg
        
        row = {
            "Config": cfg["label"],
            "α": cfg["alpha"],
            "β": cfg["beta"],
            "γ": cfg["gamma"],
            "δ": cfg["delta"],
            "Avg_Score": round(avg, 1),
            "Δ vs Default": round(avg - baseline_avg, 1),
        }
        table_rows.append(row)
        all_results.append({**kwargs, "label": cfg["label"], "avg_score": avg})
        logger.info(f"  {cfg['label']}: avg={avg:.2f}")
    
    print_result_table(table_rows, title=f"Exp5-B Weight Sensitivity | {args.model}")
    save_results_csv(table_rows, f"exp5b_weights_{args.model}_{timestamp}.csv")
    return all_results


def main():
    args = parse_args()
    
    os.makedirs("results", exist_ok=True)
    timestamp = get_timestamp()
    tasks = args.tasks if args.tasks else TASK_ORDER
    
    logger.info("Experiment 5: Hyperparameter Sensitivity")
    logger.info(f"  Model: {args.model} | Budget: {args.budget:.0%} | Mode: {args.mode}")
    
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, device="cuda"
    )
    evaluator = Evaluator(model, tokenizer, model_config, seed=args.seed)
    
    results_5a, results_5b = None, None
    
    if args.mode in ("lambda", "all"):
        results_5a = run_5a_lambda(evaluator, tasks, args, timestamp)
    
    if args.mode in ("weights", "all"):
        results_5b = run_5b_weights(evaluator, tasks, args, timestamp)
    
    save_results_json(
        {
            "experiment": "exp5",
            "model": args.model,
            "mode": args.mode,
            "5a_lambda": results_5a,
            "5b_weights": results_5b,
        },
        f"exp5_{args.mode}_{args.model}_{timestamp}.json",
    )
    
    logger.info("Experiment 5 completed!")


if __name__ == "__main__":
    main()

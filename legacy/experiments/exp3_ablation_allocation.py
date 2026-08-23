#!/usr/bin/env python3
"""
experiments/exp3_ablation_allocation.py
=========================================
실험 3: Ablation Study — 레이어별 예산 할당 전략

목적: 동적 entropy 기반 할당 vs. 대안 전략 비교 (H2 검증)

비교 전략:
- Entropy-Driven (Ours): 실시간 attention entropy 기반
- Uniform: 모든 레이어 동일 예산
- Static Pyramidal: 정적 피라미드 (얕은 → 큰 예산)
- Random: 무작위 예산 할당

실행:
    python experiments/exp3_ablation_allocation.py --model qwen3-4b
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
        logging.FileHandler(f"results/exp3_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TASK_ORDER = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "gov_report", "qmsum",
]

# 논문 Table IV: 할당 전략 비교
ALLOCATION_STRATEGIES = [
    {
        "name": "Entropy-Driven (Ours)",
        "strategy": "entropy",
        "description": "실시간 attention entropy 기반",
    },
    {
        "name": "Uniform (Equal)",
        "strategy": "uniform",
        "description": "모든 레이어 동일 예산",
    },
    {
        "name": "Static Pyramidal",
        "strategy": "pyramid",
        "description": "깊은 레이어 → 더 작은 예산 (정적)",
    },
    {
        "name": "Random",
        "strategy": "random",
        "description": "무작위 예산 할당",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Exp3: Ablation - Allocation Strategy")
    parser.add_argument("--model", default="qwen3-4b",
                        choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs("results", exist_ok=True)
    timestamp = get_timestamp()
    
    tasks = args.tasks if args.tasks else TASK_ORDER
    
    logger.info("Experiment 3: Ablation - Layer Budget Allocation")
    logger.info(f"  Model: {args.model} | Budget: {args.budget:.0%}")
    
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, device="cuda"
    )
    evaluator = Evaluator(model, tokenizer, model_config, seed=args.seed)
    
    all_results = []
    table_rows = []
    baseline_avg = None
    
    for strategy_cfg in ALLOCATION_STRATEGIES:
        name = strategy_cfg["name"]
        strategy = strategy_cfg["strategy"]
        
        logger.info(f"\n{'─'*60}")
        logger.info(f"Strategy: {name}")
        
        kv_method = OursHybrid(
            model_config,
            allocation_strategy=strategy,
        )
        
        task_scores = {}
        
        for task_name in tasks:
            try:
                samples = load_longbench_task(task_name, num_samples=args.num_samples, seed=args.seed)
                result = evaluator.evaluate_task(samples, kv_method, args.budget)
                task_scores[task_name] = result["avg_score"]
                logger.info(f"  {task_name}: {result['avg_score']:.2f}")
            except Exception as e:
                logger.error(f"  {task_name} failed: {e}")
                task_scores[task_name] = 0.0
        
        avg_score = sum(task_scores.values()) / len(task_scores)
        
        if baseline_avg is None:
            baseline_avg = avg_score  # Entropy-Driven이 기준
        
        delta = avg_score - baseline_avg
        
        row = {
            "Strategy": name,
            "Description": strategy_cfg["description"],
        }
        row["Avg_Score"] = round(avg_score, 1)
        row["Δ vs Entropy-Driven"] = round(delta, 1)
        
        table_rows.append(row)
        all_results.append({
            "strategy": name,
            "task_scores": task_scores,
            "avg_score": avg_score,
            "delta": delta,
        })
    
    print_result_table(table_rows, title=f"Exp3 Ablation - Allocation Strategy | {args.model}")
    save_results_csv(table_rows, f"exp3_allocation_{args.model}_{timestamp}.csv")
    save_results_json(
        {"experiment": "exp3", "model": args.model, "results": all_results},
        f"exp3_allocation_{args.model}_{timestamp}.json",
    )
    
    logger.info("Experiment 3 completed!")


if __name__ == "__main__":
    main()

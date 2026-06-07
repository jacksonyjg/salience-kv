#!/usr/bin/env python3
"""
experiments/exp4_budget_sensitivity.py
=========================================
실험 4: 예산 민감도 분석 (Budget Sensitivity)

목적: 캐시 예산 비율에 따른 성능 확장성 분석, 최적 operating point 확인
검증: 30% 예산에서 Full KV 동등 성능 달성 (Ada-KV는 40~50% 필요)

실행:
    python experiments/exp4_budget_sensitivity.py --model qwen3-4b
    python experiments/exp4_budget_sensitivity.py --model qwen3-4b --num_samples 20
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.kv_methods import create_kv_method
from core.evaluator import Evaluator
from core.results_manager import (
    save_results_csv, save_results_json, print_result_table, get_timestamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"results/exp4_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TASK_ORDER = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "gov_report", "qmsum",
]

# 논문 Table V: 예산 범위
BUDGET_RATIOS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# 비교 메서드
METHODS = ["adakv", "ours"]  # Full KV는 budget_ratio=1.0으로 처리


def parse_args():
    parser = argparse.ArgumentParser(description="Exp4: Budget Sensitivity")
    parser.add_argument("--model", default="qwen3-4b",
                        choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--budgets", nargs="+", type=float, default=None,
                        help="예산 비율 리스트 (기본값: 0.10 0.15 0.20 0.25 0.30 0.40 0.50)")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_at_budget(evaluator, method_name, tasks, budget_ratio, num_samples, seed):
    """특정 예산에서 모든 태스크 평가."""
    kv_method = create_kv_method(method_name, evaluator.cfg)
    task_scores = {}
    
    for task_name in tasks:
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            result = evaluator.evaluate_task(samples, kv_method, budget_ratio)
            task_scores[task_name] = result["avg_score"]
        except Exception as e:
            logger.error(f"  [{method_name}@{budget_ratio:.0%}] {task_name} failed: {e}")
            task_scores[task_name] = 0.0
    
    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return avg_score, task_scores


def main():
    args = parse_args()
    
    os.makedirs("results", exist_ok=True)
    timestamp = get_timestamp()
    
    tasks = args.tasks if args.tasks else TASK_ORDER
    budgets = args.budgets if args.budgets else BUDGET_RATIOS
    
    logger.info("Experiment 4: Budget Sensitivity Analysis")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Budgets: {[f'{b:.0%}' for b in budgets]}")
    
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, device="cuda"
    )
    evaluator = Evaluator(model, tokenizer, model_config, seed=args.seed)
    
    # Full KV 기준값 측정 (budget=1.0 = 압축 없음)
    logger.info("\nMeasuring Full KV baseline ...")
    fullkv_avg, _ = evaluate_at_budget(
        evaluator, "fullkv", tasks, 1.0, args.num_samples, args.seed
    )
    logger.info(f"Full KV avg_score: {fullkv_avg:.2f}")
    
    all_results = []
    table_rows = []
    method_scores = {m: {} for m in METHODS}
    
    for budget in budgets:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Budget: {budget:.0%}")
        
        row = {"Budget": f"{budget:.0%}", "Full_KV": round(fullkv_avg, 1)}
        
        for method_name in METHODS:
            avg_score, task_scores = evaluate_at_budget(
                evaluator, method_name, tasks, budget, args.num_samples, args.seed
            )
            method_scores[method_name][budget] = avg_score
            
            col_name = method_name.upper()
            pct_of_fullkv = (avg_score / fullkv_avg * 100) if fullkv_avg > 0 else 0.0
            
            row[col_name] = round(avg_score, 1)
            row[f"Δ_{col_name}_vs_AdaKV"] = (
                round(avg_score - method_scores["adakv"].get(budget, 0), 1)
                if method_name == "ours" else "-"
            )
            
            if method_name == "ours":
                row["Pct_of_FullKV_%"] = round(pct_of_fullkv, 1)
            
            logger.info(
                f"  {method_name.upper()}: {avg_score:.2f} ({pct_of_fullkv:.1f}% of FullKV)"
            )
            
            all_results.append({
                "method": method_name,
                "budget": budget,
                "avg_score": avg_score,
                "pct_of_fullkv": pct_of_fullkv,
                "task_scores": task_scores,
            })
        
        table_rows.append(row)
    
    print_result_table(table_rows, title=f"Exp4 Budget Sensitivity | {args.model}")
    save_results_csv(table_rows, f"exp4_budget_sensitivity_{args.model}_{timestamp}.csv")
    save_results_json(
        {
            "experiment": "exp4",
            "model": args.model,
            "fullkv_avg": fullkv_avg,
            "results": all_results,
        },
        f"exp4_budget_sensitivity_{args.model}_{timestamp}.json",
    )
    
    # 핵심 검증: 몇 % 예산에서 Full KV 달성?
    logger.info("\n" + "="*60)
    logger.info("KEY FINDING: Budget at which Full KV performance is reached")
    for method_name in METHODS:
        for budget, score in sorted(method_scores[method_name].items()):
            if fullkv_avg > 0 and score >= fullkv_avg:
                logger.info(f"  {method_name.upper()}: {budget:.0%} budget achieves Full KV level")
                break
    
    logger.info("Experiment 4 completed!")


if __name__ == "__main__":
    main()

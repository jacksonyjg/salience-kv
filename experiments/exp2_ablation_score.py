#!/usr/bin/env python3
"""
experiments/exp2_ablation_score.py
=====================================
실험 2: Ablation Study — Hybrid Score 구성 요소 기여도

목적: 4개 점수 성분(Attention, Entropy, Semantic, Position) 각각의 기여도 검증 (H1 검증)

실행:
    python experiments/exp2_ablation_score.py --model qwen3-4b
    python experiments/exp2_ablation_score.py --model qwen3-4b --num_samples 20
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
        logging.FileHandler(f"results/exp2_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TASK_ORDER = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "gov_report", "qmsum",
]

# 논문 Table III: 구성 요소 제거 실험
ABLATION_CONFIGS = [
    {
        "name": "Full (All 4 signals)",
        "use_attention": True, "use_entropy": True,
        "use_semantic": True, "use_position": True,
    },
    {
        "name": "w/o Attention",
        "use_attention": False, "use_entropy": True,
        "use_semantic": True, "use_position": True,
        # 가중치 재분배: α→0, 나머지 균등
        "alpha": 0.0, "beta": 0.33, "gamma": 0.33, "delta": 0.34,
    },
    {
        "name": "w/o Semantic",
        "use_attention": True, "use_entropy": True,
        "use_semantic": False, "use_position": True,
        "alpha": 0.50, "beta": 0.25, "gamma": 0.0, "delta": 0.25,
    },
    {
        "name": "w/o Entropy",
        "use_attention": True, "use_entropy": False,
        "use_semantic": True, "use_position": True,
        "alpha": 0.50, "beta": 0.0, "gamma": 0.25, "delta": 0.25,
    },
    {
        "name": "w/o Position",
        "use_attention": True, "use_entropy": True,
        "use_semantic": True, "use_position": False,
        "alpha": 0.50, "beta": 0.25, "gamma": 0.25, "delta": 0.0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Exp2: Ablation - Score Components")
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
    
    logger.info("Experiment 2: Ablation - Hybrid Score Components")
    logger.info(f"  Model: {args.model} | Budget: {args.budget:.0%}")
    
    # 모델 로드
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, device="cuda"
    )
    evaluator = Evaluator(model, tokenizer, model_config, seed=args.seed)
    
    all_results = []
    table_rows = []
    baseline_avg = None
    
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"Condition: {name}")
        
        # OursHybrid 인스턴스 생성 (ablation 플래그 적용)
        method_kwargs = {
            "alpha": cfg.get("alpha", 0.40),
            "beta": cfg.get("beta", 0.20),
            "gamma": cfg.get("gamma", 0.20),
            "delta": cfg.get("delta", 0.20),
            "use_attention": cfg["use_attention"],
            "use_entropy": cfg["use_entropy"],
            "use_semantic": cfg["use_semantic"],
            "use_position": cfg["use_position"],
        }
        kv_method = OursHybrid(model_config, **method_kwargs)
        
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
            baseline_avg = avg_score
        
        delta = avg_score - baseline_avg
        
        row = {"Condition": name}
        for task in tasks:
            row[task] = round(task_scores.get(task, 0.0), 1)
        row["Avg"] = round(avg_score, 1)
        row["Δ vs Full"] = round(delta, 1)
        
        table_rows.append(row)
        all_results.append({
            "condition": name,
            "task_scores": task_scores,
            "avg_score": avg_score,
            "delta": delta,
        })
    
    # 출력 및 저장
    print_result_table(table_rows, title=f"Exp2 Ablation - Score Components | {args.model}")
    save_results_csv(table_rows, f"exp2_score_ablation_{args.model}_{timestamp}.csv")
    save_results_json(
        {"experiment": "exp2", "model": args.model, "results": all_results},
        f"exp2_score_ablation_{args.model}_{timestamp}.json",
    )
    
    logger.info("Experiment 2 completed!")


if __name__ == "__main__":
    main()

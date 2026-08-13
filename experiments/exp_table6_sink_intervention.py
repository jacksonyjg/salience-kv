#!/usr/bin/env python3
"""
experiments/exp_table6_sink_intervention.py
========================================
TABLE VI: Sink Intervention (m 스윕)

목적: sink4가 SalienceKV에만 유리한지, 다른 score-based eviction 방법에도
범용적인지 검증 (StreamingLLM 대비 novelty 방어 핵심 근거, Plan v3 §3-3 참고)

5개 방법(SalienceKV, AdaKV-adapted, PyramidKV-adapted, H2O, SnapKV) x
6개 sink 크기(m=0,1,2,4,8,16) = 30개 설정.
m 범위를 StreamingLLM 원 논문의 1/2/4/8과 겹치게 확장(+0, +16)해서 직접 비교 가능.

실행:
    python experiments/exp_table6_sink_intervention.py --model qwen3-4b \
        --budget 0.20 --tasks qmsum gov_report --num_samples 30
"""

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
        logging.FileHandler(f"logs/v2_verified/exp6_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

# 5개 base 방법 x 6개 sink 크기 = 30개 설정
BASE_METHODS = [
    ("ours", "SalienceKV"),
    ("adakv", "AdaKV-adapted"),
    ("pyramidkv", "PyramidKV-adapted"),
    ("h2o", "H2O"),
    ("snapkv", "SnapKV"),
]
SINK_SIZES = [0, 1, 2, 4, 8, 16]

METHODS = [
    (method_name, f"{label}_m{m}", {"sink_size": m})
    for method_name, label in BASE_METHODS
    for m in SINK_SIZES
]


def run_method_on_all_tasks(
    evaluator: EvaluatorV2,
    method_name: str,
    tasks: List[str],
    budget_ratio: float,
    num_samples: int,
    seed: int,
    label: str = None,
    method_kwargs: Dict = None,
) -> Dict:
    display = label or method_name
    logger.info(f"\n{'─'*60}")
    logger.info(f"Method: {display} | Budget: {budget_ratio:.0%} | kwargs={method_kwargs}")
    logger.info(f"{'─'*60}")

    task_scores = {}
    task_collapse_rates = {}
    task_collapse_fracs = {}
    task_ttfts = []
    task_throughputs = []
    task_mem_reductions = []
    total_collapse_count = 0
    total_collapse_total = 0

    for task_name in tasks:
        logger.info(f"\n  Task: {task_name}")
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            result = evaluator.evaluate_task(
                samples=samples,
                method_name=method_name,
                budget_ratio=budget_ratio,
                method_kwargs=method_kwargs,
            )
            task_scores[task_name] = result["avg_score"]
            task_collapse_rates[task_name] = result["avg_collapse_rate_pct"]
            task_collapse_fracs[task_name] = f"{result['collapse_count']}/{result['collapse_total']}"
            total_collapse_count += result["collapse_count"]
            total_collapse_total += result["collapse_total"]
            task_ttfts.append(result["avg_ttft_ms"])
            task_throughputs.append(result["avg_throughput"])
            task_mem_reductions.append(result["avg_memory_reduction_pct"])

            logger.info(
                f"  → score={result['avg_score']:.2f}, "
                f"collapse={result['collapse_count']}/{result['collapse_total']} ({result['avg_collapse_rate_pct']:.1f}%), "
                f"mem_red={result['avg_memory_reduction_pct']:.1f}%"
            )
        except Exception as e:
            logger.error(f"  Task {task_name} failed: {e}")
            task_scores[task_name] = 0.0
            task_collapse_rates[task_name] = 100.0
            task_collapse_fracs[task_name] = "N/A"

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    avg_collapse = sum(task_collapse_rates.values()) / len(task_collapse_rates) if task_collapse_rates else 0.0
    avg_ttft = sum(task_ttfts) / len(task_ttfts) if task_ttfts else 0.0
    avg_throughput = sum(task_throughputs) / len(task_throughputs) if task_throughputs else 0.0
    avg_mem = sum(task_mem_reductions) / len(task_mem_reductions) if task_mem_reductions else 0.0

    return {
        "method": display,
        "task_scores": task_scores,
        "task_collapse_rates": task_collapse_rates,
        "task_collapse_fracs": task_collapse_fracs,
        "avg_score": avg_score,
        "avg_collapse_rate_pct": avg_collapse,
        "avg_collapse_frac": f"{total_collapse_count}/{total_collapse_total}",
        "avg_ttft_ms": avg_ttft,
        "avg_throughput": avg_throughput,
        "avg_memory_reduction_pct": avg_mem,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Exp6: Sink Intervention (TABLE VI)")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--tasks", nargs="+", default=["qmsum", "gov_report"])
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = get_timestamp()

    logger.info("=" * 60)
    logger.info("Table6: Sink Intervention (TABLE VI)")
    logger.info(f"  Model: {args.model} | Budget: {args.budget:.0%}")
    logger.info(f"  Tasks: {args.tasks} | Samples: {args.num_samples}")
    logger.info(f"  Methods: {len(METHODS)}개 설정 (5방법 x 6 sink크기)")
    logger.info("=" * 60)

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    table_rows = []
    csv_filename = f"exp6_sink_intervention_{args.model}_budget{int(args.budget*100)}_{timestamp}.csv"
    json_filename = f"exp6_sink_intervention_{args.model}_budget{int(args.budget*100)}_{timestamp}.json"

    for method_name, label, method_kwargs in METHODS:
        result = run_method_on_all_tasks(
            evaluator=evaluator,
            method_name=method_name,
            tasks=args.tasks,
            budget_ratio=args.budget,
            num_samples=args.num_samples,
            seed=args.seed,
            label=label,
            method_kwargs=method_kwargs,
        )
        all_results.append(result)

        row = format_table_row(
            method=label,
            task_scores=result["task_scores"],
            task_order=args.tasks,
            avg_score=result["avg_score"],
            mem_reduction=result["avg_memory_reduction_pct"],
            ttft_ms=result["avg_ttft_ms"],
            throughput=result["avg_throughput"],
        )
        for task in args.tasks:
            row[f"{task}_collapse_%"] = round(result["task_collapse_rates"].get(task, 0.0), 1)
            row[f"{task}_collapse_n"] = result["task_collapse_fracs"].get(task, "N/A")
        row["Avg_Collapse_%"] = round(result["avg_collapse_rate_pct"], 1)
        row["Avg_Collapse_n"] = result["avg_collapse_frac"]
        table_rows.append(row)

        save_results_csv(table_rows, csv_filename)
        json_data_partial = {
            "experiment": "exp6_sink_intervention",
            "model": args.model,
            "budget_ratio": args.budget,
            "tasks": args.tasks,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "results": all_results,
            "completed_methods": len(all_results),
            "total_methods": len(METHODS),
        }
        save_results_json(json_data_partial, json_filename)
        logger.info(f"  [중간 저장 완료] {len(all_results)}/{len(METHODS)} 설정")

    print_result_table(
        table_rows,
        title=f"Table6 Sink Intervention | {args.model} | Budget={args.budget:.0%}"
    )
    logger.info("Table VI (Sink Intervention) completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
experiments/exp_table7_signal_ablation.py
============================================
TABLE VII: Signal Ablation (Ablation A - sink 없이, m=0 고정)
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
        logging.FileHandler(f"logs/v2_verified/exp7_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

def make_variants(sink_size: int):
    """신호 조합 6개, sink_size는 외부에서 고정값으로 주입."""
    return [
        ("3signal_NVP", {"use_attention": True, "use_entropy": True, "use_position": True, "use_semantic": False, "sink_size": sink_size}),
        ("4signal_NVPS", {"use_attention": True, "use_entropy": True, "use_position": True, "use_semantic": True, "sink_size": sink_size}),
        ("wo_N", {"use_attention": False, "use_entropy": True, "use_position": True, "use_semantic": False, "sink_size": sink_size}),
        ("wo_V", {"use_attention": True, "use_entropy": False, "use_position": True, "use_semantic": False, "sink_size": sink_size}),
        ("wo_P", {"use_attention": True, "use_entropy": True, "use_position": False, "use_semantic": False, "sink_size": sink_size}),
        ("N_only", {"use_attention": True, "use_entropy": False, "use_position": False, "use_semantic": False, "sink_size": sink_size}),
    ]


def run_method_on_all_tasks(
    evaluator: EvaluatorV2,
    tasks: List[str],
    budget_ratio: float,
    num_samples: int,
    seed: int,
    label: str,
    method_kwargs: Dict,
) -> Dict:
    logger.info(f"\n{'─'*60}")
    logger.info(f"Variant: {label} | Budget: {budget_ratio:.0%} | kwargs={method_kwargs}")
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
                method_name="ours",
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
                f"collapse={result['collapse_count']}/{result['collapse_total']} ({result['avg_collapse_rate_pct']:.1f}%)"
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
        "method": label,
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
    parser = argparse.ArgumentParser(description="Exp7: Signal Ablation (TABLE VII)")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--tasks", nargs="+", default=["qmsum", "gov_report"])
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sink_size", type=int, default=0, help="모든 variant에 고정 적용할 sink 크기 (기본 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = get_timestamp()
    variants = make_variants(args.sink_size)

    logger.info("=" * 60)
    logger.info("Table7: Signal Ablation (TABLE VII)")
    logger.info(f"  Model: {args.model} | Budget: {args.budget:.0%} | sink_size={args.sink_size} 고정")
    logger.info(f"  Tasks: {args.tasks} | Samples: {args.num_samples}")
    logger.info(f"  Variants: {len(variants)}개")
    logger.info("=" * 60)

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    table_rows = []
    csv_filename = f"exp7_signal_ablation_{args.model}_sink{args.sink_size}_{timestamp}.csv"
    json_filename = f"exp7_signal_ablation_{args.model}_sink{args.sink_size}_{timestamp}.json"

    for label, method_kwargs in variants:
        result = run_method_on_all_tasks(
            evaluator=evaluator,
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
            "experiment": "exp7_signal_ablation",
            "model": args.model,
            "budget_ratio": args.budget,
            "tasks": args.tasks,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "results": all_results,
            "completed_variants": len(all_results),
            "total_variants": len(variants),
        }
        save_results_json(json_data_partial, json_filename)
        logger.info(f"  [중간 저장 완료] {len(all_results)}/{len(variants)} 설정")

    print_result_table(
        table_rows,
        title=f"Table7 Signal Ablation | {args.model} | Budget={args.budget:.0%}"
    )
    logger.info("Table VII (Signal Ablation) completed!")


if __name__ == "__main__":
    main()

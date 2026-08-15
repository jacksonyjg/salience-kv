#!/usr/bin/env python3
"""
experiments/exp_table9_efficiency.py
============================================
TABLE IX: Efficiency (Prefill / Compression / First Decode Step / Decode Throughput)

목적: V5의 뭉뚱그린 ttft_ms 단일 지표를 4개 항목으로 세분화하여 재측정.
evaluator_v2.py(2026-08-15 수정)가 반환하는 prefill_ms, compress_ms,
first_decode_step_ms, decode_throughput 필드를 사용.

8개 방법 비교 (budget=20% 고정, 논문 주 실험 조건과 동일):
  FullKV, StreamingLLM, H2O, SnapKV, PyramidKV-adapted, AdaKV-adapted,
  SalienceKV(w/o sink, sink=0), SalienceKV-Sink-4(sink=4)

입력 캡 정책: --cap 인자로 제어.
  --cap default : 기존 16k 하드캡 그대로 (다른 표들과 조건 동일, 회귀비교용)
  --cap uncapped: max_input_length=-1 (모델 실제 max_length까지, 효율성 지표 왜곡 방지)
  --cap both    : 두 조건 모두 실행해서 비교 (권장 - 준비계획 6번 결정 근거 확보)

실행 (TABLE VIII 완료 후, evaluator_v2.py 교체 + 회귀검증 끝난 뒤):
    # 1~2샘플 형식 검증 먼저
    python experiments/exp_table9_efficiency.py --model qwen3-4b \
        --budget 0.20 --tasks qmsum --num_samples 2 --cap default

    # 본실행 (권장: both로 캡 영향까지 같이 확인)
    python experiments/exp_table9_efficiency.py --model qwen3-4b \
        --budget 0.20 --tasks narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa gov_report qmsum \
        --num_samples 30 --cap both
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
    save_results_csv, save_results_json, get_timestamp,
)

os.makedirs("logs/v2_verified", exist_ok=True)
os.makedirs("results/v2_verified", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/v2_verified/exp9_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

METHODS = [
    ("FullKV", "fullkv", {}),
    ("StreamingLLM", "streamingllm", {}),
    ("H2O", "h2o", {}),
    ("SnapKV", "snapkv", {}),
    ("PyramidKV-adapted", "pyramidkv", {}),
    ("AdaKV-adapted", "adakv", {}),
    ("SalienceKV (w/o sink)", "ours", {"sink_size": 0}),
    ("SalienceKV-Sink-4", "ours", {"sink_size": 4}),
]


def run_method_on_all_tasks(
    evaluator: EvaluatorV2,
    tasks: List[str],
    budget_ratio: float,
    num_samples: int,
    seed: int,
    label: str,
    method_name: str,
    method_kwargs: Dict,
    max_input_length,
    cap_label: str,
) -> Dict:
    logger.info(f"\n{'─'*60}")
    logger.info(f"Method: {label} ({method_name}) | Budget: {budget_ratio:.0%} | "
                f"kwargs={method_kwargs} | cap={cap_label}")
    logger.info(f"{'─'*60}")

    task_prefill_ms, task_compress_ms = [], []
    task_first_decode_ms, task_decode_tp = [], []
    task_ttfts, task_throughputs, task_mem_reductions = [], [], []

    for task_name in tasks:
        logger.info(f"\n  Task: {task_name}")
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            result = evaluator.evaluate_task(
                samples=samples,
                method_name=method_name,
                budget_ratio=budget_ratio,
                method_kwargs=method_kwargs,
                max_input_length=max_input_length,
            )
            task_prefill_ms.append(result["avg_prefill_ms"])
            task_compress_ms.append(result["avg_compress_ms"])
            task_first_decode_ms.append(result["avg_first_decode_step_ms"])
            task_decode_tp.append(result["avg_decode_throughput"])
            task_ttfts.append(result["avg_ttft_ms"])
            task_throughputs.append(result["avg_throughput"])
            task_mem_reductions.append(result["avg_memory_reduction_pct"])

            logger.info(
                f"  → prefill={result['avg_prefill_ms']:.1f}ms, "
                f"compress={result['avg_compress_ms']:.1f}ms, "
                f"first_decode={result['avg_first_decode_step_ms']:.1f}ms, "
                f"decode_tp={result['avg_decode_throughput']:.2f}tok/s"
            )
        except Exception as e:
            logger.error(f"  Task {task_name} failed: {e}")

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "method": label,
        "method_name": method_name,
        "cap": cap_label,
        "avg_prefill_ms": _avg(task_prefill_ms),
        "avg_compress_ms": _avg(task_compress_ms),
        "avg_first_decode_step_ms": _avg(task_first_decode_ms),
        "avg_decode_throughput": _avg(task_decode_tp),
        "avg_ttft_ms": _avg(task_ttfts),
        "avg_throughput": _avg(task_throughputs),
        "avg_memory_reduction_pct": _avg(task_mem_reductions),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Exp9: Efficiency (TABLE IX)")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--tasks", nargs="+", default=["qmsum", "gov_report"])
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", choices=["default", "uncapped", "both"], default="default",
                         help="default=기존 16k 캡 유지, uncapped=캡 해제(-1), both=두 조건 모두 실행")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = get_timestamp()

    if args.cap == "default":
        cap_conditions = [("default", None)]
    elif args.cap == "uncapped":
        cap_conditions = [("uncapped", -1)]
    else:
        cap_conditions = [("default", None), ("uncapped", -1)]

    logger.info("=" * 60)
    logger.info("Table9: Efficiency (TABLE IX)")
    logger.info(f"  Model: {args.model} | Budget: {args.budget:.0%}")
    logger.info(f"  Tasks: {args.tasks} | Samples: {args.num_samples}")
    logger.info(f"  Methods: {len(METHODS)}개 | Cap conditions: {[c[0] for c in cap_conditions]}")
    logger.info("=" * 60)

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    table_rows = []
    csv_filename = f"exp9_efficiency_{args.model}_{timestamp}.csv"
    json_filename = f"exp9_efficiency_{args.model}_{timestamp}.json"

    for cap_label, max_input_length in cap_conditions:
        for label, method_name, method_kwargs in METHODS:
            result = run_method_on_all_tasks(
                evaluator=evaluator,
                tasks=args.tasks,
                budget_ratio=args.budget,
                num_samples=args.num_samples,
                seed=args.seed,
                label=label,
                method_name=method_name,
                method_kwargs=method_kwargs,
                max_input_length=max_input_length,
                cap_label=cap_label,
            )
            all_results.append(result)

            table_rows.append({
                "Method": label,
                "Cap": cap_label,
                "Budget": f"{args.budget:.0%}",
                "Prefill_ms": round(result["avg_prefill_ms"], 1),
                "Compression_ms": round(result["avg_compress_ms"], 1),
                "First_Decode_Step_ms": round(result["avg_first_decode_step_ms"], 1),
                "Decode_Throughput_tok_s": round(result["avg_decode_throughput"], 2),
                "Legacy_TTFT_ms": round(result["avg_ttft_ms"], 1),
                "Legacy_Throughput_tok_s": round(result["avg_throughput"], 2),
                "Memory_Reduction_pct": round(result["avg_memory_reduction_pct"], 1),
            })

            save_results_csv(table_rows, csv_filename)
            json_data_partial = {
                "experiment": "exp9_efficiency",
                "model": args.model,
                "budget_ratio": args.budget,
                "tasks": args.tasks,
                "num_samples": args.num_samples,
                "seed": args.seed,
                "cap_conditions": [c[0] for c in cap_conditions],
                "results": all_results,
                "completed": len(all_results),
                "total": len(METHODS) * len(cap_conditions),
            }
            save_results_json(json_data_partial, json_filename)
            logger.info(f"  [중간 저장 완료] {len(all_results)}/{len(METHODS) * len(cap_conditions)}")

    logger.info("\n" + "=" * 60)
    logger.info("Table IX (Efficiency) completed!")
    logger.info(f"CSV: results/v2_verified/{csv_filename}")
    logger.info(f"JSON: results/v2_verified/{json_filename}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

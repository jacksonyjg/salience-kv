#!/usr/bin/env python3
"""
experiments/exp1_main_results.py
==================================
실험 1: 주요 성능 비교 (Main Results)

목적: 제안 방법 vs. 6개 베이스라인 성능 비교 (H3 검증)
- 1-A: Qwen3-4B (주 실험) — 전체 7개 베이스라인 비교
- 1-B: Phi-3-mini (교차 아키텍처 검증)
- 1-C: Gemma-2-2B (교차 아키텍처 검증)

실행:
    # 주 실험 (Qwen3-4B 전체)
    python experiments/exp1_main_results.py --model qwen3-4b --mode full

    # Phi-3 교차 검증
    python experiments/exp1_main_results.py --model phi-3-mini --mode cross

    # 빠른 테스트 (샘플 수 제한)
    python experiments/exp1_main_results.py --model qwen3-4b --mode full --num_samples 20
"""

import sys
import os
import argparse
import logging
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task, ALL_TASKS
from core.evaluator_v2 import EvaluatorV2
from core.results_manager import (
    save_results_csv, save_results_json, format_table_row,
    print_result_table, get_timestamp,
)

logger = logging.getLogger(__name__)

# 실험계획서 Table I 태스크 순서
TASK_ORDER = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "gov_report", "qmsum",
]

# 모드별 비교 메서드
# [2026-08-12] Plan v3 반영: SalienceKV(ours)를 w/o sink(m=0)와 Sink-4(m=4) 두 설정으로 분리.
METHODS_FULL = [
    ("fullkv", "FullKV", {}),
    ("streaming", "StreamingLLM", {}),
    ("h2o", "H2O", {}),
    ("snapkv", "SnapKV", {}),
    ("pyramidkv", "PyramidKV-adapted", {}),
    ("adakv", "AdaKV-adapted", {}),
    ("ours", "SalienceKV_wo_sink", {"sink_size": 0}),
    ("ours", "SalienceKV_Sink4", {"sink_size": 4}),
]
METHODS_CROSS = [
    ("fullkv", "FullKV", {}),
    ("adakv", "AdaKV-adapted", {}),
    ("ours", "SalienceKV_Sink4", {"sink_size": 4}),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Exp1: Main Results")
    parser.add_argument("--model", default="qwen3-4b",
                        choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--mode", default="full",
                        choices=["full", "cross"],
                        help="full=모든 베이스라인, cross=Full KV+AdaKV+Ours만")
    parser.add_argument("--budget", type=float, default=0.20,
                        help="KV 캐시 예산 비율 (기본값: 0.20 = 20%%)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="태스크별 샘플 수 제한 (None=전체)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="평가할 태스크 (기본값: 전체 7개)")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="FlashAttention2 활성화")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invert_norm", action="store_true", default=True,
                        help="key-norm 선택 방향을 low-norm 우선(Devoto et al. 방향)으로 둔다. "
                             "논문의 모든 결과가 이 설정이며 기본값이다.")
    parser.add_argument("--no_invert_norm", dest="invert_norm", action="store_false",
                        help="legacy 방향(high-norm 우선). 논문 수치를 재현하지 않는다.")
    return parser.parse_args()


def run_method_on_all_tasks(
    evaluator: EvaluatorV2,
    method_name: str,
    tasks: List[str],
    budget_ratio: float,
    num_samples: int,
    seed: int,
    preloaded_data: Dict = None,
    label: str = None,
    method_kwargs: Dict = None,
) -> Dict:
    """단일 메서드를 모든 태스크에서 평가."""
    display = label or method_name
    logger.info(f"\n{'─'*60}")
    logger.info(f"Method: {display.upper()} | Budget: {budget_ratio:.0%} | kwargs={method_kwargs}")
    logger.info(f"{'─'*60}")
    
    task_scores = {}
    task_collapse_rates = {}
    task_collapse_fracs = {}
    task_ttfts = []
    task_throughputs = []
    task_mem_reductions = []
    total_collapse_count = 0
    total_collapse_total = 0
    all_sample_records = {}
    
    for task_name in tasks:
        logger.info(f"\n  Task: {task_name}")
        
        try:
            if preloaded_data and task_name in preloaded_data:
                samples = preloaded_data[task_name]
            else:
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
            all_sample_records[task_name] = result.get("sample_records", [])
            
            logger.info(
                f"  → score={result['avg_score']:.2f}, "
                f"collapse={result['collapse_count']}/{result['collapse_total']} ({result['avg_collapse_rate_pct']:.1f}%), "
                f"mem_red={result['avg_memory_reduction_pct']:.1f}%, "
                f"ttft={result['avg_ttft_ms']:.1f}ms"
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
        "sample_records": all_sample_records,
    }


def main():
    args = parse_args()

    log_dir = "logs/v3_verified"
    results_dir = "results/v3_verified"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    timestamp = get_timestamp()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/exp1_{timestamp}.log"),
        ],
        force=True,
    )

    tasks = args.tasks if args.tasks else TASK_ORDER
    methods = METHODS_FULL if args.mode == "full" else METHODS_CROSS
    if args.invert_norm:
        methods = [(m, l, {**kw, "invert_norm": True}) for m, l, kw in methods]

    import core.results_manager as rm
    rm.RESULTS_DIR = results_dir
    logger.info(f"  Key-norm direction: {'CORRECTED (low-norm 우선)' if args.invert_norm else 'legacy (high-norm 우선)'}")
    logger.info(f"  Results dir: {rm.RESULTS_DIR} | Log dir: {log_dir}")
    
    logger.info(f"Experiment 1: Main Results")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Budget: {args.budget:.0%}")
    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  Methods: {methods}")
    logger.info(f"  Samples/task: {args.num_samples}")
    
    # 모델 로드
    logger.info(f"\nLoading model: {args.model} ...")
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model,
        device="cuda",
        use_flash_attn=args.use_flash_attn,
    )
    
    evaluator = EvaluatorV2(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        seed=args.seed,
    )
    
    # 모든 태스크 데이터 사전 로딩 (메서드마다 반복 다운로드 방지)
    logger.info("사전 데이터 로딩 중...")
    preloaded_data = {}
    for task_name in tasks:
        preloaded_data[task_name] = load_longbench_task(
            task_name, num_samples=args.num_samples, seed=args.seed
        )
        logger.info(f"  ✓ {task_name}: {len(preloaded_data[task_name])}샘플 로드 완료")
    logger.info("모든 태스크 데이터 로딩 완료!")

    # 각 메서드 평가
    all_results = []
    table_rows = []
    csv_filename = f"exp1_{args.model}_{args.mode}_{timestamp}.csv"
    json_filename = f"exp1_{args.model}_{args.mode}_{timestamp}.json"

    for method_name, label, method_kwargs in methods:
        result = run_method_on_all_tasks(
            evaluator=evaluator,
            method_name=method_name,
            tasks=tasks,
            budget_ratio=args.budget,
            num_samples=args.num_samples,
            seed=args.seed,
            preloaded_data=preloaded_data,
            label=label,
            method_kwargs=method_kwargs,
        )
        all_results.append(result)
        
        # Table 행 생성
        row = format_table_row(
            method=label,
            task_scores=result["task_scores"],
            task_order=tasks,
            avg_score=result["avg_score"],
            mem_reduction=result["avg_memory_reduction_pct"],
            ttft_ms=result["avg_ttft_ms"],
            throughput=result["avg_throughput"],
        )
        for task in tasks:
            row[f"{task}_collapse_%"] = round(result["task_collapse_rates"].get(task, 0.0), 1)
            row[f"{task}_collapse_n"] = result["task_collapse_fracs"].get(task, "N/A")
        row["Avg_Collapse_%"] = round(result["avg_collapse_rate_pct"], 1)
        row["Avg_Collapse_n"] = result["avg_collapse_frac"]
        table_rows.append(row)

        # 무인 장시간 실행 안전장치: method 하나 끝날 때마다 즉시 저장(덮어쓰기)
        save_results_csv(table_rows, csv_filename)
        json_data_partial = {
            "experiment": "exp1_main_results",
            "model": args.model,
            "mode": args.mode,
            "budget_ratio": args.budget,
            "tasks": tasks,
            "methods": methods,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "results": all_results,
            "completed_methods": len(all_results),
            "total_methods": len(methods),
        }
        save_results_json(json_data_partial, json_filename)
        logger.info(f"  [중간 저장 완료] {len(all_results)}/{len(methods)} 설정")

    # 결과 출력 (최종)
    print_result_table(
        table_rows,
        title=f"Exp1 Main Results | {args.model} | Budget={args.budget:.0%}"
    )

    logger.info("Experiment 1 completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
experiments/exp_table7_extra_signals.py
============================================
TABLE VII 절제 실험의 비대칭성 보완: V_only, P_only 2개 변형 추가.

배경(§III-B.3, 2026-08-15 논의): 기존 6개 변형(3signal_NVP, 4signal_NVPS,
wo_N, wo_V, wo_P, N_only)은 단일 신호 조건이 N_only 하나뿐이라, "N이 나쁘다"와
"V가 좋다"는 관찰이 대칭적으로 검증되지 않았다. 이 스크립트는 V_only, P_only
2개 변형을 동일 조건(sink=0, LongBench 7태스크, 30샘플/task, seed=42)으로
추가 실행하여 8개 변형의 완전한 그리드를 완성한다.

기존 exp_table7_signal_ablation.py는 건드리지 않고 별도 실행 — 결과는 동일
CSV 포맷으로 저장되어 TABLE VII(a)에 두 행만 추가하면 된다.

실행 (RunPod):
    python experiments/exp_table7_extra_signals.py --model qwen3-4b \
        --tasks narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa gov_report qmsum \
        --num_samples 30 --sink_size 0
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
from core.collapse_metrics import is_collapsed
from core.results_manager import save_results_csv, save_results_json, get_timestamp

logger = logging.getLogger(__name__)


def make_extra_variants(sink_size: int):
    """기존 6개 변형에 없던 단일신호 조건 2개. exp_table7_signal_ablation.py의
    make_variants()와 동일한 키(use_attention=N, use_entropy=V, use_position=P,
    use_semantic=S) 규약을 따른다."""
    return [
        ("V_only", {"use_attention": False, "use_entropy": True, "use_position": False, "use_semantic": False, "sink_size": sink_size}),
        ("P_only", {"use_attention": False, "use_entropy": False, "use_position": True, "use_semantic": False, "sink_size": sink_size}),
    ]


def run_variant_on_all_tasks(evaluator, tasks: List[str], budget_ratio: float,
                              num_samples: int, seed: int, label: str,
                              method_kwargs: Dict) -> Dict:
    logger.info(f"\n{'─'*60}")
    logger.info(f"Variant: {label} | Budget: {budget_ratio:.0%} | kwargs={method_kwargs}")
    logger.info(f"{'─'*60}")

    task_scores = {}
    task_collapse_pct = {}
    all_scores = []
    all_collapse = []
    all_sample_records = {}

    for task_name in tasks:
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            result = evaluator.evaluate_task(
                samples=samples, method_name="ours", budget_ratio=budget_ratio,
                method_kwargs=method_kwargs,
            )
            task_scores[task_name] = result["avg_score"]
            task_collapse_pct[task_name] = result["avg_collapse_rate_pct"]
            all_scores.append(result["avg_score"])
            all_collapse.append(result["avg_collapse_rate_pct"])
            all_sample_records[task_name] = result.get("sample_records", [])
            logger.info(f"  {task_name}: score={result['avg_score']:.2f}, "
                        f"collapse={result['avg_collapse_rate_pct']:.1f}%")
        except Exception as e:
            logger.error(f"  {task_name} failed: {e}")

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    avg_collapse = sum(all_collapse) / len(all_collapse) if all_collapse else 0.0
    return {
        "variant": label, "avg_score": avg_score, "avg_collapse_pct": avg_collapse,
        "task_scores": task_scores, "task_collapse_pct": task_collapse_pct,
        "sample_records": all_sample_records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="TABLE VII 보완: V_only/P_only")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--tasks", nargs="+",
                         default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
                                  "2wikimqa", "gov_report", "qmsum"])
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--budget", type=float, default=0.20)
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
            logging.FileHandler(f"{log_dir}/exp7_extra_{timestamp}.log"),
        ],
        force=True,
    )

    logger.info("=" * 60)
    logger.info("TABLE VII 보완: V_only, P_only 절제 실험")
    logger.info(f"  Model: {args.model} | Sink: {args.sink_size} | Budget: {args.budget:.0%}")
    logger.info(f"  Tasks: {args.tasks} | Samples: {args.num_samples}")
    logger.info("=" * 60)

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    variants = make_extra_variants(args.sink_size)
    if args.invert_norm:
        variants = [(name, {**kw, "invert_norm": True}) for name, kw in variants]

    import core.results_manager as rm
    rm.RESULTS_DIR = "results/v3_verified"
    os.makedirs(rm.RESULTS_DIR, exist_ok=True)
    logger.info(f"  Key-norm direction: {'CORRECTED' if args.invert_norm else 'legacy'} | Results dir: {rm.RESULTS_DIR} | Log dir: {log_dir}")

    table_rows = []
    all_results = []

    csv_filename = f"exp7_extra_signals_{args.model}_{timestamp}.csv"
    json_filename = f"exp7_extra_signals_{args.model}_{timestamp}.json"

    for label, kwargs in variants:
        r = run_variant_on_all_tasks(evaluator, args.tasks, args.budget,
                                      args.num_samples, args.seed, label, kwargs)
        all_results.append(r)
        table_rows.append({
            "Variant": r["variant"],
            "Avg_Score_7tasks": round(r["avg_score"], 2),
            "Avg_Collapse_pct": round(r["avg_collapse_pct"], 1),
        })
        save_results_csv(table_rows, csv_filename)
        save_results_json({"results": all_results, "tasks": args.tasks,
                            "num_samples": args.num_samples, "sink_size": args.sink_size},
                           json_filename)
        logger.info(f"  [중간 저장 완료] {label}")

    logger.info("\n" + "=" * 60)
    logger.info("완료. 기존 TABLE VII(a) 6개 변형과 비교:")
    logger.info("  참고 - 기존: 3signal_NVP=13.67, 4signal_NVPS=12.50, wo_N=16.52,")
    logger.info("                wo_V=12.77, wo_P=16.11, N_only=12.39")
    for r in all_results:
        logger.info(f"  {r['variant']}: score={r['avg_score']:.2f}, collapse={r['avg_collapse_pct']:.1f}%")
    logger.info(f"CSV: results/v2_verified/{csv_filename}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

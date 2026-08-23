#!/usr/bin/env python3
"""
experiments/exp_keynorm_reversal_check.py
============================================
키노름(N) 기여 방향 역전 가설 검증 (2026-08-15 세션인계 §2 참고)

가설: 버그 수정 전(b511f9a, 864836f 이전) model.generate()가 압축캐시를 붙인 채
원본 전체 input_ids를 다시 넣어(이중 forward) 원본 내용이 계속 누출됨.
이 누출이 어떤 신호로 토큰을 선택했는지와 무관하게 스코어에 반영되어, 신호 간
실질적 품질 차이가 희석(band 폭이 좁아짐)되었을 가능성.

검증 방법: 동일 조건(sink=0, 6개 신호 조합, 소규모 샘플)을
  (A) 현재(수정된) EvaluatorV2
  (B) EvaluatorV2BuggyRepro (구버전 generate() 경로만 재현)
양쪽으로 돌려서 점수 폭(최댓값-최솟값)을 비교.
가설이 맞다면: (B)의 점수 폭이 (A)보다 뚜렷하게 좁아야 함.

주의: EvaluatorV2BuggyRepro는 이 검증 전용 임시 클래스. 논문 결과에 사용 금지.

실행 (경량 점검 권장 - 7태스크 × 3~5샘플):
    python experiments/exp_keynorm_reversal_check.py --model qwen3-4b \
        --tasks narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa gov_report qmsum \
        --num_samples 3 --mode both
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
from core.evaluator_v2_buggy_repro import EvaluatorV2BuggyRepro
from core.results_manager import save_results_csv, save_results_json, get_timestamp

os.makedirs("logs/v2_verified", exist_ok=True)
os.makedirs("results/v2_verified", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/v2_verified/exp_keynorm_check_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)


def make_variants(sink_size: int):
    """신호 조합 6개 (exp_table7_signal_ablation.py와 동일 정의, sink_size=0 고정 사용)."""
    return [
        ("3signal_NVP", {"use_attention": True, "use_entropy": True, "use_position": True, "use_semantic": False, "sink_size": sink_size}),
        ("4signal_NVPS", {"use_attention": True, "use_entropy": True, "use_position": True, "use_semantic": True, "sink_size": sink_size}),
        ("wo_N", {"use_attention": False, "use_entropy": True, "use_position": True, "use_semantic": False, "sink_size": sink_size}),
        ("wo_V", {"use_attention": True, "use_entropy": False, "use_position": True, "use_semantic": False, "sink_size": sink_size}),
        ("wo_P", {"use_attention": True, "use_entropy": True, "use_position": False, "use_semantic": False, "sink_size": sink_size}),
        ("N_only", {"use_attention": True, "use_entropy": False, "use_position": False, "use_semantic": False, "sink_size": sink_size}),
    ]


def run_variant_avg_score(evaluator, tasks: List[str], num_samples: int, seed: int,
                           label: str, method_kwargs: Dict) -> Dict:
    logger.info(f"  Variant: {label} | kwargs={method_kwargs}")
    task_scores = []
    for task_name in tasks:
        try:
            samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
            result = evaluator.evaluate_task(
                samples=samples, method_name="ours", budget_ratio=0.20,
                method_kwargs=method_kwargs,
            )
            task_scores.append(result["avg_score"])
            logger.info(f"    {task_name}: {result['avg_score']:.2f}")
        except Exception as e:
            logger.error(f"    {task_name} failed: {e}")
    avg = sum(task_scores) / len(task_scores) if task_scores else 0.0
    return {"variant": label, "avg_score": avg, "task_scores": task_scores}


def run_pass(evaluator, evaluator_label: str, tasks, num_samples, seed) -> List[Dict]:
    logger.info(f"\n{'='*60}\n{evaluator_label} 실행\n{'='*60}")
    variants = make_variants(sink_size=0)
    rows = []
    for label, kwargs in variants:
        r = run_variant_avg_score(evaluator, tasks, num_samples, seed, label, kwargs)
        rows.append({"evaluator": evaluator_label, "variant": r["variant"], "avg_score": round(r["avg_score"], 2)})
    scores = [r["avg_score"] for r in rows]
    band = max(scores) - min(scores) if scores else 0.0
    logger.info(f"\n[{evaluator_label}] 점수 폭(max-min) = {band:.2f}")
    for r in rows:
        r["band_this_pass"] = round(band, 2)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="키노름 반전 가설 검증")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--tasks", nargs="+",
                         default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
                                  "2wikimqa", "gov_report", "qmsum"])
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["fixed_only", "buggy_only", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = get_timestamp()

    logger.info("=" * 60)
    logger.info("키노름 반전 가설 검증")
    logger.info(f"  Model: {args.model} | Tasks: {args.tasks} | Samples: {args.num_samples} | Mode: {args.mode}")
    logger.info("=" * 60)

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)

    all_rows = []

    if args.mode in ("fixed_only", "both"):
        evaluator_fixed = EvaluatorV2(model, tokenizer, model_config)
        all_rows += run_pass(evaluator_fixed, "FIXED(현재,수정후)", args.tasks, args.num_samples, args.seed)
        del evaluator_fixed

    if args.mode in ("buggy_only", "both"):
        evaluator_buggy = EvaluatorV2BuggyRepro(model, tokenizer, model_config)
        all_rows += run_pass(evaluator_buggy, "BUGGY(구버전재현)", args.tasks, args.num_samples, args.seed)
        del evaluator_buggy

    csv_filename = f"exp_keynorm_reversal_check_{args.model}_{timestamp}.csv"
    json_filename = f"exp_keynorm_reversal_check_{args.model}_{timestamp}.json"
    save_results_csv(all_rows, csv_filename)
    save_results_json({"rows": all_rows, "tasks": args.tasks, "num_samples": args.num_samples}, json_filename)

    logger.info("\n" + "=" * 60)
    logger.info("완료. 결과:")
    for r in all_rows:
        logger.info(f"  {r}")
    logger.info(f"CSV: results/v2_verified/{csv_filename}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

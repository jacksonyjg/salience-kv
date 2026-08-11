#!/usr/bin/env python3
"""
run_all_experiments.py
=======================
전체 실험 순차 실행 마스터 스크립트.

실험계획서 §8 '실험 실행 순서'에 따라 필수 실험 → 권장 실험 순서로 실행.

실행:
    # 전체 실험 (전체 샘플)
    python run_all_experiments.py --model qwen3-4b

    # 빠른 검증 (샘플 수 제한)
    python run_all_experiments.py --model qwen3-4b --quick --num_samples 20

    # 특정 실험만
    python run_all_experiments.py --model qwen3-4b --experiments 1a 2 3
"""

import sys
import os
import argparse
import subprocess
import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MASTER] %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], desc: str) -> bool:
    """서브프로세스로 실험 실행."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting: {desc}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        logger.info(f"✓ Completed: {desc}")
        return True
    else:
        logger.error(f"✗ Failed: {desc} (returncode={result.returncode})")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run All Experiments")
    parser.add_argument("--model", default="qwen3-4b",
                        choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--quick", action="store_true",
                        help="빠른 모드: 샘플 수 제한")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="태스크별 샘플 수 (--quick 시 기본 20)")
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="실행할 실험 ID: sanity 1a 1b 1c 2 3 4 5 6")
    parser.add_argument("--skip_sanity", action="store_true",
                        help="Sanity check 건너뜀")
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    # 샘플 수 결정
    if args.quick and args.num_samples is None:
        num_samples = 20
    else:
        num_samples = args.num_samples
    
    ns_flag = ["--num_samples", str(num_samples)] if num_samples else []
    budget_flag = ["--budget", str(args.budget)]
    model_flag = ["--model", args.model]
    
    # 실행할 실험 목록
    all_exps = ["sanity", "1a", "1b", "1c", "2", "3", "4", "5", "6"]
    run_exps = args.experiments if args.experiments else all_exps
    
    if args.skip_sanity and "sanity" in run_exps:
        run_exps.remove("sanity")
    
    logger.info("=" * 70)
    logger.info("KV Cache Experiment Master Runner")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Budget: {args.budget:.0%}")
    logger.info(f"  Samples/task: {num_samples}")
    logger.info(f"  Experiments: {run_exps}")
    logger.info("=" * 70)
    
    results = {}
    
    # ── Sanity Check ──────────────────────────────
    if "sanity" in run_exps:
        ok = run_command(
            [sys.executable, "experiments/sanity_check.py"] + model_flag,
            "Sanity Check"
        )
        results["sanity"] = ok
        if not ok:
            logger.error("Sanity check failed! Fix issues before running experiments.")
            sys.exit(1)
    
    # ── 실험 1-A: Qwen3-4B 전체 베이스라인 ──────────
    if "1a" in run_exps:
        ok = run_command(
            [sys.executable, "experiments/exp1_main_results.py"]
            + model_flag + budget_flag + ns_flag + ["--mode", "full"],
            "Exp 1-A: Main Results (Full Baselines)"
        )
        results["1a"] = ok
    
    # ── 실험 1-B: Phi-3-mini 교차 검증 ──────────
    if "1b" in run_exps:
        ok = run_command(
            [sys.executable, "experiments/exp1_main_results.py",
             "--model", "phi-3-mini"] + budget_flag + ns_flag + ["--mode", "cross"],
            "Exp 1-B: Cross-Architecture (Phi-3-mini)"
        )
        results["1b"] = ok
    
    # ── 실험 1-C: Gemma-2-2B 교차 검증 ──────────
    if "1c" in run_exps:
        ok = run_command(
            [sys.executable, "experiments/exp1_main_results.py",
             "--model", "gemma-2-2b"] + budget_flag + ns_flag + ["--mode", "cross"],
            "Exp 1-C: Cross-Architecture (Gemma-2-2B)"
        )
        results["1c"] = ok
    
    # ── 실험 2: Ablation - Score Components ──────────
    if "2" in run_exps:
        ok = run_command(
            [sys.executable, "experiments/exp2_ablation_score.py"]
            + model_flag + budget_flag + ns_flag,
            "Exp 2: Ablation - Hybrid Score Components"
        )
        results["2"] = ok
    
    # ── 실험 3: Ablation - Allocation Strategy ──────────
    # [2026-08-11] Exp3는 legacy/로 격리됨 - HF generate() API가 레이어마다
    # 다른 캐시 길이를 지원하지 않아 이 실험(H2 가설: 레이어별 동적 예산
    # "크기" 할당) 자체가 이 파이프라인 구조에서 의미 있게 테스트 불가능하다고
    # 결론 (AdaKV/PyramidKV의 레이어 적응형 설계 무력화와 동일한 아키텍처 제약).
    # 논문 V5에 이미 이 한계가 기록됨. 실행하지 않음.
    if "3" in run_exps:
        logger.warning(
            "Exp 3는 실행하지 않습니다 - legacy/exp3_ablation_allocation.py 참고 "
            "(아키텍처 제약으로 테스트 불가 결론, legacy/README.md 참고)"
        )
        results["3"] = None
    
    # ── 실험 4: Budget Sensitivity ──────────
    if "4" in run_exps:
        ok = run_command(
            [sys.executable, "experiments/exp4_budget_sensitivity.py"]
            + model_flag + ns_flag,
            "Exp 4: Budget Sensitivity Analysis"
        )
        results["4"] = ok
    
    # ── 실험 5: Hyperparameter Sensitivity ──────────
    if "5" in run_exps:
        ok = run_command(
            [sys.executable, "experiments/exp5_hyperparam_sensitivity.py"]
            + model_flag + budget_flag + ns_flag + ["--mode", "all"],
            "Exp 5: Hyperparameter Sensitivity"
        )
        results["5"] = ok
    
    # ── 실험 6: Overhead Analysis ──────────
    if "6" in run_exps:
        ok = run_command(
            [sys.executable, "experiments/exp6_overhead.py"]
            + model_flag + budget_flag,
            "Exp 6: Computational Overhead"
        )
        results["6"] = ok
    
    # 최종 요약
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    for exp_id, ok in results.items():
        status = "✓ PASSED" if ok else "✗ FAILED"
        logger.info(f"  Exp {exp_id}: {status}")
    
    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.warning(f"\nFailed experiments: {failed}")
        logger.info("Check logs in results/ directory for details.")
    else:
        logger.info("\n✅ All experiments completed successfully!")
        logger.info("Results saved to: results/")


if __name__ == "__main__":
    main()

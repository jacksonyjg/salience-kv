"""
experiments/exp_table10_cross_arch_sink.py
========================================
TABLE VIII. Cross-Architecture Compatibility — Phi-3-mini에서 sink 앵커링 효과 재현 검증.
Qwen3-4B에서 확인된 "sink 보존이 collapse를 크게 낮춘다"는 발견이 다른 아키텍처
(다른 챗 템플릿, 다른 EOS 구조)에서도 재현되는지 확인하는 축소판 실험.
Gemma-2-2B는 제외(SWA/HybridCache 이슈로 별도 확정, key-norm과 무관, §18 참고).

evaluator_v2.py의 multi-EOS 수정(2026-08-22)이 이미 core에 반영돼 있어 별도 처리 불필요.
"""
import sys, os, argparse, logging
sys.path.insert(0, os.getcwd())

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed, word_repetition_ratio, char_repetition_ratio
from core.results_manager import save_results_csv, save_results_json, get_timestamp

MODEL_KEY = "phi-3-mini"
TASKS = ["qmsum", "gov_report"]
BUDGET_RATIO = 0.20

METHODS = [
    ("fullkv", "FullKV", {}),
    ("streamingllm", "StreamingLLM", {}),
    ("h2o", "H2O(sink=0)", {"sink_size": 0}),
    ("h2o", "H2O(sink=4)", {"sink_size": 4}),
    ("snapkv", "SnapKV", {}),
    ("pyramidkv", "PyramidKV-adapted", {}),
    ("ours", "SalienceKV(sink=0)", {"sink_size": 0}),
    ("ours", "SalienceKV(sink=4)", {"sink_size": 4}),
]

logger = logging.getLogger(__name__)


def run_method(evaluator, method_name, label, base_kwargs, tasks, num_samples, seed, invert_norm):
    kwargs = {**base_kwargs, "invert_norm": invert_norm}
    logger.info(f"\n{'='*60}")
    logger.info(f"Method: {label} | kwargs={kwargs}")
    logger.info(f"{'='*60}")

    task_scores = {}
    all_sample_records = {}
    for task_name in tasks:
        samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
        scores = []
        sample_records = []
        for i, sample in enumerate(samples):
            r = evaluator.evaluate_sample(sample, method_name, BUDGET_RATIO,
                                           measure_efficiency=False, method_kwargs=kwargs)
            scores.append(r["score"])
            pred = r["prediction"]
            sample_records.append({
                "sample_idx": i, "score": r["score"], "prediction": pred,
                "word_rep": word_repetition_ratio(pred), "char_rep": char_repetition_ratio(pred),
                "collapsed": is_collapsed(pred),
            })
        avg = sum(scores) / len(scores) if scores else 0.0
        task_scores[task_name] = round(avg, 2)
        all_sample_records[task_name] = sample_records
        n_collapsed = sum(1 for r in sample_records if r["collapsed"])
        logger.info(f"  → {task_name}: avg={avg:.2f}  collapse={n_collapsed}/{len(sample_records)}")

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    all_c = [r["collapsed"] for recs in all_sample_records.values() for r in recs]
    avg_collapse = 100 * sum(all_c) / len(all_c) if all_c else 0.0

    return {
        "method": label, "task_scores": task_scores, "avg_score": round(avg_score, 2),
        "avg_collapse_pct": round(avg_collapse, 2), "sample_records": all_sample_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invert_norm", action="store_true",
                        help="key-norm 선택 방향을 corrected(low-norm 우선)로 전환.")
    args = parser.parse_args()

    log_dir = "logs/v3_verified"
    results_dir = "results/v3_verified"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_dir}/exp10_crossarch_{get_timestamp()}.log")],
        force=True,
    )

    import core.results_manager as rm
    rm.RESULTS_DIR = results_dir

    logger.info(f"TABLE VIII: Cross-Architecture Sink Anchoring (Phi-3-mini)")
    logger.info(f"  Model: {MODEL_KEY} | Tasks: {TASKS} | Samples: {args.num_samples}/task | "
                f"Budget: {BUDGET_RATIO:.0%} | invert_norm: {args.invert_norm}")

    model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    timestamp = get_timestamp()
    for method_name, label, base_kwargs in METHODS:
        result = run_method(evaluator, method_name, label, base_kwargs, TASKS,
                             args.num_samples, args.seed, args.invert_norm)
        all_results.append(result)

        json_data = {
            "experiment": "exp10_cross_arch_sink", "model": MODEL_KEY, "tasks": TASKS,
            "num_samples": args.num_samples, "budget_ratio": BUDGET_RATIO,
            "invert_norm": args.invert_norm, "results": all_results,
        }
        save_results_json(json_data, f"exp10_crossarch_phi3_{timestamp}.json")
        logger.info(f"[중간 저장 완료] {len(all_results)}/{len(METHODS)} 방법")

    save_results_csv(all_results, f"exp10_crossarch_phi3_{timestamp}.csv")
    logger.info(f"\n완료: {len(all_results)}개 방법")
    for r in all_results:
        logger.info(f"  {r['method']:<24} avg_score={r['avg_score']:6.2f}  collapse={r['avg_collapse_pct']:5.1f}%")


if __name__ == "__main__":
    main()

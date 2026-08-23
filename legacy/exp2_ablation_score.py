#!/usr/bin/env python3
"""
experiments/exp2_ablation_score.py
Exp2: Ablation - Hybrid Score 구성 요소 기여도 (evaluator_v2 기반)
"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.results_manager import save_results_csv, save_results_json, get_timestamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"results/exp2_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TASK_ORDER = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "qmsum"]

ABLATION_CONFIGS = [
    {
        "name": "Full (All 4 signals)",
        "use_attention": True, "use_entropy": True, "use_semantic": True, "use_position": True,
        "alpha": 0.40, "beta": 0.20, "gamma": 0.20, "delta": 0.20,
    },
    {
        "name": "w/o Attention",
        "use_attention": False, "use_entropy": True, "use_semantic": True, "use_position": True,
        "alpha": 0.0, "beta": 0.33, "gamma": 0.33, "delta": 0.34,
    },
    {
        "name": "w/o Semantic",
        "use_attention": True, "use_entropy": True, "use_semantic": False, "use_position": True,
        "alpha": 0.50, "beta": 0.25, "gamma": 0.0, "delta": 0.25,
    },
    {
        "name": "w/o Entropy",
        "use_attention": True, "use_entropy": False, "use_semantic": True, "use_position": True,
        "alpha": 0.50, "beta": 0.0, "gamma": 0.25, "delta": 0.25,
    },
    {
        "name": "w/o Position",
        "use_attention": True, "use_entropy": True, "use_semantic": True, "use_position": False,
        "alpha": 0.50, "beta": 0.25, "gamma": 0.25, "delta": 0.0,
    },
]

def parse_args():
    parser = argparse.ArgumentParser(description="Exp2: Ablation - Score Components")
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
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

    model, tokenizer, model_config = load_model_and_tokenizer(args.model, device="cuda")
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    table_rows = []
    baseline_avg = None

    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"Condition: {name}")

        method_kwargs = {
            "alpha":        cfg.get("alpha", 0.40),
            "beta":         cfg.get("beta",  0.20),
            "gamma":        cfg.get("gamma", 0.20),
            "delta":        cfg.get("delta", 0.20),
            "use_attention": cfg["use_attention"],
            "use_entropy":   cfg["use_entropy"],
            "use_semantic":  cfg["use_semantic"],
            "use_position":  cfg["use_position"],
        }

        task_scores = {}
        for task_name in tasks:
            try:
                samples = load_longbench_task(task_name, num_samples=args.num_samples, seed=args.seed)
                scores = []
                for i, sample in enumerate(samples):
                    r = evaluator.evaluate_sample(sample, "ours", args.budget, method_kwargs=method_kwargs)
                    scores.append(r["score"])
                    if (i+1) % 5 == 0:
                        logger.info(f"  [{i+1}/{len(samples)}] {task_name} avg={sum(scores)/len(scores):.2f}")
                avg = sum(scores) / len(scores) if scores else 0.0
                task_scores[task_name] = round(avg, 1)
                logger.info(f"  {task_name}: {avg:.2f}")
            except Exception as e:
                logger.error(f"  {task_name} failed: {e}", exc_info=True)
                task_scores[task_name] = 0.0

        avg_score = sum(task_scores[t] for t in tasks) / len(tasks)
        if baseline_avg is None:
            baseline_avg = avg_score
        delta = avg_score - baseline_avg

        row = {"Condition": name}
        for t in tasks:
            row[t] = task_scores.get(t, 0.0)
        row["Avg"] = round(avg_score, 1)
        row["Δ vs Full"] = round(delta, 1)
        table_rows.append(row)
        all_results.append({"condition": name, "task_scores": task_scores, "avg": avg_score, "delta": delta})
        logger.info(f"  → Avg={avg_score:.2f}, Δ={delta:+.2f}")

    # 출력
    cols = ["Condition"] + tasks + ["Avg", "Δ vs Full"]
    header = " | ".join(f"{c:<20}" if c == "Condition" else f"{c:<15}" for c in cols)
    sep = "-" * len(header)
    print("\n" + "="*80)
    print(f"  Exp2 Ablation - Score Components | {args.model}")
    print("="*80)
    print(header)
    print(sep)
    for row in table_rows:
        line = f"{row['Condition']:<20} | " + " | ".join(f"{row.get(c, 0.0):<15}" for c in tasks) + f" | {row['Avg']:<8} | {row['Δ vs Full']}"
        print(line)
    print("="*80)

    csv_file = f"exp2_score_ablation_{args.model}_{timestamp}.csv"
    json_file = f"exp2_score_ablation_{args.model}_{timestamp}.json"
    save_results_csv(table_rows, csv_file)
    save_results_json({"model": args.model, "budget": args.budget, "results": all_results}, json_file)
    logger.info(f"Experiment 2 completed!")

if __name__ == "__main__":
    main()

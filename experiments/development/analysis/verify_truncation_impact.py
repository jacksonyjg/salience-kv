import sys, os, argparse, json
sys.path.insert(0, os.getcwd())

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2

TARGET_INDICES = {
    "gov_report": [2, 3, 9],
    "qmsum": [21],
}

METHODS = [
    ("fullkv", "FullKV", {}),
    ("ours", "SalienceKV_sink0", {"sink_size": 0}),
    ("ours", "SalienceKV_sink4", {"sink_size": 4}),
]

BUDGET_RATIO = 0.20

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=["CAPPED", "EXPANDED"])
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--max_input_length", type=int, default=16000,
                        help="16000 for CAPPED, 31768 for EXPANDED")
    args = parser.parse_args()

    print(f"=== Truncation 영향 검증 [{args.label}] ===")
    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    evaluator = EvaluatorV2(model, tokenizer, model_config)

    all_results = []
    for task_name, indices in TARGET_INDICES.items():
        full_samples = load_longbench_task(task_name, num_samples=30, seed=42)
        target_samples = [full_samples[i] for i in indices]
        print(f"\n--- Task: {task_name}, 대상 idx: {indices} ---")

        for method_name, label, method_kwargs in METHODS:
            result = evaluator.evaluate_task(
                samples=target_samples,
                method_name=method_name,
                budget_ratio=BUDGET_RATIO,
                method_kwargs=method_kwargs,
                max_input_length=args.max_input_length,
            )
            print(f"[{label}] task={task_name} score={result['avg_score']:.2f} "
                  f"collapse={result['collapse_count']}/{result['collapse_total']}")
            all_results.append({
                "cap_label": args.label,
                "task": task_name,
                "indices": indices,
                "method": label,
                "avg_score": result["avg_score"],
                "collapse_count": result["collapse_count"],
                "collapse_total": result["collapse_total"],
            })

    out_path = f"results/final/truncation_verify_{args.label}.json"
    os.makedirs("results/final", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n저장됨: {out_path}")

if __name__ == "__main__":
    main()

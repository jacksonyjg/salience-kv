import sys, os
sys.path.insert(0, os.getcwd())

from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2

MODELS = ["phi-3-mini", "gemma-2-2b"]
TASK = "qmsum"
NUM_SAMPLES = 2
BUDGET_RATIO = 0.20

def check_model(model_key):
    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"{'='*60}")

    try:
        model, tokenizer, model_config = load_model_and_tokenizer(model_key)
        evaluator = EvaluatorV2(model, tokenizer, model_config)
        print(f"  모델 로드 성공")
    except Exception as e:
        print(f"  [실패] 모델 로드: {e}")
        return

    samples = load_longbench_task(TASK, num_samples=NUM_SAMPLES, seed=42)

    for method_name, label in [("fullkv", "FullKV"), ("adakv", "AdaKV")]:
        print(f"\n  --- {label} ---")
        for i, sample in enumerate(samples):
            try:
                r = evaluator.evaluate_sample(
                    sample, method_name, BUDGET_RATIO, measure_efficiency=False,
                )
                pred_preview = r["prediction"][:150].replace("\n", " ")
                print(f"  [{i}] score={r['score']:.2f} pred={pred_preview!r}")
            except Exception as e:
                print(f"  [{i}] [예외 발생] {e}")

    del model
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for model_key in MODELS:
        check_model(model_key)

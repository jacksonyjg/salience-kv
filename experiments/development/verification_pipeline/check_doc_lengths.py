import sys
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer, make_prompt
from core.dataset_loader import load_longbench_task

_, tok, _ = load_model_and_tokenizer("qwen3-4b")
for task in ["gov_report", "qmsum", "narrativeqa"]:
    samples = load_longbench_task(task, num_samples=15, seed=42)
    lens = []
    for s in samples:
        p = make_prompt(model_key="qwen3-4b", tokenizer=tok, context=s["context"],
                         question=s["question"], task_type=s["task_type"])
        ids = tok(p, add_special_tokens=False)["input_ids"]
        lens.append(len(ids))
    lens.sort()
    print(f"{task}: min={lens[0]} max={lens[-1]} median={lens[len(lens)//2]}")
    print(f"  전체: {lens}")

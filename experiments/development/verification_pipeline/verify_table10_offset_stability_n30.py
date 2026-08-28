import sys
sys.path.insert(0, "/workspace/kv-cache-exp")
from transformers import AutoTokenizer
from core.model_loader import make_prompt
from core.dataset_loader import load_longbench_task

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

marker_context = "\uE000\uE000\uE000\uE000 MARKERMARKERMARKER \uE000\uE000\uE000\uE000"


def offset_for_sample(real_context, question, task_type="summarization"):
    p1 = make_prompt("qwen3-4b", tokenizer, real_context, question, task_type)
    p2 = make_prompt("qwen3-4b", tokenizer, marker_context, question, task_type)
    ids1 = tokenizer(p1, add_special_tokens=False)["input_ids"]
    ids2 = tokenizer(p2, add_special_tokens=False)["input_ids"]
    offset = 0
    for a, b in zip(ids1, ids2):
        if a != b:
            break
        offset += 1
    return offset


for task in ["gov_report", "qmsum"]:
    samples = load_longbench_task(task, num_samples=30, seed=42)
    offsets = []
    for s in samples:
        off = offset_for_sample(s["context"], s["question"])
        offsets.append(off)
    unique = sorted(set(offsets))
    print(f"=== {task} (N=30) ===")
    print(f"offset 분포: {offsets}")
    print(f"고유값: {unique}")
    print(f"mismatch: {'없음' if len(unique) == 1 else f'있음! {len(unique)}개 서로 다른 값'}")
    print()

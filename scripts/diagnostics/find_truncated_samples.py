import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from core.dataset_loader import load_longbench_task
from core.model_loader import make_prompt, MODEL_CONFIGS

MODEL_KEY = "qwen3-4b"
TARGET_LENGTHS = {16224, 16844, 16924, 18386, 18553, 19434, 25564, 28135, 41630, 51949}
TOLERANCE = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[MODEL_KEY]["hf_name"])

for task in ["qmsum", "gov_report"]:
    samples = load_longbench_task(task, num_samples=30, seed=42)
    for idx, sample in enumerate(samples):
        prompt = make_prompt(
            MODEL_KEY, tokenizer,
            context=sample.get("context", sample.get("input", "")),
            question=sample.get("question", ""),
            task_type="summarization",
        )
        encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
        seq_len = len(encoded["input_ids"])
        for target in TARGET_LENGTHS:
            if abs(seq_len - target) <= TOLERANCE:
                loss_pct = max(0, (seq_len - 16000) / seq_len * 100)
                print(f"[MATCH] task={task} idx={idx} seq_len={seq_len} loss={loss_pct:.1f}%")

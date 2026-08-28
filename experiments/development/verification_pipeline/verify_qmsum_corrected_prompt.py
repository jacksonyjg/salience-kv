"""
QMSum legacy prompt vs LongBench 공식 형식 prompt 비교.
core/model_loader.py는 아직 건드리지 않음 - 여기서 별도로 두 프롬프트를 만들어 비교.
GPU/모델 로드 불필요 - AutoTokenizer만 사용.
"""
import sys
sys.path.insert(0, "/workspace/kv-cache-exp")
from transformers import AutoTokenizer
from core.model_loader import make_prompt, MODEL_CONFIGS
from core.dataset_loader import load_longbench_task

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


def make_prompt_corrected_qmsum(tokenizer, context, question):
    """LongBench 공식 dataset2prompt.json의 QMSum 형식에 맞춘 corrected 버전.
    (GPT 제안, THUDM/LongBench/config/dataset2prompt.json 참고)"""
    user_content = (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        f"Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        f"Query: {question}\n"
        "Answer:"
    )
    # Qwen3 챗 템플릿 적용 (기존 make_prompt의 qwen3_chat 분기와 동일하게)
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)


samples = load_longbench_task("qmsum", num_samples=3, seed=42)

for i, s in enumerate(samples):
    print(f"{'='*70}\nSAMPLE {i}\n{'='*70}")
    print(f"QUERY: {s['question'][:150]!r}\n")

    legacy_prompt = make_prompt(model_key="qwen3-4b", tokenizer=tokenizer, context=s["context"],
                                 question=s["question"], task_type=s["task_type"])
    corrected_prompt = make_prompt_corrected_qmsum(tokenizer, s["context"], s["question"])

    print(f"--- LEGACY (현재 core/model_loader.py, 미수정) ---")
    print(f"QUERY IN PROMPT: {s['question'] in legacy_prompt}")
    print(f"앞부분: {legacy_prompt[:200]!r}")

    print(f"\n--- CORRECTED (LongBench 공식 형식) ---")
    print(f"QUERY IN PROMPT: {s['question'] in corrected_prompt}")
    print(f"앞부분: {corrected_prompt[:200]!r}")
    print(f"뒷부분(query가 Answer: 직전에 있는지): ...{corrected_prompt[-250:]!r}")
    print()

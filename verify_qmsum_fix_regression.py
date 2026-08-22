"""
core/model_loader.py의 QMSum 공식 템플릿 정식 반영(qmsum_official_template_freeze.diff) 회귀 검증.
GPU/모델 로드 불필요 - AutoTokenizer만 사용. core.make_prompt를 직접 그대로 사용(재구현 안 함).
"""
import sys
sys.path.insert(0, "/workspace/kv-cache-exp")
from transformers import AutoTokenizer
from core.model_loader import make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# 1) gov_report - 완전히 동일해야 함(question="" 이므로)
print("=== gov_report 회귀 확인(question='', 기존과 동일해야 함) ===")
gov_samples = load_longbench_task("gov_report", num_samples=3, seed=42)
for i, s in enumerate(gov_samples):
    assert s["question"] == "", f"gov_report[{i}]의 question이 빈 문자열이 아님: {s['question']!r}"
    prompt = make_prompt(model_key="qwen3-4b", tokenizer=tokenizer, context=s["context"],
                          question=s["question"], task_type=s["task_type"])
    expected_start = "<|im_start|>user\nPlease summarize the following document concisely.\n\nDocument:\n"
    ok = prompt.startswith(expected_start)
    print(f"  [{i}] question='' 확인, 프롬프트 시작 동일함={ok}")
    assert ok, f"gov_report[{i}] 프롬프트가 기존 형식과 다름!"
    # [2026-08-22 GPT 21차 검토 반영] QMSum 분기가 gov_report로 새어들어가지 않는지 강화 확인
    assert "Query:" not in prompt, f"gov_report[{i}]에 'Query:'가 섞여 들어감!"
    assert "\nAnswer:" not in prompt, f"gov_report[{i}]에 'Answer:'가 섞여 들어감!"
    assert "\nSummary:" in prompt, f"gov_report[{i}]에 'Summary:'가 없음!"

# 2) qmsum - 공식 LongBench 형식으로 쿼리가 포함되어야 함
print("\n=== qmsum 공식 템플릿 확인 ===")
qmsum_samples = load_longbench_task("qmsum", num_samples=3, seed=42)
for i, s in enumerate(qmsum_samples):
    prompt = make_prompt(model_key="qwen3-4b", tokenizer=tokenizer, context=s["context"],
                          question=s["question"], task_type=s["task_type"])
    in_prompt = s["question"] in prompt
    expected_marker = f"Query: {s['question']}\nAnswer:"
    marker_ok = expected_marker in prompt
    print(f"  [{i}] QUERY: {s['question'][:80]!r}")
    print(f"      QUERY IN PROMPT: {in_prompt}  |  'Query: ...\\nAnswer:' 정확한 위치: {marker_ok}")
    assert in_prompt and marker_ok, f"qmsum[{i}] 공식 템플릿이 정확히 적용 안 됨!"

    # [2026-08-22 수정] 강제 truncation(max_input_length=2000) 후에도 query 텍스트가 살아있는지
    # 확인 - 실제 평가에서는 16000 토큰 캡을 거치므로 이 문제가 재현될 가능성은 낮지만,
    # prompt integrity 문제였던 만큼 확실히 검증.
    # (주의: 토큰 ID 완전일치 비교는 서브워드 경계 문제로 오탐 가능 - 디코드 후 텍스트로 확인)
    inputs = tokenize_prompt(prompt, tokenizer, "qwen3-4b", max_input_length=2000, device="cpu")
    ids = inputs["input_ids"][0].tolist()
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    truncation_ok = s["question"] in decoded
    print(f"      강제 truncation(2000) 후 query 텍스트 생존(디코드 기준): {truncation_ok}")
    assert truncation_ok, f"qmsum[{i}] query가 truncation 후 디코드 결과에서 사라짐!"

print("\n=== 전체 검증 통과 ===")

"""
회귀 방지용 진단 스크립트 (2026-08-11, 파이프라인 정식 수정 이후 버전)
model.forward를 몽키패치해서 압축 경로 생성 중 실제로 전달되는
cache_position 값들을 수집하고, 원본 절대위치(input_length) 기준으로
정확히 이어지는지 assert로 검증한다.
"""
import sys
sys.path.insert(0, '.')
import torch
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2

MODEL_KEY = 'qwen3-4b'
METHOD = 'h2o'
BUDGET = 0.2

model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
evaluator = EvaluatorV2(model, tokenizer, model_config)
samples = load_longbench_task('qmsum', num_samples=1, seed=42)
sample = samples[0]

seen_cache_positions = []
seen_attn_lens = []

orig_forward = model.forward

def hooked_forward(*args, **kwargs):
    cp = kwargs.get('cache_position', None)
    am = kwargs.get('attention_mask', None)
    if cp is not None and kwargs.get('past_key_values', None) is not None:
        seen_cache_positions.append(cp.clone().cpu())
    if am is not None and kwargs.get('past_key_values', None) is not None:
        seen_attn_lens.append(am.shape[1])
    return orig_forward(*args, **kwargs)

model.forward = hooked_forward
try:
    result = evaluator.evaluate_sample(sample, METHOD, budget_ratio=BUDGET)
finally:
    model.forward = orig_forward

print(f"score={result['score']:.2f}  prediction={result['prediction'][:100]!r}")
print(f"수집된 decode step 수: {len(seen_cache_positions)}")

if len(seen_cache_positions) == 0:
    print("경고: forward 후킹이 압축 경로에서 아무것도 못 잡았음 - 스크립트 자체 점검 필요")
    sys.exit(1)

first_pos = seen_cache_positions[0].item()

from core.model_loader import make_prompt, tokenize_prompt
prompt = make_prompt(MODEL_KEY, tokenizer, sample['context'], sample['question'], sample['task_type'])
inputs = tokenize_prompt(prompt, tokenizer, MODEL_KEY, max_input_length=16000, device=model.device)
true_input_length = inputs['input_ids'].shape[1]

print(f"\n[검증 1] 첫 cache_position == input_length?  {first_pos} vs {true_input_length}")
assert first_pos == true_input_length, f"FAIL: 첫 cache_position({first_pos})이 input_length({true_input_length})와 다름 - 회귀 의심"

print("[검증 2] cache_position이 매 스텝 +1씩 증가하는지 확인")
positions = [p.item() for p in seen_cache_positions]
for i in range(1, len(positions)):
    assert positions[i] == positions[i-1] + 1, f"FAIL: step {i}에서 위치가 연속적이지 않음 ({positions[i-1]} -> {positions[i]})"

print("[검증 3] attention_mask 길이가 매 스텝 1씩 증가하는지 확인")
for i in range(1, len(seen_attn_lens)):
    assert seen_attn_lens[i] == seen_attn_lens[i-1] + 1, f"FAIL: step {i}에서 attention_mask 길이가 이상함"

print(f"\n===== 모든 검증 통과 =====")
print(f"cache_position 범위: {positions[0]} ~ {positions[-1]} (원본 길이 N={true_input_length} 기준으로 정확히 이어짐)")
print(f"이전 버그였다면 cache_position이 cache_len 기준(작은 값)에서 시작했어야 함 - 지금은 input_length부터 시작하므로 정상")

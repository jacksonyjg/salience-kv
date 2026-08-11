"""
TTFT/Throughput 실측 점검 (2026-08-11)
"""
import sys
sys.path.insert(0, '.')
import time
import torch
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2

MODEL_KEY = 'qwen3-4b'
TASK = 'gov_report'
NUM_SAMPLES = 3
BUDGET = 0.2
CONFIGS = [
    ('fullkv', None),
    ('h2o', 4),
    ('adakv', 4),
    ('ours', 4),
]

model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
evaluator = EvaluatorV2(model, tokenizer, model_config)
samples = load_longbench_task(TASK, num_samples=NUM_SAMPLES, seed=42)

orig_forward = model.forward
call_log = []


def hooked_forward(*args, **kwargs):
    input_ids = kwargs.get('input_ids', None)
    n = input_ids.shape[1] if input_ids is not None else -1
    t0 = time.perf_counter()
    out = orig_forward(*args, **kwargs)
    torch.cuda.synchronize()
    call_log.append((n, time.perf_counter() - t0))
    return out


print(f"{'설정':16s} {'샘플':4s} {'score':>7s} {'ttft_ms':>9s} {'thpt':>7s} "
      f"{'forward호출수':>12s} {'앞3개호출길이':>20s}")

for method, sink in CONFIGS:
    kwargs = {} if sink is None else {'sink_size': sink}
    for i, sample in enumerate(samples):
        call_log.clear()
        model.forward = hooked_forward

        try:
            r = evaluator.evaluate_sample(sample, method, budget_ratio=BUDGET, method_kwargs=kwargs)
        finally:
            model.forward = orig_forward

        num_calls = len(call_log)
        first3_lens = [c[0] for c in call_log[:3]]
        tag = f"{method}" + (f"+sink{sink}" if sink else "")

        print(f"{tag:16s} {i:4d} {r['score']:7.2f} {r['ttft_ms']:9.1f} {r['throughput']:7.1f} "
              f"{num_calls:12d} {str(first3_lens):>20s}")

print("\n해석 가이드:")
print("- forward호출수: 압축 방법은 '1(prefill) + 생성토큰수'만큼 나와야 정상")
print("- fullkv의 forward호출수 == 2(또는 그 이상)면 이중 prefill 의심")

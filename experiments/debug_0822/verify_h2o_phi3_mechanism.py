import sys
sys.path.insert(0, "/workspace/kv-cache-exp")
import torch
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
import core.kv_cache_hook as kvh

# ── 1단계: Phi-3 / Qwen3 실제 토큰 0~15 확인 ──
print("=" * 70)
print("1단계: 실제 토큰 0~15 (gov_report 샘플 하나, 두 모델 비교)")
print("=" * 70)

for model_key in ["phi-3-mini", "qwen3-4b"]:
    model, tok, cfg = load_model_and_tokenizer(model_key)
    samples = load_longbench_task("gov_report", num_samples=1, seed=42)
    sample = samples[0]
    prompt = make_prompt(model_key=model_key, tokenizer=tok, context=sample["context"],
                          question=sample["question"], task_type=sample["task_type"])
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    print(f"\n--- {model_key} ---")
    for i in range(16):
        print(f"  [{i:2d}] id={ids[i]:6d} -> {tok.decode([ids[i]])!r}")
    del model
    torch.cuda.empty_cache()

# ── 2+3단계: Phi-3 H2O(sink=0), 레이어별 position 0~15 key-norm 순위 + 보존율 ──
print("\n" + "=" * 70)
print("2+3단계: Phi-3 H2O(sink=0), 레이어별 key-norm 순위 + position 1 보존율")
print("=" * 70)

model, tok, cfg = load_model_and_tokenizer("phi-3-mini")
ev = EvaluatorV2(model, tok, cfg)

num_layers = cfg.get("num_layers") or model.config.num_hidden_layers
print(f"레이어 수: {num_layers}")

# _compress를 몽키패치해서 모든 레이어의 idx와 position 0~15 score를 캡처
captured_per_sample = {}  # sample_i -> {layer_idx: {"indices": set, "score_0_15": [...]}}
_orig_compress = kvh.H2OCache._compress

def patched_compress(self, key_states, value_states, layer_idx, budget):
    device = key_states.device
    seq_len = key_states.shape[2]
    pk = self._prefill_keys[layer_idx]
    ref_k = pk.to(device) if (pk is not None and pk.shape[2] == seq_len) else key_states
    score = kvh._key_importance(ref_k).to(device)
    if self.invert_norm:
        score = -score
    window = min(16, seq_len // 4, budget // 4) if budget > 4 else 0
    window = max(window, 0)
    indices = kvh._select_with_sink(score, seq_len, budget, window, self.sink_size, device)

    cur = captured_per_sample.setdefault(captured_per_sample["_cur_sample"], {})
    cur[layer_idx] = {
        "indices": set(indices.cpu().tolist()),
        "score_0_15": score[:16].detach().cpu().tolist() if seq_len >= 16 else None,
    }
    return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()

kvh.H2OCache._compress = patched_compress

samples = load_longbench_task("gov_report", num_samples=30, seed=42)
for i, sample in enumerate(samples):
    captured_per_sample["_cur_sample"] = i
    ev.evaluate_sample(sample, "h2o", 0.20, method_kwargs=dict(sink_size=0, invert_norm=True))

kvh.H2OCache._compress = _orig_compress

# 레이어별 position 1 보존율 집계
print(f"\n=== 레이어별 position 1 보존율 (30개 샘플 기준) ===")
n_samples = len([k for k in captured_per_sample if k != "_cur_sample"])
for layer in range(num_layers):
    count = sum(1 for i in range(n_samples) if layer in captured_per_sample.get(i, {})
                and 1 in captured_per_sample[i][layer]["indices"])
    marker = " <-- 100%" if count == n_samples else ("" if count > 0 else " (0%)")
    print(f"  Layer {layer:2d}: {count}/{n_samples}{marker}")

# position 0~15 평균 key-norm score (레이어 0 기준, 참고용)
print(f"\n=== Layer 0, position 0~15 평균 key-norm score(invert_norm 적용 후, 높을수록 우선 선택) ===")
sums = [0.0] * 16
counts = [0] * 16
for i in range(n_samples):
    s015 = captured_per_sample.get(i, {}).get(0, {}).get("score_0_15")
    if s015:
        for p in range(16):
            sums[p] += s015[p]
            counts[p] += 1
for p in range(16):
    avg = sums[p] / counts[p] if counts[p] else float("nan")
    print(f"  position {p:2d}: 평균 score={avg:8.3f}")

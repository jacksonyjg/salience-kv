"""
GPT 지적 검증: H2O-no-pos1(B, sink=0)이 이미 pos0를 자연 선택하고 있어서
H2O-no-pos1+pos0(C, sink=1)와 사실상 동일한 캐시가 되는지 확인.
Generation 없이 prefill+selection만 봄(빠름).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
import core.kv_cache_hook as kvh

model, tok, cfg = load_model_and_tokenizer("phi-3-mini")
ev = EvaluatorV2(model, tok, cfg)

_orig_compress = kvh.H2OCache._compress

def make_patched(exclude_positions, sink_size):
    def patched(self, key_states, value_states, layer_idx, budget):
        device = key_states.device
        seq_len = key_states.shape[2]
        pk = self._prefill_keys[layer_idx]
        ref_k = pk.to(device) if (pk is not None and pk.shape[2] == seq_len) else key_states
        score = kvh._key_importance(ref_k).to(device)
        if self.invert_norm:
            score = -score
        for p in exclude_positions:
            if p < seq_len:
                score[p] = float("-inf")
        window = min(16, seq_len // 4, budget // 4) if budget > 4 else 0
        window = max(window, 0)
        indices = kvh._select_with_sink(score, seq_len, budget, window, self.sink_size, device)
        captured.setdefault(captured["_key"], {})[layer_idx] = set(indices.cpu().tolist())
        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()
    return patched

captured = {}
samples = load_longbench_task("gov_report", num_samples=30, seed=42)

results_B = {}
results_C = {}

for cond_name, exclude, sink_size, store in [
    ("B", {1}, 0, results_B),
    ("C", {1}, 1, results_C),
]:
    kvh.H2OCache._compress = make_patched(exclude, sink_size)
    for i, sample in enumerate(samples):
        captured.clear()
        captured["_key"] = i
        # generation 없이 prefill+compress만 트리거하기 위해 evaluate_sample을 그대로 쓰되
        # 결과(prediction)는 버림 - _compress 몽키패치가 captured에 다 기록해줌
        ev.evaluate_sample(sample, "h2o", 0.20, method_kwargs=dict(sink_size=sink_size, invert_norm=True))
        store[i] = dict(captured.get(i, {}))
    kvh.H2OCache._compress = _orig_compress

n_samples = len(samples)
num_layers = 32

# 1) B에서 pos0 자연 선택률
b_pos0_count = 0
total = 0
for i in range(n_samples):
    for layer_idx in range(num_layers):
        if layer_idx in results_B.get(i, {}):
            total += 1
            if 0 in results_B[i][layer_idx]:
                b_pos0_count += 1
print(f"=== B(H2O-no-pos1, sink=0)에서 pos0 자연 선택률 ===")
print(f"{b_pos0_count}/{total} ({100*b_pos0_count/total:.1f}%)")

# 2) B와 C의 selected indices 완전 일치 비율
exact_match = 0
total2 = 0
sym_diff_sizes = []
for i in range(n_samples):
    for layer_idx in range(num_layers):
        if layer_idx in results_B.get(i, {}) and layer_idx in results_C.get(i, {}):
            total2 += 1
            b_set = results_B[i][layer_idx]
            c_set = results_C[i][layer_idx]
            if b_set == c_set:
                exact_match += 1
            sym_diff_sizes.append(len(b_set.symmetric_difference(c_set)))

print(f"\n=== B/C selected indices 완전 일치율 ===")
print(f"{exact_match}/{total2} ({100*exact_match/total2:.1f}%)")
print(f"평균 symmetric difference 크기: {sum(sym_diff_sizes)/len(sym_diff_sizes):.2f}")
print(f"symmetric difference 분포: min={min(sym_diff_sizes)}, max={max(sym_diff_sizes)}")

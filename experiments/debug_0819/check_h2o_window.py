import sys, json
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed
import core.kv_cache_hook as kvh

# 원본 H2OCache._compress를 저장해두고, window만 바꾼 버전으로 교체
_orig_compress = kvh.H2OCache._compress

def make_patched_compress(window_override):
    def patched(self, key_states, value_states, layer_idx, budget):
        device = key_states.device
        seq_len = key_states.shape[2]
        pk = self._prefill_keys[layer_idx]
        ref_k = pk.to(device) if (pk is not None and pk.shape[2] == seq_len) else key_states
        score = kvh._key_importance(ref_k).to(device)
        if self.invert_norm:
            score = -score
        window = min(window_override, seq_len // 4, budget // 4) if budget > 4 else 0
        window = max(window, 0)
        indices = kvh._select_with_sink(score, seq_len, budget, window, self.sink_size, device)
        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()
    return patched

model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tok, cfg)
data = load_longbench_task("gov_report", num_samples=15, seed=42)

results = {}
for label, window_val in [("window=16 (원본 재현)", 16), ("window=32 (SnapKV와 동일)", 32)]:
    kvh.H2OCache._compress = make_patched_compress(window_val)
    sc, col = [], []
    for s in data:
        r = ev.evaluate_sample(s, method_name="h2o", budget_ratio=0.2,
                                method_kwargs=dict(sink_size=0, invert_norm=True))
        sc.append(r["score"])
        col.append(is_collapsed(r["prediction"]))  # 전체 텍스트, 정식 함수
    avg = sum(sc)/len(sc)
    cr = 100*sum(col)/len(col)
    results[label] = (avg, cr)
    print(f"{label:28s} score={avg:6.2f}  collapse={cr:5.1f}%  ({sum(col)}/{len(col)})", flush=True)

kvh.H2OCache._compress = _orig_compress
print("\n결과:", results)

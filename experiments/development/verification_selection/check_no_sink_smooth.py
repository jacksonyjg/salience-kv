import sys, json
import torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed
import core.kv_cache_hook as kvh

def make_patched_compress(window_val, kernel_size, ban_positions):
    def patched(self, key_states, value_states, layer_idx, budget):
        device = key_states.device
        seq_len = key_states.shape[2]
        pk = self._prefill_keys[layer_idx]
        ref_k = pk.to(device) if (pk is not None and pk.shape[2] == seq_len) else key_states
        score = kvh._key_importance(ref_k).to(device)
        if self.invert_norm:
            score = -score
        if kernel_size > 1 and score.shape[0] > kernel_size:
            score = F.avg_pool1d(
                score.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            ).squeeze()
        # 위치 0~ban_positions-1을 선택 후보에서 강제 배제 (-inf로 마스킹)
        if ban_positions > 0:
            score = score.clone()
            score[:min(ban_positions, score.shape[0])] = float("-inf")
        window = min(window_val, seq_len // 4, budget // 4) if budget > 4 else 0
        window = max(window, 0)
        indices = kvh._select_with_sink(score, seq_len, budget, window, self.sink_size, device)
        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()
    return patched

model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tok, cfg)
data = load_longbench_task("gov_report", num_samples=15, seed=42)

for label, ban in [("smoothing=5, 위치0~3 배제 안함(재현)", 0),
                    ("smoothing=5, 위치0~3 강제 배제", 4)]:
    kvh.H2OCache._compress = make_patched_compress(window_val=16, kernel_size=5, ban_positions=ban)
    sc, col = [], []
    for s in data:
        r = ev.evaluate_sample(s, method_name="h2o", budget_ratio=0.2,
                                method_kwargs=dict(sink_size=0, invert_norm=True))
        sc.append(r["score"]); col.append(is_collapsed(r["prediction"]))
    avg = sum(sc)/len(sc); cr = 100*sum(col)/len(col)
    print(f"{label:36s} score={avg:6.2f}  collapse={cr:5.1f}%  ({sum(col)}/{len(col)})", flush=True)

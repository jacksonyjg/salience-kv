import sys, json
import torch.nn.functional as F
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed
import core.kv_cache_hook as kvh
import core.evaluator_v2 as ev2

# 1) H2O._compress를 smoothing=5로 패치 (지난번 0% collapse를 만든 그 조건)
def make_patched_compress(window_val, kernel_size):
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
        window = min(window_val, seq_len // 4, budget // 4) if budget > 4 else 0
        window = max(window, 0)
        indices = kvh._select_with_sink(score, seq_len, budget, window, self.sink_size, device)
        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()
    return patched

kvh.H2OCache._compress = make_patched_compress(window_val=16, kernel_size=5)

# 2) cache 캡처 패치
_orig_factory = kvh.make_hook_cache
captured = {}
def patched_factory(*args, **kwargs):
    cache = _orig_factory(*args, **kwargs)
    captured["cache"] = cache
    return cache
kvh.make_hook_cache = patched_factory
if hasattr(ev2, "make_hook_cache"):
    ev2.make_hook_cache = patched_factory

model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tok, cfg)
data = load_longbench_task("gov_report", num_samples=15, seed=42)

SINK_POS = {0, 1, 2, 3}
overlap_counts = []
collapse_flags = []

for idx, s in enumerate(data):
    captured.clear()
    r = ev.evaluate_sample(s, method_name="h2o", budget_ratio=0.2,
                            method_kwargs=dict(sink_size=0, invert_norm=True))
    collapsed = is_collapsed(r["prediction"])
    collapse_flags.append(collapsed)
    cache = captured.get("cache")
    if cache is not None and hasattr(cache, "_selected_positions") and cache._selected_positions[0] is not None:
        sel_set = set(cache._selected_positions[0].tolist())
        overlap = len(sel_set & SINK_POS)
        overlap_counts.append(overlap)
        print(f"sample {idx:2d}  collapsed={collapsed}  0~3 겹침={overlap}/4  전체선택={len(sel_set)}개  최소위치={min(sel_set)}")
    else:
        print(f"sample {idx:2d}  collapsed={collapsed}  캡처 실패")

print(f"\n[smoothing=5 조건] collapse rate: {100*sum(collapse_flags)/len(collapse_flags):.1f}%  (검산: 0.0%가 나와야 지난 결과와 일치)")
if overlap_counts:
    avg_overlap = sum(overlap_counts) / len(overlap_counts)
    full_rate = 100 * sum(1 for x in overlap_counts if x == 4) / len(overlap_counts)
    print(f"평균 0~3 겹침: {avg_overlap:.2f}/4")
    print(f"4개 전부 우연 포함 비율: {full_rate:.1f}%")

"""
Phi-3 H2O implicit early-token retention의 causal test.
GPT 설계 그대로: 프롬프트는 절대 안 건드리고, cache selection에서만 위치를 조작.

A. H2O-natural       : sink_size=0, 조작 없음 (기존 baseline)
B. H2O-no-pos1        : 위치 1을 score=-inf로 만들어 절대 선택 안 되게 강제 배제, sink_size=0
C. H2O-no-pos1+pos0   : 위치 1 배제(B와 동일) + sink_size=1로 위치 0 강제 보존

전체 KV budget은 세 조건 다 동일(budget=20%) — 빠진 자리는 항상 다음 순위 토큰으로 자동 보충됨
(_select_with_sink의 top-k 로직 그대로 사용하므로).
"""
import sys, json
sys.path.insert(0, "/workspace/kv-cache-exp")
import torch
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed, word_repetition_ratio, char_repetition_ratio
import core.kv_cache_hook as kvh

model, tok, cfg = load_model_and_tokenizer("phi-3-mini")
ev = EvaluatorV2(model, tok, cfg)

_orig_h2o_compress = kvh.H2OCache._compress

def make_patched_compress(exclude_positions, sink_size_check):
    """exclude_positions에 있는 위치의 score를 -inf로 만들어 절대 선택 안 되게 함.
    나머지 로직(sink_size 포함)은 원본 H2OCache._compress와 완전히 동일.
    [2026-08-22 GPT 지적 반영] budget 일치·pos1 배제·pos0 포함을 런타임에 직접 검증."""
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

        # sanity assertion (GPT 지적) — budget 초과분이 있을 수 있어 seq_len<budget 케이스는 스킵
        expected_budget = min(budget, seq_len)
        assert len(indices) == expected_budget, \
            f"layer {layer_idx}: budget mismatch {len(indices)} != {expected_budget}"
        if 1 in exclude_positions and seq_len > 1:
            assert not (indices == 1).any(), f"layer {layer_idx}: pos1 still selected despite exclusion"
        if sink_size_check >= 1 and seq_len > 0:
            assert (indices == 0).any(), f"layer {layer_idx}: pos0 not retained despite sink_size>=1"

        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()
    return patched


CONDITIONS = [
    ("H2O-natural", set(), 0),
    ("H2O-no-pos1", {1}, 0),
    ("H2O-no-pos1+pos0", {1}, 1),
]

samples = load_longbench_task("gov_report", num_samples=30, seed=42)

all_results = {}
for label, exclude_pos, sink_size in CONDITIONS:
    print(f"\n{'='*60}\n조건: {label} (exclude={exclude_pos}, sink_size={sink_size})\n{'='*60}")
    kvh.H2OCache._compress = make_patched_compress(exclude_pos, sink_size)

    records = []
    for i, sample in enumerate(samples):
        r = ev.evaluate_sample(sample, "h2o", 0.20, method_kwargs=dict(sink_size=sink_size, invert_norm=True))

        # [2026-08-22 GPT 지적 반영] 생성 실패를 "정상"으로 오인하지 않도록 에러 가드
        if r.get("error"):
            raise RuntimeError(f"[{label}] Sample {i}: generation 실패 - {r['error']}")
        if r["prediction"] == "" and r["score"] == 0.0 and r.get("num_new_tokens") is None:
            raise RuntimeError(f"[{label}] Sample {i}: 빈 prediction + num_new_tokens=None, "
                                f"generation 실패 의심 - 결과를 신뢰할 수 없음")

        pred = r["prediction"]
        rec = {
            "sample_idx": i,
            "score": r["score"],
            "prediction": pred,
            "word_rep": word_repetition_ratio(pred),
            "char_rep": char_repetition_ratio(pred),
            "collapsed": is_collapsed(pred),
            "num_new_tokens": r.get("num_new_tokens"),
            "terminated_by_eos": r.get("terminated_by_eos"),
            "hit_max_new_tokens": r.get("hit_max_new_tokens"),
        }
        records.append(rec)
        if (i + 1) % 10 == 0:
            n_c = sum(1 for x in records if x["collapsed"])
            print(f"  [{i+1}/30] 누적 collapse: {n_c}")

    kvh.H2OCache._compress = _orig_h2o_compress  # 조건 끝날 때마다 원복

    n_c = sum(1 for x in records if x["collapsed"])
    avg_score = sum(x["score"] for x in records) / len(records)
    avg_tokens = sum(x["num_new_tokens"] or 0 for x in records) / len(records)
    n_hit_max = sum(1 for x in records if x["hit_max_new_tokens"])
    print(f"결과: avg_score={avg_score:.2f}  collapse={n_c}/30  avg_tokens={avg_tokens:.0f}  "
          f"max_tokens_도달={n_hit_max}/30")
    all_results[label] = {"records": records, "avg_score": avg_score, "collapse_n": n_c,
                           "avg_tokens": avg_tokens, "n_hit_max": n_hit_max}

print("\n\n=== 최종 요약 ===")
for label, _, _ in CONDITIONS:
    r = all_results[label]
    print(f"  {label:20s} avg_score={r['avg_score']:6.2f}  collapse={r['collapse_n']}/30  "
          f"avg_tokens={r['avg_tokens']:.0f}  max_tokens_도달={r['n_hit_max']}/30")

with open("/workspace/kv-cache-exp/results/v3_verified/phi3_h2o_causal_test.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("\n저장 완료: results/v3_verified/phi3_h2o_causal_test.json")

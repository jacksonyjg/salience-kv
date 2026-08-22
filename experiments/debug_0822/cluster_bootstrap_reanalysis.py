import json
import numpy as np
from scipy.stats import binomtest

np.random.seed(42)

SINK0 = "/home/claude/salience-kv-review/results/v3_verified/exp7_signal_ablation_qwen3-4b_sink0_20260819_130428.json"
SINK4 = "/home/claude/salience-kv-review/results/v3_verified/exp7_signal_ablation_qwen3-4b_sink4_20260819_202543.json"
SINK0_EXTRA = "/home/claude/salience-kv-review/results/v3_verified/exp7_extra_signals_qwen3-4b_20260819_151144.json"
SINK4_EXTRA = "/home/claude/salience-kv-review/results/v3_verified/exp7_extra_signals_qwen3-4b_20260819_223806.json"

def load(path):
    d = json.load(open(path))
    return {r["method"]: r for r in d["results"]}

def load_extra(path):
    d = json.load(open(path))
    return {r["variant"]: r for r in d["results"]}

d0 = load(SINK0); d0.update(load_extra(SINK0_EXTRA))
d4 = load(SINK4); d4.update(load_extra(SINK4_EXTRA))

variants = ["3signal_NVP", "4signal_NVPS", "wo_N", "wo_V", "wo_P", "N_only", "V_only", "P_only"]

def paired_collapse_arrays(variant):
    r0 = d0[variant]["sample_records"]
    r4 = d4[variant]["sample_records"]
    c0, c4, keys = [], [], []
    for task in r0.keys():
        if task not in r4:
            continue
        recs0 = {r["sample_idx"]: r["collapsed"] for r in r0[task]}
        recs4 = {r["sample_idx"]: r["collapsed"] for r in r4[task]}
        common_idx = sorted(set(recs0.keys()) & set(recs4.keys()))
        for i in common_idx:
            c0.append(recs0[i]); c4.append(recs4[i])
            keys.append((task, i))  # 클러스터 키
    return np.array(c0), np.array(c4), keys

print("=" * 70)
print("1) Variant별 exact McNemar + Holm-Bonferroni 보정 (주 분석)")
print("=" * 70)

pvals = []
summary = []
for v in variants:
    c0, c4, _ = paired_collapse_arrays(v)
    only0 = int(np.sum(c0 & ~c4))
    only4 = int(np.sum(~c0 & c4))
    n_disc = only0 + only4
    p = binomtest(only4, n_disc, p=0.5).pvalue if n_disc > 0 else 1.0
    pvals.append(p)
    summary.append((v, c0.mean()*100, c4.mean()*100, only0, only4, p))

# Holm-Bonferroni
order = np.argsort(pvals)
m = len(pvals)
holm_p = [None] * m
for rank, idx in enumerate(order):
    adj = pvals[idx] * (m - rank)
    holm_p[idx] = min(adj, 1.0)
# 단조성 보정(Holm 표준 절차: 이전 값보다 작아지지 않게)
sorted_idx = order
running_max = 0
for idx in sorted_idx:
    running_max = max(running_max, holm_p[idx])
    holm_p[idx] = running_max

for (v, s0, s4, only0, only4, p), hp in zip(summary, holm_p):
    sig = "유의(p<0.05)" if hp < 0.05 else "비유의"
    print(f"{v:14s} sink0={s0:5.1f}%  sink4={s4:5.1f}%  discordant(0only={only0:2d},4only={only4:2d})  "
          f"raw_p={p:.5f}  Holm_p={hp:.5f}  [{sig}]")

print("\n" + "=" * 70)
print("2) Pooled 효과 — cluster bootstrap ((task, sample_idx) 단위로 재표본화)")
print("=" * 70)

# 전체 (variant, task, idx) -> collapsed 매트릭스 구성
all_data = {}  # (task, idx) -> {variant: (c0, c4)}
for v in variants:
    c0, c4, keys = paired_collapse_arrays(v)
    for (task, idx), a, b in zip(keys, c0, c4):
        all_data.setdefault((task, idx), {})[v] = (a, b)

clusters = sorted(all_data.keys())  # 최대 210개 (task, idx) 조합
n_clusters = len(clusters)
print(f"클러스터 수(고유 (task, sample_idx) 조합): {n_clusters}")

def pooled_rates(selected_clusters):
    c0_all, c4_all = [], []
    for key in selected_clusters:
        for v, (a, b) in all_data[key].items():
            c0_all.append(a); c4_all.append(b)
    c0_all = np.array(c0_all); c4_all = np.array(c4_all)
    return c0_all.mean(), c4_all.mean()

obs0, obs4 = pooled_rates(clusters)
print(f"관측 pooled collapse rate: sink0={obs0*100:.2f}%  sink4={obs4*100:.2f}%  차이={100*(obs0-obs4):.2f}pp")

N_BOOT = 5000
diffs = np.zeros(N_BOOT)
n = n_clusters
for b in range(N_BOOT):
    idxs = np.random.randint(0, n, size=n)
    sampled = [clusters[i] for i in idxs]
    r0, r4 = pooled_rates(sampled)
    diffs[b] = r0 - r4

lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\ncluster bootstrap 95% CI (sink0-sink4 차이): [{lo*100:.2f}pp, {hi*100:.2f}pp]")
print(f"bootstrap 평균 차이: {diffs.mean()*100:.2f}pp")
print(f"0을 포함하는가: {'예 (비유의)' if lo <= 0 <= hi else '아니오 (유의)'}")

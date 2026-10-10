"""TABLE 5 paired significance analysis (added in revision, R1 comment 3).

Uses only the per-sample scores already stored in results/final (no model re-run).

Design (fixed before inspecting test outcomes; families cover ALL contrasts):
  * Pairing    : per-sample score differences matched by (task, sample_idx).
  * Statistic  : mean paired difference over all 210 samples (= mean of the 7 task means,
                 since every task has 30 samples).
  * Test       : two-sided paired sign-flip permutation test, R = 20,000, seed 42.
  * CI         : 95% task-stratified paired bootstrap, B = 10,000, seed 42
                 (sample indices resampled with replacement within each task; the same
                 resampled indices are applied to both configurations).
  * Families   : F1 = each of the 7 compression configurations vs FullKV (7 contrasts);
                 F2 = all pairs of compression configurations (21 contrasts).
  * Correction : Holm–Bonferroni within each family.

Data: exp1_qwen3-4b_full_20260820_112755.json, with QMSum replaced by the corrected run
      exp1_qwen3-4b_full_20260822_220917.json (see PAPER_TABLE_MAPPING.md, TABLE 5/6).

Usage: python experiments/exp_table5_significance.py [--results_dir results/final] [--out results/final/table5_significance.json]
"""
import argparse, itertools, json, os
import numpy as np

BASE = "exp1_qwen3-4b_full_20260820_112755.json"
CORR = "exp1_qwen3-4b_full_20260822_220917.json"
R_PERM, B_BOOT, SEED = 20_000, 10_000, 42


def load(results_dir):
    def table(d):
        return {r["method"]: {t: {s["sample_idx"]: s["score"] for s in recs}
                              for t, recs in r["sample_records"].items()}
                for r in d["results"]}
    base = json.load(open(os.path.join(results_dir, BASE)))
    corr = json.load(open(os.path.join(results_dir, CORR)))
    B, C = table(base), table(corr)
    for m in B:
        for t in C.get(m, {}):
            B[m][t] = C[m][t]
    tasks, methods = base["tasks"], list(B)
    for t in tasks:  # pairing check
        ref = sorted(B[methods[0]][t])
        assert all(sorted(B[m][t]) == ref for m in methods) and len(ref) == 30, t
    M = {m: np.array([[B[m][t][i] for i in sorted(B[m][t])] for t in tasks]) for m in methods}
    return tasks, methods, M


def perm_p(d, rng):
    d = d.ravel()
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(R_PERM, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return (1 + np.sum(null >= obs - 1e-12)) / (R_PERM + 1)


def boot_ci(d, idx):
    vals = np.take_along_axis(np.broadcast_to(d, idx.shape), idx, axis=2).mean(axis=(1, 2))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/final")
    ap.add_argument("--out", default="results/final/table5_significance.json")
    a = ap.parse_args()
    tasks, methods, M = load(a.results_dir)
    full, comp = "FullKV", [m for m in methods if m != "FullKV"]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 30, size=(B_BOOT, len(tasks), 30))
    out = {"design": {"R_perm": R_PERM, "B_boot": B_BOOT, "seed": SEED},
           "aggregate": {m: float(M[m].mean()) for m in methods}}
    prng = np.random.default_rng(SEED)
    f1 = []
    for m in comp:
        d = M[m] - M[full]
        f1.append({"config": m, "delta": float(d.mean()), "ci95": boot_ci(d, idx), "p": perm_p(d, prng)})
    for r, q in zip(f1, holm([r["p"] for r in f1])): r["p_holm"] = float(q)
    f2 = []
    for x, y in itertools.combinations(comp, 2):
        d = M[x] - M[y]
        f2.append({"a": x, "b": y, "delta": float(d.mean()), "ci95": boot_ci(d, idx), "p": perm_p(d, prng)})
    for r, q in zip(f2, holm([r["p"] for r in f2])): r["p_holm"] = float(q)
    out["family1_vs_fullkv"], out["family2_pairwise"] = f1, f2
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("Family 1 (vs FullKV):")
    for r in f1:
        print(f"  {r['config']:20s} Δ={r['delta']:+6.2f} CI=[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] p={r['p']:.5f} holm={r['p_holm']:.5f}")
    print("Family 2 (pairwise):")
    for r in f2:
        print(f"  {r['a']:20s}-{r['b']:20s} Δ={r['delta']:+6.2f} CI=[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] p={r['p']:.5f} holm={r['p_holm']:.5f}{'  *' if r['p_holm'] < 0.05 else ''}")


if __name__ == "__main__":
    main()

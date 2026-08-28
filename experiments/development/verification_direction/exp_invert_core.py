import sys, json, time
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2

def is_collapse(t):
    if not t or not t.strip(): return False
    w = t.split()
    if len(w) >= 6:
        tri = [" ".join(w[i:i+3]) for i in range(len(w)-2)]
        if tri and 1 - len(set(tri))/len(tri) > 0.3: return True
    c = t.replace(" ", "")
    if len(c) >= 10:
        g = [c[i:i+5] for i in range(len(c)-4)]
        if g and 1 - len(set(g))/len(g) > 0.7: return True
    return False

TASKS   = ["qmsum", "gov_report", "2wikimqa"]
METHODS = ["fullkv","h2o","snapkv","pyramidkv","adakv","ours"]
BUDGETS = [0.2, 0.8]
N = 15
OUT = "/workspace/kv-cache-exp/results/invert_core.json"

model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tok, cfg)
out = []; t0 = time.time()

for bud in BUDGETS:
    for inv in [True, False]:
        for m in METHODS:
            if m == "fullkv" and inv:
                continue
            for task in TASKS:
                data = load_longbench_task(task, num_samples=N, seed=42)
                sc, col = [], []
                for s in data:
                    try:
                        r = ev.evaluate_sample(s, method_name=m, budget_ratio=bud,
                                               method_kwargs=dict(sink_size=0, invert_norm=inv),
                                               measure_efficiency=False)
                        sc.append(r["score"]); col.append(is_collapse(r["prediction"]))
                        out.append({"budget":bud,"invert":inv,"method":m,"task":task,
                                    "score":r["score"],"collapse":col[-1],
                                    "pred":r["prediction"][:500]})
                    except Exception as e:
                        out.append({"budget":bud,"invert":inv,"method":m,"task":task,
                                    "error":str(e)[:200]})
                with open(OUT,"w") as f: json.dump(out,f,ensure_ascii=False)
                avg = sum(sc)/len(sc) if sc else 0
                cr  = 100*sum(col)/len(col) if col else 0
                print(f"[{(time.time()-t0)/60:6.1f}m] bud={int(bud*100)}% inv={str(inv):5s} "
                      f"{m:10s} {task:12s} score={avg:6.2f}  collapse={cr:5.1f}%", flush=True)
print("ALL DONE")

"""Local model vs Haiku 4.5 on the people both rated (cache only, no network). Usage: vs_haiku.py <ollama tag> <extra json>"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import llm_crossdomain as X
from aas.readers import llm_rater as L
from aas.readers import text as R
from sklearn.metrics import roc_auc_score
tag, extra = sys.argv[1], json.loads(sys.argv[2])
D = X.load(150)
loc = L.LLMRater(X.LocalProviderExtra(L.LocalProvider("http://127.0.0.1:11434/v1", tag), extra))
hai = L.LLMRater(L.AnthropicProvider("claude-haiku-4-5-20251001"))
T = X.TRAITS; rng = np.random.default_rng(0); out = {"model": tag}
S = {}; yb = {}; bow = {}; info = {}
for n, d in D.items():
    rl = [loc.cached(d["docs"][i]) for i in d["people"]]
    rh = [hai.cached(d["docs"][i]) for i in d["people"]]
    assert all(r is not None for r in rl + rh), "uncached rating: run the main script first"
    okl = np.array([r.ok for r in rl]); okh = np.array([r.ok for r in rh]); both = okl & okh
    Sl = np.array([[r.scores[t] for t in T] if r.ok else [np.nan]*5 for r in rl])
    Sh = np.array([[r.scores[t] for t in T] if r.ok else [np.nan]*5 for r in rh])
    Y = {t: d["Yb"][t].to_numpy()[d["people"]] for t in T}
    budgeted = [L.budget_text("\n".join(c for c in (R.clean(x) for x in d["docs"][i].split("\n")) if c), 1500)[0] for i in d["people"]]
    B = X.bow_scorer(n, D)(budgeted)
    # template-copy / constant-rating diagnostics
    const = np.nanmean([len(set(row)) == 1 for row in Sl[okl]]) if okl.any() else float("nan")
    all3 = np.nanmean([all(v == 3.0 for v in row) for row in Sl[okl]]) if okl.any() else float("nan")
    uniq = {t: int(len(np.unique(Sl[okl, j]))) for j, t in enumerate(T)}
    # missingness bias: bow AUC on kept vs dropped (local)
    def aucm(sc, idx): 
        try: return float(np.mean([roc_auc_score(Y[t][idx], sc[t][idx]) for t in T]))
        except ValueError: return float("nan")
    info[n] = {"n": len(okl), "local_missing": int((~okl).sum()), "haiku_missing": int((~okh).sum()), "both": int(both.sum()),
               "share_all_traits_equal": round(float(const), 3), "share_all_3.0": round(float(all3), 3), "distinct_scores_per_trait": uniq,
               "bow_auc_kept": aucm(B, np.where(okl)[0]), "bow_auc_dropped": aucm(B, np.where(~okl)[0]) if (~okl).sum() >= 10 else None,
               "local_r_with_haiku": {t: round(float(np.corrcoef(Sl[both, j], Sh[both, j])[0, 1]), 3) if np.std(Sl[both, j]) > 0 else None for j, t in enumerate(T)}}
    idx = np.where(both)[0]
    S[n] = {"local": {t: Sl[idx, j] for j, t in enumerate(T)}, "haiku": {t: Sh[idx, j] for j, t in enumerate(T)}, "bow": {t: B[t][idx] for t in T}}
    yb[n] = {t: Y[t][idx] for t in T}
def m(arm, ii, doms): return float(np.mean([np.mean([roc_auc_score(yb[n][t][ii[n]], S[n][arm][t][ii[n]]) for t in T]) for n in doms]))
full = {n: np.arange(len(yb[n]["O"])) for n in D}
H = X.HELD_OUT_FOR_BOW
pt = {a: m(a, full, H) for a in ("local", "haiku", "bow")}
per = {n: {a: round(m(a, full, [n]), 4) for a in ("local", "haiku", "bow")} for n in D}
ds = {"local-haiku": [], "local-bow": [], "haiku-bow": []}
for _ in range(2000):
    ii = {}
    for n in H:
        k = len(full[n])
        while True:
            i = rng.integers(0, k, k)
            if all(0 < yb[n][t][i].mean() < 1 for t in T): break
        ii[n] = i
    b = {a: m(a, ii, H) for a in ("local", "haiku", "bow")}
    for c in ds: x, y = c.split("-"); ds[c].append(b[x] - b[y])
out.update({"people_both_rated": {n: info[n]["both"] for n in D}, "heldout_mean_auc_on_shared_people": pt, "per_dataset_mean_auc_on_shared_people": per,
            "contrasts_on_shared_people": {c: {"delta": pt[c.split('-')[0]] - pt[c.split('-')[1]], "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for c, v in ds.items()},
            "diagnostics": info})
print(json.dumps(out, indent=1, default=float))

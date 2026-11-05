"""Distil Claude Haiku 4.5's zero-shot Big Five ratings into a small on-device text reader.

    cd py
    uv run python scripts/distill.py --dry-run            # the pool, prompts, cost estimate; no network
    # spends money (owner approved $10): Message Batches API, half price, disk cache
    export ANTHROPIC_API_KEY=...   # your own key
    AAS_LLM_ALLOW_SPEND=1 uv run python scripts/distill.py --rate
    uv run python scripts/distill.py --evaluate           # train students, score on Essays / PAN 2015 / vlogs, save

The pre-registration (pool, students, metric, pass rule) is in docs/distill.md and was committed before any
rating of the pool was obtained.

Pool: PersonalityCafe forum users (MBTI dataset), a seeded sample. Their MBTI labels are dropped on load and
never read. None of the three test sets (Essays, PAN 2015, vlogs) is in the pool or used for any fitting or tuning.

Teacher: the existing LLMRater (prompt b5-ipip50-v1, claude-haiku-4-5-20251001, same request settings, same leak
strip, label guard and 1,500-token cut). Batch answers are written into the same disk cache under the same key a
synchronous call would use, so nothing is ever paid twice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_crossdomain as LX  # noqa: E402
import text_crossdomain as XD  # noqa: E402

from aas.readers import llm_rater as L  # noqa: E402
from aas.readers import text as R  # noqa: E402
from aas.readers import text_train as TT  # noqa: E402
from aas.schema import RAW  # noqa: E402

DOCS = HERE.parent / "docs"
MODELS = HERE.parent / "models"
OUT_MODEL = MODELS / "text_distilled"
CACHE = RAW / "distill_cache"
SPEND_LOG = RAW.parent / "distill_spend.jsonl"
TRAITS = list(L.TRAITS)
MODEL = "claude-haiku-4-5-20251001"

POOL_N = 2000
POOL_SEED = 20261105
POOL_MIN_WORDS = 100
BUDGET_USD = 10.0
STOP_IF_ESTIMATE_OVER = 8.0
ALPHAS = np.logspace(-2, 4, 25)          # same grid as evaluate.ALPHAS
CV_FOLDS = 5
EMB_CHUNK, EMB_MAX_CHUNKS = 100, 40      # as text_crossdomain recipe d
BOW_CFG = {"name": "uni+bi", "ngram_range": [1, 2], "max_features": 20000}
STUDENTS = ["bow", "emb", "mix"]         # order of simplicity (ties within 0.01 go to the earlier one)
TIE = 0.01
BOOTS = 2000
HELD = ["pan15", "vlogs"]
ALL3 = ["essays", "pan15", "vlogs"]
log = TT.log


# ---------------------------------------------------------------- pool

def cleaned(raw: str) -> str:
    """Exactly what LLMRater.prepare does before budgeting."""
    return "\n".join(c for c in (R.clean(x) for x in raw.split("\n")) if c)


def budgeted(raw: str, budget: int = L.TEXT_TOKEN_BUDGET) -> str:
    """The text Haiku reads (and the student reads): cleaned, then cut to the token budget, seed 0."""
    return L.budget_text(cleaned(raw), budget, 0)[0]


def load_pool() -> pd.DataFrame:
    """POOL_N PersonalityCafe users in a seeded order, skipping anyone with < POOL_MIN_WORDS cleaned words or whose
    prompt the label guard refuses. The MBTI type column is dropped here and never looked at."""
    from aas.slices.text import load_mbti

    d = load_mbti().drop(columns=["mbti"])
    order = np.random.default_rng(POOL_SEED).permutation(len(d))
    rows, skipped = [], {"short": 0, "label_guard": 0}
    for i in order:
        raw = "\n".join(d.posts.iloc[i])
        if R.n_words(cleaned(raw)) < POOL_MIN_WORDS:
            skipped["short"] += 1
            continue
        try:
            L.build_messages(budgeted(raw))
        except L.LabelLeak:
            skipped["label_guard"] += 1
            continue
        rows.append({"person_id": d.person_id.iloc[i], "raw": raw})
        if len(rows) == POOL_N:
            break
    p = pd.DataFrame(rows)
    p.attrs["skipped"] = skipped
    return p


def rater() -> L.LLMRater:
    return L.LLMRater(L.AnthropicProvider(MODEL))


def pool_ratings(pool: pd.DataFrame, r: L.LLMRater) -> tuple[np.ndarray, dict]:
    """Cached Haiku scores per pool person (NaN when missing), and counts."""
    S = np.full((len(pool), 5), np.nan)
    n = {"cached_ok": 0, "cached_failed": 0, "uncached": 0}
    for k, raw in enumerate(pool.raw):
        c = r.cached(raw)
        if c is None:
            n["uncached"] += 1
        elif c.ok:
            S[k] = [c.scores[t] for t in TRAITS]
            n["cached_ok"] += 1
        else:
            n["cached_failed"] += 1
    return S, n


# ---------------------------------------------------------------- spend

def spent_so_far() -> float:
    if not SPEND_LOG.exists():
        return 0.0
    return sum(json.loads(x)["usd"] for x in SPEND_LOG.read_text().splitlines() if x.strip())


def log_spend(rec: dict) -> None:
    SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SPEND_LOG.open("a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")


def usd(in_tok: int, out_tok: int, batch: bool) -> float:
    p = LX.PRICES[MODEL]
    return (in_tok * p["in"] + out_tok * p["out"]) / 1e6 * (0.5 if batch else 1.0)


def estimate(pool: pd.DataFrame, r: L.LLMRater, only_uncached: bool) -> dict:
    tin, n, words = 0, 0, []
    for raw in pool.raw:
        if only_uncached and r.cached(raw) is not None:
            continue
        s, u, info = r.prepare(raw)
        tin += L.estimate_tokens(s + "\n" + u, "pre-4.7") + LX.MESSAGE_OVERHEAD_TOKENS
        words.append(info["words_out"])
        n += 1
    out_exp, out_max = n * LX.EXPECTED_OUTPUT_TOKENS, n * r.provider.max_tokens
    return {"requests": n, "input_tokens_est": tin, "median_words_sent": float(np.median(words)) if words else 0,
            "usd_sync_expected": usd(tin, out_exp, False), "usd_sync_upper": usd(tin, out_max, False),
            "usd_batch_expected": usd(tin, out_exp, True), "usd_batch_upper": usd(tin, out_max, True)}


def dry_run() -> int:
    pool = load_pool()
    r = rater()
    e = estimate(pool, r, only_uncached=False)
    e2 = estimate(pool, r, only_uncached=True)
    print(f"pool: {len(pool)} people, skipped while drawing: {pool.attrs['skipped']}")
    print(f"words in (median): {np.median([R.n_words(cleaned(x)) for x in pool.raw]):.0f}")
    print("all pool requests:", json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in e.items()}))
    print("not yet cached:   ", json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in e2.items()}))
    print(f"spent so far (this experiment's log): ${spent_so_far():.4f}")
    s, u, _ = r.prepare(pool.raw.iloc[0])
    print("\n---- example prompt ----\n" + (s + "\n\n" + u)[:1500] + "\n[...]")
    return 0


# ---------------------------------------------------------------- rating via the Batch API

def rate(poll_s: int = 30) -> int:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    pool = load_pool()
    r = rater()
    client = r.provider.client  # raises unless AAS_LLM_ALLOW_SPEND=1
    todo = {}
    for raw in pool.raw:
        s, u, info = r.prepare(raw)
        k = r.key(s, u)
        if not r._path(k).exists():
            todo[k] = (s, u, info)
    e = estimate(pool, r, only_uncached=True)
    already = spent_so_far()
    log(f"{len(todo)} uncached of {len(pool)}; estimate batch expected ${e['usd_batch_expected']:.2f}, upper "
        f"${e['usd_batch_upper']:.2f}; spent so far ${already:.4f}")
    if not todo:
        return 0
    if e["usd_batch_expected"] > STOP_IF_ESTIMATE_OVER or already + e["usd_batch_upper"] > BUDGET_USD:
        log("STOP: estimate over the limit, nothing submitted")
        return 2
    reqs = [Request(custom_id=k, params=MessageCreateParamsNonStreaming(
        model=r.provider.model, max_tokens=r.provider.max_tokens, system=s,
        messages=[{"role": "user", "content": u}], **r.provider.settings)) for k, (s, u, _) in todo.items()]
    b = client.messages.batches.create(requests=reqs)
    log(f"batch {b.id} submitted ({len(reqs)} requests)")
    log_spend({"event": "batch_submitted", "batch": b.id, "requests": len(reqs), "usd": 0.0})
    while True:
        b = client.messages.batches.retrieve(b.id)
        if b.processing_status == "ended":
            break
        c = b.request_counts
        log(f"  {b.processing_status}: processing {c.processing}, succeeded {c.succeeded}, errored {c.errored}")
        time.sleep(poll_s)
    tin = tout = 0
    ok = bad = err = 0
    retry = []
    for res in client.messages.batches.results(b.id):
        k = res.custom_id
        if res.result.type != "succeeded":
            err += 1
            continue
        m = res.result.message
        tin += m.usage.input_tokens
        tout += m.usage.output_tokens
        text = "".join(x.text for x in m.content if getattr(x, "type", None) == "text")
        att = [{"seconds": None, "input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens,
                "stop_reason": m.stop_reason, "via": f"batch {b.id}"}]
        s, u, info = todo[k]
        if m.stop_reason == "refusal":
            parsed, error = None, "refusal"
        else:
            try:
                parsed, error = L.parse_reply(text), None
            except L.BadReply as ex:
                retry.append((k, att, str(ex)))
                continue
        _write(r, k, info, att, text, error, parsed)
        ok += error is None
        bad += error is not None
    cost = usd(tin, tout, True)
    log_spend({"event": "batch_results", "batch": b.id, "input_tokens": tin, "output_tokens": tout, "usd": cost})
    log(f"batch done: {ok} ok, {bad} refused, {err} errored/expired (left uncached), {len(retry)} bad JSON; "
        f"tokens in {tin:,} out {tout:,}; ${cost:.4f}")
    # one retry for a bad reply, as LLMRater.rate does, synchronously (full price, few requests)
    rin = rout = 0
    for k, att, why in retry:
        s, u, info = todo[k]
        rep = r.provider.complete(s, u)
        rin += rep.input_tokens or 0
        rout += rep.output_tokens or 0
        att.append({"seconds": None, "input_tokens": rep.input_tokens, "output_tokens": rep.output_tokens,
                    "stop_reason": rep.stop_reason, "via": "sync retry"})
        try:
            parsed, error = L.parse_reply(rep.text), None
        except L.BadReply as ex:
            parsed, error = None, f"bad reply: {ex}"
        if rep.stop_reason == "refusal":
            parsed, error = None, "refusal"
        _write(r, k, info, att, rep.text, error, parsed)
    if retry:
        c2 = usd(rin, rout, False)
        log_spend({"event": "sync_retries", "requests": len(retry), "input_tokens": rin, "output_tokens": rout, "usd": c2})
        log(f"retried {len(retry)} bad replies synchronously: ${c2:.4f}")
    log(f"total spent on this experiment: ${spent_so_far():.4f}")
    return 0


def _write(r: L.LLMRater, k: str, info: dict, attempts: list, reply: str, error: str | None, parsed: dict | None) -> None:
    rec = {"prompt_version": L.PROMPT_VERSION, "provider": r.provider.name, "model": r.provider.model,
           "settings": r.provider.settings, "text_info": info, "attempts": attempts, "reply": reply, "error": error,
           "parsed": parsed, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p = r._path(k)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- student features

def embed_docs(tag: str, docs: list[str]) -> np.ndarray:
    import hashlib

    h = hashlib.sha256("\x00".join(docs).encode()).hexdigest()[:12]
    f = CACHE / f"emb_{tag}_{len(docs)}_{h}.npy"
    if f.exists():
        return np.load(f)
    from aas.slices.text import Embedder

    E = embed_with(Embedder(), docs)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, E)
    return E


def chunks_of(text: str) -> list[str]:
    w = text.split()
    return [" ".join(w[s:s + EMB_CHUNK]) for s in range(0, len(w), EMB_CHUNK)][:EMB_MAX_CHUNKS] or [""]


def embed_with(emb, docs: list[str]) -> np.ndarray:
    pieces, owner = [], []
    for i, t in enumerate(docs):
        for p in chunks_of(t):
            pieces.append(p)
            owner.append(i)
    log(f"embedding {len(pieces)} chunks for {len(docs)} people")
    V = emb.embed(pieces, batch_size=32)
    owner = np.array(owner)
    return np.stack([V[owner == i].mean(0) for i in range(len(docs))]).astype(np.float32)


class Student:
    """Ridge per trait from features to Haiku's 1-5 score. alpha by GCV inside whatever people it is fitted on."""

    def __init__(self, kind: str):
        assert kind in ("bow", "emb")
        self.kind = kind

    def fit(self, docs: list[str], E: np.ndarray | None, Y: np.ndarray) -> "Student":
        if self.kind == "bow":
            _, self.idf, self.vec = TT.fit_vocab(docs, BOW_CFG, 5)
            X = R.tfidf(self.vec.transform(docs), self.idf)
        else:
            self.sc = StandardScaler().fit(E)
            X = self.sc.transform(E)
        self.m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(X, Y)
        return self

    def X(self, docs, E):
        return R.tfidf(self.vec.transform(docs), self.idf) if self.kind == "bow" else self.sc.transform(E)

    def predict(self, docs, E) -> np.ndarray:
        return self.m.predict(self.X(docs, E))


def zcols(P: np.ndarray) -> np.ndarray:
    return (P - P.mean(0)) / np.where(P.std(0) > 0, P.std(0), 1.0)


def oof(docs: list[str], E: np.ndarray, Y: np.ndarray) -> dict:
    """Out-of-fold student predictions on the pool (5 folds, seed 0)."""
    P = {k: np.zeros_like(Y) for k in ("bow", "emb")}
    for tr, te in KFold(CV_FOLDS, shuffle=True, random_state=0).split(Y):
        for k in P:
            s = Student(k).fit([docs[i] for i in tr], E[tr], Y[tr])
            P[k][te] = s.predict([docs[i] for i in te], E[te])
    P["mix"] = (zcols(P["bow"]) + zcols(P["emb"])) / 2
    return P


def fidelity(P: np.ndarray, Y: np.ndarray) -> dict:
    return {t: float(np.corrcoef(P[:, j], Y[:, j])[0, 1]) for j, t in enumerate(TRAITS)}


# ---------------------------------------------------------------- evaluation

def mean_auc(yb, sc, idx, doms) -> float:
    return float(np.mean([np.mean([roc_auc_score(yb[n][t][idx[n]], sc[n][t][idx[n]]) for t in TRAITS]) for n in doms]))


def boot_idx(yb, n_of, rng, doms) -> dict:
    idx = {}
    for n in doms:
        m = n_of[n]
        while True:
            i = rng.integers(0, m, m)
            if all(0 < yb[n][t][i].mean() < 1 for t in TRAITS):
                break
        idx[n] = i
    return idx


def eval_sets(D: dict, students: dict, with_haiku: bool, r: L.LLMRater) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Per-trait scores for every arm on the people in D, plus arrays for the paired bootstrap."""
    rng = np.random.default_rng(0)
    rows, arms, yb, yz = [], {}, {}, {}
    for name, d in D.items():
        people = d["people"]
        if with_haiku:
            H = np.full((len(people), 5), np.nan)
            for k, i in enumerate(people):
                c = r.cached(d["docs"][i])
                if c is not None and c.ok:
                    H[k] = [c.scores[t] for t in TRAITS]
            keep = ~np.isnan(H).any(1)
            if not keep.all():
                log(f"{name}: {int((~keep).sum())} people without a cached Haiku rating dropped")
            people, H = people[keep], H[keep]
        texts = [budgeted(d["docs"][i]) for i in people]
        E = embed_docs(f"test_{name}", texts)
        a = {}
        P = {k: students[k].predict(texts, E) for k in ("bow", "emb")}
        P["mix"] = (zcols(P["bow"]) + zcols(P["emb"])) / 2   # z within this dataset, from scores alone
        for k in STUDENTS:
            a[f"student_{k}"] = {t: P[k][:, j] for j, t in enumerate(TRAITS)}
        if with_haiku:
            a["haiku"] = {t: H[:, j] for j, t in enumerate(TRAITS)}
        a["bow_reader"] = LX.bow_scorer(name, D)(texts)
        arms[name] = a
        yb[name] = {t: d["Yb"][t].to_numpy()[people] for t in TRAITS}
        yz[name] = {t: d["Yz"][t].to_numpy()[people] for t in TRAITS} if d["Yz"] is not None else None
        for arm, sc in a.items():
            for t in TRAITS:
                res = XD.score_block(sc[t], yb[name][t], yz[name][t] if yz[name] else None, None, rng)
                rows.append({"dataset": name, "arm": arm, "trait": t, **res})
        log(f"{name}: scored {len(people)} people")
    return pd.DataFrame(rows), arms, yb, yz


def contrasts(arms, yb, primary: str, others: list[str]) -> dict:
    rng = np.random.default_rng(1)
    names = list(arms)
    n_of = {n: len(yb[n]["O"]) for n in names}
    full = {n: np.arange(n_of[n]) for n in names}
    sets = {"pan15+vlogs": HELD, "essays+pan15+vlogs": ALL3}
    out = {}
    for sname, doms in sets.items():
        doms = [d for d in doms if d in names]
        allarms = [primary] + others
        point = {a: mean_auc(yb, {n: arms[n][a] for n in names}, full, doms) for a in allarms}
        bs = {a: [] for a in allarms}
        for _ in range(BOOTS):
            idx = boot_idx(yb, n_of, rng, doms)
            for a in allarms:
                bs[a].append(mean_auc(yb, {n: arms[n][a] for n in names}, idx, doms))
        bs = {a: np.array(v) for a, v in bs.items()}
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        out[sname] = {"point": point, "ci": {a: ci(bs[a]) for a in allarms},
                      "delta": {f"{primary}-{o}": {"delta": point[primary] - point[o], "ci": ci(bs[primary] - bs[o])} for o in others}}
        if "haiku" in others:
            # share of Haiku's lead over bag of words that the student keeps, with CI
            lead = bs["haiku"] - bs["bow_reader"]
            kept = (bs[primary] - bs["bow_reader"]) / np.where(np.abs(lead) > 1e-9, lead, np.nan)
            out[sname]["share_of_haiku_lead_kept"] = {
                "point": (point[primary] - point["bow_reader"]) / (point["haiku"] - point["bow_reader"]),
                "ci": [float(np.nanpercentile(kept, 2.5)), float(np.nanpercentile(kept, 97.5))]}
    return out


def evaluate() -> int:
    pool = load_pool()
    r = rater()
    S, cnt = pool_ratings(pool, r)
    keep = ~np.isnan(S).any(1)
    log(f"pool ratings: {cnt}; training on {int(keep.sum())} people")
    docs = [budgeted(x) for x, k in zip(pool.raw, keep) if k]
    Y = S[keep]
    E = embed_docs("pool", docs)

    # 1. pool-only selection (no test-set number is computed before this is fixed)
    P = oof(docs, E, Y)
    fid = {k: fidelity(P[k], Y) for k in STUDENTS}
    fid_mean = {k: float(np.mean(list(v.values()))) for k, v in fid.items()}
    best = max(fid_mean.values())
    chosen = next(k for k in STUDENTS if fid_mean[k] >= best - TIE)
    log(f"pool CV fidelity (r with Haiku, mean over traits): {fid_mean} -> chosen student: {chosen}")
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    (CACHE / "selection.json").write_text(json.dumps({"fidelity": fid, "fidelity_mean": fid_mean, "chosen": chosen,
                                                      "when": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))

    # 2. final students on the whole pool
    students = {k: Student(k).fit(docs, E, Y) for k in ("bow", "emb")}

    # 3. test sets: paired 150-person samples (Haiku cached), then every test person (student and bag of words only)
    primary = f"student_{chosen}"
    others_paired = ["haiku", "bow_reader"] + [f"student_{k}" for k in STUDENTS if k != chosen]
    D150 = LX.load(150)
    df150, arms150, yb150, _ = eval_sets(D150, students, True, r)
    c150 = contrasts(arms150, yb150, primary, others_paired)
    Dall = LX.load(None)
    dfall, armsall, yball, _ = eval_sets(Dall, students, False, r)
    call = contrasts(armsall, yball, primary, ["bow_reader"] + [f"student_{k}" for k in STUDENTS if k != chosen])

    # 4. the pre-registered rule
    s2 = c150["pan15+vlogs"]["point"]
    s3 = c150["essays+pan15+vlogs"]["point"]
    thr2 = 0.588
    thr3 = s3["bow_reader"] + 0.5 * (s3["haiku"] - s3["bow_reader"])
    passed = bool(s2[primary] >= thr2 and s3[primary] >= thr3)
    verdict = {"chosen_student": chosen, "pass": passed,
               "rule": "pass iff student mean AUC on PAN15+vlogs >= 0.588 AND student mean AUC on Essays+PAN15+vlogs >= "
                       "bow + 0.5*(haiku - bow) on the same paired 150-person samples",
               "pan15+vlogs": {"student": s2[primary], "threshold": thr2, "haiku": s2["haiku"], "bow": s2["bow_reader"]},
               "three_sets": {"student": s3[primary], "threshold": thr3, "haiku": s3["haiku"], "bow": s3["bow_reader"]}}

    res = {"pool": {"n_people": len(pool), "n_rated_ok": int(keep.sum()), "counts": cnt, "skipped_when_drawing": pool.attrs["skipped"],
                    "haiku_score_mean": dict(zip(TRAITS, Y.mean(0).round(3).tolist())),
                    "haiku_score_sd": dict(zip(TRAITS, Y.std(0).round(3).tolist()))},
           "selection": {"fidelity": fid, "fidelity_mean": fid_mean, "chosen": chosen},
           "paired150": c150, "all_test_people": call, "verdict": verdict,
           "spent_usd": spent_so_far()}
    (DOCS / "distill_results.json").write_text(json.dumps(res, indent=1, default=float))
    pd.concat([df150.assign(people="paired150"), dfall.assign(people="all")]).round(4).to_csv(DOCS / "distill_results.csv", index=False)
    print(df150.groupby(["arm", "dataset"]).auc.mean().unstack().round(3).to_string())
    print(dfall.groupby(["arm", "dataset"]).auc.mean().unstack().round(3).to_string())
    print(json.dumps(verdict, indent=1, default=float))

    save_model(students, chosen, docs, E, Y, fid, dfall, call, verdict, res)
    return 0


# ---------------------------------------------------------------- artifact

KIND_SOURCE = {"essay_like": "essays", "tweets": "pan15"}


def save_model(students, chosen, docs, E, Y, fid, dfall, call, verdict, res) -> None:
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    parts = ["bow", "emb"] if chosen == "mix" else [chosen]
    m = {"recipe": chosen, "traits": TRAITS, "budget_tokens": L.TEXT_TOKEN_BUDGET, "emb_chunk": EMB_CHUNK,
         "emb_max_chunks": EMB_MAX_CHUNKS}
    if "bow" in parts:
        s = students["bow"]
        m["bow"] = {"vocab": {k: int(v) for k, v in s.vec.vocabulary_.items()}, "ngram_range": tuple(BOW_CFG["ngram_range"]),
                    "idf": s.idf.astype(np.float32), "coef": s.m.coef_.astype(np.float32), "intercept": s.m.intercept_}
    if "emb" in parts:
        s = students["emb"]
        m["emb"] = {"model": "sentence-transformers/all-MiniLM-L6-v2", "mean": s.sc.mean_, "scale": s.sc.scale_,
                    "coef": s.m.coef_, "intercept": s.m.intercept_}
    # reference distribution of the student's own score on the pool (to turn a score into a z; no labels)
    pool_pred = {k: students[k].predict(docs, E) for k in ("bow", "emb")}
    if chosen == "mix":
        m["mix_ref"] = {k: {"mean": pool_pred[k].mean(0), "sd": pool_pred[k].std(0)} for k in ("bow", "emb")}
        ref = (zcols(pool_pred["bow"]) + zcols(pool_pred["emb"])) / 2
    else:
        ref = pool_pred[chosen]
    m["ref"] = {"mean": ref.mean(0), "sd": ref.std(0)}
    joblib.dump(m, OUT_MODEL / "model.joblib", compress=3)

    arm = f"student_{chosen}"
    a = dfall[dfall.arm == arm].set_index(["dataset", "trait"])
    per = {}
    for ds in ALL3:
        per[ds] = {t: {"auc": round(float(a.loc[(ds, t), "auc"]), 4), "auc_ci": [round(float(a.loc[(ds, t), "auc_lo"]), 4), round(float(a.loc[(ds, t), "auc_hi"]), 4)],
                       "r": (round(float(a.loc[(ds, t), "r"]), 4) if "r" in a and pd.notna(a.loc[(ds, t), "r"]) else None),
                       "n": int(a.loc[(ds, t), "n"])} for t in TRAITS}
    kinds = {}
    for kind, ds in KIND_SOURCE.items():
        kinds[kind] = {t: {"auc": per[ds][t]["auc"], "auc_ci": per[ds][t]["auc_ci"], "measured_on": ds} for t in TRAITS}
    # other prose (Reddit, YouTube comments, ...): not measured; stand-in = the lower of the two held-out own-words sets
    kinds["other_prose"] = {t: min(({"auc": per[ds][t]["auc"], "auc_ci": per[ds][t]["auc_ci"], "measured_on": ds} for ds in ("pan15", "vlogs")),
                                   key=lambda x: x["auc"]) | {"note": "stand-in, not measured"} for t in TRAITS}
    refused = [t for t in TRAITS if all(per[ds][t]["auc_ci"][0] <= 0.5 for ds in ALL3)]
    card = {
        "reader": "text-distilled", "version": "1.0.0", "trained": time.strftime("%Y-%m-%d"),
        "target": "teacher_score", "traits": TRAITS, "trait_names": L.TRAIT_KEYS,
        "default": False,
        "status": ("PASSED the pre-registered rule (docs/distill.md)" if verdict["pass"] else
                   "FAILED the pre-registered rule (docs/distill.md); saved for the record, not wired as the default reader"),
        "teacher": {"model": MODEL, "prompt_version": L.PROMPT_VERSION, "budget_tokens": L.TEXT_TOKEN_BUDGET,
                    "note": "Claude Haiku 4.5 read each pool person's leak-stripped text zero-shot and rated the five traits 1-5 (IPIP-50 scale). "
                            "The student learns to reproduce those ratings. No self-report label of any kind was used in training."},
        "data": {"pool": "PersonalityCafe forum users (Kaggle datasnaek/mbti-type via Hugging Face jingjietan/kaggle-mbti); MBTI labels dropped, never read",
                 "n": res["pool"]["n_rated_ok"], "seed": POOL_SEED,
                 "excluded": "Essays, PAN 2015 and YouTube vlogs (the test sets) are not in the pool and were not used for any fitting or tuning"},
        "features": {"recipe": chosen, "parts": parts,
                     "text": "readers.text.clean on every item, then cut to ~1,500 tokens (seeded random sample of whole items, or the first N words of one long text): the text the teacher read",
                     "bow": "uni+bigram counts, top 20,000 terms in >= 5 pool people, sublinear TF-IDF, L2 per person; ridge per trait, alpha by GCV on the pool",
                     "emb": "all-MiniLM-L6-v2 (fastembed ONNX, on device) mean over 100-word chunks (<= 40), standardised; ridge per trait, alpha by GCV on the pool",
                     "mix": "mean of the two students' z-scored predictions"},
        "fidelity_pool_cv": {"what": "Pearson r between the student's out-of-fold prediction and Haiku's rating, 5-fold CV on the pool", **fid},
        "test": {"what": "AUC against each test set's own median split (r against continuous scores where released), every test person, "
                         "student reading the same ~1,500-token cut text; 95% CI by bootstrap. All three are kinds of writing the student never saw.",
                 "per_dataset": per, "mean_auc": call["essays+pan15+vlogs"]["point"]},
        "verdict": verdict,
        "validity_by_source": {"date": time.strftime("%Y-%m-%d"), "kinds": kinds, "refused_traits": refused,
                               "classify": "as text-bigfive: readers.text.kind_of_writing"},
        "interval": "z = r * (score - pool mean) / pool sd, r implied by the stated AUC for that kind of writing; 80% interval z +- 1.28*sqrt(1 - r^2). "
                    "The z is derived for display: the student predicts a rating, not a questionnaire score.",
        "min_text": {"floor_words": 50, "note": "below 50 words the reader says nothing (the teacher was never asked about less than ~100 words)"},
        "privacy": "Runs entirely on the person's machine: TF-IDF and/or a 90 MB ONNX sentence encoder plus 5 linear heads. No text leaves the device.",
        "terms_note": "Trained on outputs of Claude Haiku 4.5. Anthropic's commercial terms restrict using outputs to build competing AI models; "
                      "a 5-number personality regressor is unlikely to be that, but check before any commercial release.",
    }
    (OUT_MODEL / "card.json").write_text(json.dumps(card, indent=1, default=float))
    log(f"saved {OUT_MODEL} ({sum(f.stat().st_size for f in OUT_MODEL.iterdir()) / 1e6:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--rate", action="store_true")
    g.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run()
    if a.rate:
        return rate()
    return evaluate()


if __name__ == "__main__":
    raise SystemExit(main())

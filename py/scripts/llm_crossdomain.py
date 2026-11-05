"""Zero-shot LLM Big Five ratings vs the trained bag-of-words model, on the same people, across kinds of writing.

    cd py
    uv run python scripts/llm_crossdomain.py --dry-run                 # no network: prompts, tokens, cost
    uv run python scripts/llm_crossdomain.py --dry-run --full          # cost for every test person
    # REAL run (spends money; owner approval first):
    AAS_LLM_ALLOW_SPEND=1 AAS_LLM_MODEL=claude-haiku-4-5-20251001 uv run python scripts/llm_crossdomain.py --live
    # local model on this laptop (free, nothing leaves the machine):
    AAS_LLM_PROVIDER=local AAS_LLM_LOCAL_URL=http://127.0.0.1:11434/v1 AAS_LLM_LOCAL_MODEL=qwen3:8b \\
        uv run python scripts/llm_crossdomain.py --live

The pre-registered decision rule is in docs/llm_rater.md and is restated in DECISION_RULE below; it was
written before any rating was obtained.

People ("test sets only")
  essays  the 494 people in the saved Essays model's held-out 20% test split (seed 0, stratified, as in
          text_train.train). The bag-of-words comparison uses a model refitted on the 80% dev split only,
          because the saved artifact was finally refit on everyone.
  pan15   all 294 Twitter users. The bag-of-words model never saw any PAN data, so every PAN person is a test person.
  vlogs   all 404 vloggers, same reason (and vlogs are evaluation-only by licence).
  A fixed random sample of --n people (default 150, seed 20261101) is drawn from each pool.

Scores compared, all on the SAME people
  llm        the LLM's 1-5 rating for the trait
  bow        the bag-of-words model's score on the SAME budgeted text the LLM saw (primary comparison)
  bow_full   the bag-of-words model's score on the person's full text (as in text_crossdomain.md)
  combined   z(llm) + z(bow), z-scored within dataset from the scores alone (no labels, nothing fitted)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import text_crossdomain as XD  # noqa: E402  (reuses its scoring + saved-artifact helpers)

from aas.readers import llm_rater as L  # noqa: E402
from aas.readers import text as R  # noqa: E402
from aas.readers import text_train as TT  # noqa: E402

DOCS = HERE.parent / "docs"
DATASETS = ["essays", "pan15", "vlogs"]
HELD_OUT_FOR_BOW = ["pan15", "vlogs"]   # domains the bag-of-words model never trained on: the decision is made here
TRAITS = list(L.TRAITS)
SAMPLE_SEED = 20261101
BOOTS = 2000
THRESHOLD = 0.03

# Anthropic list prices, USD per million tokens, read 2026-11-01 from
# https://platform.claude.com/docs/en/about-claude/pricing (Batch API = 50% of both).
PRICES = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00, "label": "Claude Haiku 4.5 (requested default)"},
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "label": "Claude Haiku 5.5 (prompts <= 100k tokens)"},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "label": "Claude Sonnet 5.5"},
}
EXPECTED_OUTPUT_TOKENS = 250   # five scores + five one-line reasons as JSON (estimate; the live run records the truth)
MESSAGE_OVERHEAD_TOKENS = 15   # role/format tokens around system + user

DECISION_RULE = f"""Pre-registered (docs/llm_rater.md), before any rating was obtained:
  Primary metric: AUC against each dataset's own median split, mean over the 5 traits, averaged over the
  bag-of-words model's HELD-OUT domains {HELD_OUT_FOR_BOW} with each domain weighted equally.
  Contrast: delta = that mean for the LLM minus that mean for bag-of-words on the SAME people and SAME text.
  95% CI by paired bootstrap ({BOOTS} resamples of people within each dataset).
  REPLACE  if delta(llm - bow) >= {THRESHOLD} and its CI excludes 0.
  AUGMENT  else if delta(combined - bow) >= {THRESHOLD} and its CI excludes 0.
  KEEP     otherwise (bag-of-words stays; the LLM is not adopted).
  Essays (in-domain for bag-of-words), Pearson r and per-trait numbers are reported but do not enter the rule.
  People whose LLM rating is missing (refusal or invalid reply twice) are dropped from ALL arms and counted."""


# ---------------------------------------------------------------- people and texts

def essays_test_people() -> np.ndarray:
    c = TT.load_essays_corpus()
    idx = np.arange(len(c.ids))
    strat = c.Y.astype(str).agg("".join, axis=1)
    dev, test = train_test_split(idx, test_size=0.2, random_state=0, stratify=strat)
    return np.sort(dev), np.sort(test)


def load(n: int | None) -> dict:
    out = {}
    for name in DATASETS:
        d = TT.load_b5_domain(name)
        docs = d["docs"]
        if name == "essays":
            # loader joins 20-word pieces with newlines; the LLM should read one continuous essay
            docs = [" ".join(x.split("\n")) for x in docs]
            dev, pool = essays_test_people()
            d["dev"] = dev
        else:
            pool = np.arange(len(docs))
        if n is not None and n < len(pool):
            pool = np.sort(np.random.default_rng(SAMPLE_SEED).choice(pool, n, replace=False))
        d["docs"], d["people"] = docs, pool
        out[name] = d
    return out


# ---------------------------------------------------------------- bag-of-words scores

def bow_scorer(name: str, D: dict):
    """A function docs -> {trait: raw score}. Essays: refit on the dev split only (the saved model saw the
    test people). Others: the saved artifact (it never saw them)."""
    if name != "essays":
        return XD.current_artifact_scores
    card = json.loads((XD.MODELS / "text_bigfive" / "card.json").read_text())
    pen = card["features"]["penalty"]
    d = D["essays"]
    dev_docs = [d["docs"][i] for i in d["dev"]]
    _, idf, vec = TT.fit_vocab(dev_docs, XD.BOW_CFG, 5)
    X = R.tfidf(vec.transform(dev_docs), idf)
    heads = {t: TT.fit_head(X, d["Yb"][t].to_numpy()[d["dev"]], "binary_median", pen[t]) for t in TRAITS}

    def score(docs):
        Xt = R.tfidf(vec.transform(docs), idf)
        return {t: Xt @ heads[t][0] + heads[t][1] for t in TRAITS}
    return score


# ---------------------------------------------------------------- dry run

def dry_run(D: dict, rater: L.LLMRater, out: Path) -> int:
    rows = []
    for name, d in D.items():
        for i in d["people"]:
            system, user, info = rater.prepare(d["docs"][i])
            rows.append({"dataset": name, "person": int(i), "words_in": info["words_in"], "words_sent": info["words_out"],
                         "cut": info["cut"], "chars": len(system) + len(user),
                         "tok_pre47": L.estimate_tokens(system + "\n" + user, "pre-4.7") + MESSAGE_OVERHEAD_TOKENS,
                         "tok_47": L.estimate_tokens(system + "\n" + user, "4.7+") + MESSAGE_OVERHEAD_TOKENS})
            if len(rows) == 1:
                example = system + "\n\n" + user
    df = pd.DataFrame(rows)
    print("\nPeople and text sent (words), per dataset:")
    print(df.groupby("dataset").agg(people=("person", "size"), median_words_in=("words_in", "median"),
                                    median_words_sent=("words_sent", "median"), share_cut=("cut", "mean"),
                                    est_input_tokens=("tok_pre47", "sum")).round(2).to_string())
    print(f"\nToken estimate: max(chars/4, words*1.33) per request (+{MESSAGE_OVERHEAD_TOKENS} overhead); x1.30 for the "
          "Claude 4.7+ tokenizer (Haiku 5.5, Sonnet 5.5). Output assumed "
          f"{EXPECTED_OUTPUT_TOKENS} tokens per reply (upper bound: every reply hits max_tokens).")
    cost = []
    for m, p in PRICES.items():
        cfg = L.ANTHROPIC_MODELS[m]
        tin = df["tok_47" if cfg["tokenizer"] == "4.7+" else "tok_pre47"].sum()
        n = len(df)
        exp = (tin * p["in"] + n * EXPECTED_OUTPUT_TOKENS * p["out"]) / 1e6
        hi = (tin * p["in"] + n * cfg["max_tokens"] * p["out"]) / 1e6
        cost.append({"model": m, "requests": n, "input_tokens": int(tin), "usd_expected": round(exp, 2),
                     "usd_upper": round(hi, 2), "usd_expected_batch": round(exp / 2, 2), "price_in_out": f"${p['in']}/${p['out']} per MTok"})
    cost.append({"model": "local (this laptop)", "requests": len(df), "input_tokens": int(df.tok_pre47.sum()), "usd_expected": 0.0,
                 "usd_upper": 0.0, "usd_expected_batch": 0.0, "price_in_out": "electricity only"})
    cdf = pd.DataFrame(cost)
    print("\nEstimated cost (USD; prices from platform.claude.com/docs/en/about-claude/pricing, read 2026-11-01):")
    print(cdf.to_string(index=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(out, index=False)
    print(f"\nwritten {out}")
    print("\n---- example prompt (first person) ----\n" + example[:1800] + ("\n[...]" if len(example) > 1800 else ""))
    print("\n" + DECISION_RULE)
    return 0


# ---------------------------------------------------------------- live run and scoring

def rate_all(D: dict, rater: L.LLMRater) -> dict:
    out = {}
    for name, d in D.items():
        S = np.full((len(d["people"]), 5), np.nan)
        errs, new = {}, 0
        for k, i in enumerate(d["people"]):
            r = rater.rate(d["docs"][i])
            new += not r.cached
            if r.ok:
                S[k] = [r.scores[t] for t in TRAITS]
            else:
                errs[r.error] = errs.get(r.error, 0) + 1
            if (k + 1) % 25 == 0:
                TT.log(f"{name}: {k + 1}/{len(d['people'])} rated ({new} new calls)")
        out[name] = {"S": S, "errors": errs}
        TT.log(f"{name}: done, {int(np.isnan(S[:, 0]).sum())} missing {errs}")
    return out


def zs(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() or 1.0)


def mean_auc(yb: dict, sc: dict, idx: dict, doms: list[str]) -> float:
    return float(np.mean([np.mean([roc_auc_score(yb[n][t][idx[n]], sc[n][t][idx[n]]) for t in TRAITS]) for n in doms]))


def live(D: dict, rater: L.LLMRater, outbase: Path) -> int:
    ratings = rate_all(D, rater)
    rng = np.random.default_rng(0)
    rows, arms, yb_all = [], {}, {}
    for name, d in D.items():
        S = ratings[name]["S"]
        keep = ~np.isnan(S).any(1)
        people = d["people"][keep]
        budgeted = [L.budget_text("\n".join(c for c in (R.clean(x) for x in d["docs"][i].split("\n")) if c), rater.budget_tokens)[0] for i in people]
        scorer = bow_scorer(name, D)
        bow = scorer(budgeted)
        bow_full = scorer([d["docs"][i] for i in people])
        llm = {t: S[keep, j] for j, t in enumerate(TRAITS)}
        comb = {t: zs(llm[t]) + zs(bow[t]) for t in TRAITS}
        arms[name] = {"llm": llm, "bow": bow, "bow_full": bow_full, "combined": comb}
        yb_all[name] = {t: d["Yb"][t].to_numpy()[people] for t in TRAITS}
        for arm, sc in arms[name].items():
            for t in TRAITS:
                yz = d["Yz"][t].to_numpy()[people] if d["Yz"] is not None else None
                res = XD.score_block(sc[t], yb_all[name][t], yz, None, rng)
                rows.append({"dataset": name, "arm": arm, "trait": t, "missing": int((~keep).sum()), **res})
    df = pd.DataFrame(rows)
    # paired bootstrap of the decision contrasts
    full = {n: np.arange(len(yb_all[n]["O"])) for n in D}
    point = {a: mean_auc(yb_all, {n: arms[n][a] for n in D}, full, HELD_OUT_FOR_BOW) for a in ("llm", "bow", "combined", "bow_full")}
    deltas = {"llm-bow": [], "combined-bow": []}
    for _ in range(BOOTS):
        idx = {}
        for n in HELD_OUT_FOR_BOW:
            m = len(full[n])
            while True:
                i = rng.integers(0, m, m)
                if all(0 < yb_all[n][t][i].mean() < 1 for t in TRAITS):
                    break
            idx[n] = i
        b = {a: mean_auc(yb_all, {n: arms[n][a] for n in D}, idx, HELD_OUT_FOR_BOW) for a in ("llm", "bow", "combined")}
        deltas["llm-bow"].append(b["llm"] - b["bow"])
        deltas["combined-bow"].append(b["combined"] - b["bow"])
    summ = {k: {"delta": (point[k.split("-")[0]] - point["bow"]), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in deltas.items()}
    ok = lambda s: s["delta"] >= THRESHOLD and s["ci"][0] > 0  # noqa: E731
    decision = "REPLACE" if ok(summ["llm-bow"]) else "AUGMENT" if ok(summ["combined-bow"]) else "KEEP"
    df.round(4).to_csv(str(outbase) + ".csv", index=False)
    meta = {"model": rater.provider.model, "provider": rater.provider.name, "prompt_version": L.PROMPT_VERSION,
            "budget_tokens": rater.budget_tokens, "point_mean_auc_heldout": point, "contrasts": summ, "decision": decision,
            "missing": {n: ratings[n]["errors"] for n in D}, "rule": DECISION_RULE}
    Path(str(outbase) + ".json").write_text(json.dumps(meta, indent=1, default=float))
    print(df.groupby(["arm", "dataset"]).auc.mean().unstack().round(3).to_string())
    print(json.dumps({"point": point, "contrasts": summ, "decision": decision}, indent=1, default=float))
    return 0


class LocalProviderExtra(L.LocalProvider):
    """Local provider that adds request-body fields from AAS_LLM_LOCAL_EXTRA (JSON), e.g.
    '{"reasoning_effort": "none"}' to switch a thinking model's thinking off by flag (Ollama's OpenAI endpoint)
    instead of editing the prompt. The fields go into `settings`, so into the cache key. Added 2026-11-01 for the
    local-model run (docs/llm_rater.md); the default path (no env var) is unchanged."""

    def __init__(self, base: L.LocalProvider, extra: dict):
        super().__init__(base.url, base.model, base.timeout)
        self.extra = dict(extra)
        self.settings = {**self.settings, **self.extra}

    def complete(self, system: str, user: str) -> L.Reply:
        import urllib.request
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **self.extra}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as f:
            r = json.loads(f.read())
        ch = r["choices"][0]
        u = r.get("usage") or {}
        return L.Reply(ch["message"].get("content") or "", u.get("prompt_tokens"), u.get("completion_tokens"), ch.get("finish_reason"))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="build every prompt and estimate cost; no network")
    g.add_argument("--live", action="store_true", help="call the configured provider (Anthropic needs AAS_LLM_ALLOW_SPEND=1)")
    ap.add_argument("--n", type=int, default=150, help="people per dataset (fixed seeded sample)")
    ap.add_argument("--full", action="store_true", help="every test person instead of the sample")
    a = ap.parse_args()
    D = load(None if a.full else a.n)
    if a.dry_run:
        # provider is irrelevant to prompt building; a never-called Anthropic provider is used for settings only
        rater = L.LLMRater(L.AnthropicProvider())
        return dry_run(D, rater, DOCS / ("llm_cost_full.csv" if a.full else "llm_cost_sample.csv"))
    rater = L.LLMRater.from_env()
    extra = json.loads(os.environ.get("AAS_LLM_LOCAL_EXTRA") or "{}")
    if extra and isinstance(rater.provider, L.LocalProvider):
        rater.provider = LocalProviderExtra(rater.provider, extra)
    tag = f"{rater.provider.name}_{rater.provider.model}".replace("/", "_").replace(":", "_")
    return live(D, rater, DOCS / f"llm_crossdomain_results_{tag}")


if __name__ == "__main__":
    raise SystemExit(main())

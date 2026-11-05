"""The pre-registered analysis of the AAS whole-person study (docs/study/PREREG.md, section 6).

Written and tested on simulated data BEFORE any study data exist. Run it once on the simulated study
(output posted with the registration) and once on the real data folder. Nothing here is tuned after
seeing outcomes; every choice the pre-registration left open is listed in PREREG_ISSUES below and in
docs/study/PREREG.md "Changes before data collection".

    cd py
    uv run python scripts/simulate_study.py --n 250 --out /tmp/sim250        # a fake study folder
    uv run python scripts/study_analysis.py /tmp/sim250 --out /tmp/sim250/results
    uv run python scripts/study_analysis.py /tmp/sim250 --fast                # quick look (fewer repeats)

Outputs (deterministic given --seed and the input folder): results.json (machine-readable, every number)
and report.txt (plain text, same numbers).

What it computes, in PREREG order:
    exclusions  rules 1 to 7 in order, every excluded person logged with the rule (flow table)
    pooling     every pooled-over-traits test runs over the traits the method speaks for in >= 90% of people
                (P22: text-bigfive refuses openness on Reddit/X writing); the set and per-trait n are printed
    H1          text reader z vs quiz z per trait: Pearson r, one-sided Fisher p, 95% person bootstrap CI, Holm/5
    H2          10-fold x 10 repeats (seeds 0..9, folds stratified by source pattern): text calibration
                ("single") vs ridge-per-source stacked with text ("combined"); per-person paired error
                difference, pooled over traits (primary, t-test + 10,000 sign flips), per trait (Holm/5),
                Delta r with bootstrap CI, best-single-in-CV sensitivity, per-pair sub-analyses (Holm/5 each)
    H3          (a) shipped text intervals, (b) combined model's in-CV residual-quantile intervals:
                pooled coverage with 90% person-cluster bootstrap CI and TOST against [0.75, 0.85];
                per trait coverage and width; text coverage by word-count tercile
    H4          per-person sum over traits of |q| - |q - combined|, t-test + sign flips; skill per trait
                with CI, for the combined model and for the text reader as shipped
    S1          Google alone and Spotify alone, learned in CV, permutation p (>= 1,000), Holm/5 per source
    S2          MBTI reader P(letter) vs OEJTS letter: AUC with DeLong CI, one-sided, Holm/4
    S3          quiz z ~ reader z * log words with trait fixed effects, person-clustered SEs, one-sided product term
    S4          refit text calibration (all) and refit text weights (common-word opt-in subset) vs the shipped reader
    S5          quiz retest reliability on the retest subset: Pearson r (Fisher CI) and ICC(2,1) (bootstrap CI)
    S6          Reddit-only vs X-only writers: difference in text validity, bootstrap CI, two-sided
    S7          fusion (added 2026-11-01, PREREG change log): `bayes` (n = 0, as shipped by FusedReader) and
                `bayes_eb_cp` (fitted in the same CV): validity, coverage, skill, paired error vs text, and the
                pilot checks of docs/fusion.md (each source's real vs stated validity, text/MBTI error correlation)

The input format is assumed (aas.features was being written in parallel). ALL knowledge of the raw
format is in the "FORMAT ADAPTER" section (load_study and its helpers); everything below it works on
the normalised Participant record. See docs/study/README.md "How to run the analysis" for the folder layout.
"""

# =====================================================================================================
# PREREG_ISSUES: where PREREG.md was ambiguous or disagreed with docs/fusion.md, and what this script does.
# Items marked [EDITED] were fixed in PREREG.md (change log at its end, 2026-11-01). Items marked
# [OWNER] need the owner's decision before posting; the script implements the stated default.
#
#  P1 [EDITED] Rule 2 said "failed 2 of 3 attention checks"; PROTOCOL says "2 or more", and OEJTS (which
#     carries the third check) is optional. Now: 2 or more of the instructed items presented (2 or 3).
#  P2 [EDITED] Rule 5 counts are bucketed/rounded in the payload (words to the nearest 100, searches and
#     plays in buckets), so the 100-word / 300-search / 2,000-play floors cannot be re-checked from it.
#     The extractor's own floor flags decide. People who never sent a Session 2 payload are listed as
#     their own row under rule 5 (dropout, not misconduct). A source below its floor is treated as absent.
#  P3 [EDITED] Rule 6 "keep the first" duplicate = earliest submitted_at (ties: smaller study ID).
#  P4 [EDITED] Google "24-hour rhythm as circular mean and spread (2)": a circular mean in hours cannot
#     enter a linear ridge model (23:00 and 01:00 are close but numerically far). Replaced by the first
#     trigonometric moment of the hour histogram (mean cosine, mean sine), two numbers that carry both
#     the circular mean and the spread. Hour histogram = mean of the search and watch histograms present.
#  P5 [EDITED] "Combined model" in H2 to H4 is the CV-learned stack of PREREG section 6. docs/fusion.md
#     (written the same day) instead makes the zero-data `bayes` fusion of text-bigfive + text-mbti the
#     shipped model and `bayes_eb_cp` the main-study model; PREREG registered no analysis of either.
#     Added as secondary family S7 (validity, coverage, skill, paired error vs text, pilot checks).
#  P6 [OWNER] README's fusion update says "fusion beats text alone" is "a secondary, exploratory outcome"
#     at n = 250; PREREG keeps H2 primary. Script keeps H2 primary as registered. Proposed wording in
#     the report if the owner wants to demote it.
#  P7 [OWNER] H2/H4 per-person sums when a person lacks a quiz trait (< 8 of 10 items): summed over the
#     traits they have (proposed wording: "D_i = sum over the traits with a quiz z for person i").
#  P8 [OWNER] Minimum data for a learned source model inside a training fold: a non-text source enters the
#     combined model in a fold only if >= 20 training people have it; otherwise that fold's combined
#     prediction uses text alone for everyone.
#  P9 [OWNER] "Topic-share principal components fitted inside the training folds": PCA is fitted on the
#     outer training fold's Google people (unsupervised, no outcome, no test person); inner folds reuse it.
#  P10 [OWNER] Spotify "basic-export people flagged": flag reported, NOT a model feature (keeps 9 features).
#  P11 [OWNER] H3(b) and H4 with 10 repeats: coverage, width and |error| are averaged over repeats per
#     person-trait before pooling; H2's d_it likewise (as registered); Delta r uses repeat-averaged predictions.
#  P12 [OWNER] H4 "also reported for the text reader alone": the shipped z (no fitting) is the registered
#     comparison; the CV-calibrated text ("single") is reported beside it.
#  P13 [OWNER] S3 "pooled with trait fixed effects": trait intercepts, common slopes for z, log words and
#     their product; log words centred at the sample mean.
#  P14 [OWNER] S5 sample: Session-1 completers who pass exclusion rules 1-4 and 7 (not the data-donation
#     rules 5-6, as PROTOCOL 5.2 randomises before Session 2), whose retest also passes rule 3 and was
#     taken 12 to 20 days after Session 1 (PROTOCOL window). ICC CI by person bootstrap.
#  P15 [OWNER] Quiz-quality rule 3 with missing items: median time and runs are computed over the answered
#     IPIP items in page order; instructed (attention) items are not part of the sequence.
#  P16 [OWNER] S1 "learned in CV": the H2 source model (ridge, alpha by inner 5-fold CV on evaluate.ALPHAS),
#     10-fold x 10 repeats on people with that source; permutation null refits with 1 repeat (as
#     evaluate.predict does), 1,000 permutations.
#  P17 [OWNER] S4 "improves on the shipped reader": paired per-person squared-error difference vs the shipped
#     z, as in H2; calibration one-sided, weights two-sided (exploratory direction). Subset needs >= 30.
#  P18 [OWNER] S7 bayes_eb_cp is computed when n >= 60 and labelled exploratory below 250 (fusion.md: main
#     study n >= 250); S7 uses the same stratified folds and seeds as H2.
#  P19 [OWNER] Two-stage option: this script is run once, on the final data; it has no interim mode.
#  P21 [OWNER, found by the dry run] r of a learned model under 10 repeats of CV (H2 Delta r, S1, S4, S7b)
#     is the mean of the repeats' own out-of-fold r. evaluate.predict instead takes r of the repeat-averaged
#     prediction; for a model that learned little that prediction is (training mean - y_i / (n - n_fold)),
#     so its r is pulled towards -1 as repeats grow (seen in simulation: Spotify-alone r -0.58 where the
#     1-repeat permutation null averaged -0.15), and the 1-repeat permutation null no longer matches it.
#     Paired error tests (d_it) were never affected: they average per-repeat errors. Proposed fix for
#     evaluate.predict (not edited here): same change.
#  P22 [OWNER, 2026-11-01 reader change] text-bigfive now refuses openness (usable=False, validity 0) when most
#     of a person's words are Reddit/X, i.e. for most participants. PREREG's "pooled over the five traits" is
#     then impossible without silently dropping or imputing O. Rule: a trait enters a method's pooled tests
#     (H2 primary, H3 pooled coverage, H4 primary, S7 pooled) only if that method has a usable estimate for
#     >= 90% of usable participants (availability, never outcomes); the pooled trait set and the per-trait n
#     are printed; other traits are analysed per trait on the subset that has them (skipped below 30 people
#     for CV analyses, 10 for correlations). S4 compares the raw-score refit with z = 0 for a trait the shipped
#     reader mostly refuses (the shipped answer is "no number"). Fused O (MBTI N/S only) still exists.
#  P20 [OWNER, found by the dry run] The registered OLS stack can put large weights (up to 4.6 at n = 60,
#     2.4 at n = 250) on a source's inner out-of-fold predictions. When a source carries little signal those
#     predictions are nearly constant and run backwards under CV (evaluate.py), so the stack amplifies noise
#     and the combined model loses to text alone for reasons unrelated to the question. Constraining the
#     source weights to be >= 0 (Breiman 1996, "Stacked regressions") was better or equal in 7 of 8
#     simulated studies (mean d at n = 60: -0.294 -> -0.056 in the worst one). Registered OLS stays the
#     default (Config.stack_nonneg = False, CLI --stack-nonneg); wording proposed in PREREG.md change log.
# =====================================================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

from aas import quiz
from aas.evaluate import ALPHAS, holm
from aas.fusion import MBTI_LINKS, Fuser, auc_to_r, make_method, readings_from_estimates
from aas.person import Estimate
from aas.schema import TRAITS

LEVEL = 0.80
MBTI_AXES = ["I/E", "N/S", "T/F", "J/P"]
GOOGLE_DIRECT = ["hour_cos", "hour_sin", "days_active_share", "query_mean_words", "query_question_share",
                 "channel_entropy", "top10_channel_share", "repeat_watch_share"]
N_TOPIC_PCS = 2
SPOTIFY_FEATURES = ["new_share", "old_friends", "season_turnover", "clock_spread", "clock_regularity",
                    "finish_share", "skip_seconds", "dial_explore", "dial_regular"]
Z95, Z80POW = stats.norm.ppf(0.95), stats.norm.ppf(0.80)


@dataclass
class Config:
    seed: int = 0
    folds: int = 10
    repeats: int = 10           # PREREG: seeds 0..9
    inner: int = 5              # inner CV for ridge alpha, for stacking OOF and for H3(b) residuals
    boots: int = 2000
    flips: int = 10000
    perms: int = 1000           # S1 permutation null
    min_source_n: int = 20      # P8
    retest_window: tuple = (12, 20)   # P14
    eb_min_n: int = 60          # P18
    eb_confirm_n: int = 250
    min_quiz_items: int = 8     # PREREG section 5
    pool_min_share: float = 0.90  # P22: a trait enters pooled tests if the method speaks for >= 90% of people
    min_trait_n: int = 30       # per-trait CV analyses need this many people with the estimate
    stack_nonneg: bool = True   # P20, accepted 2026-11-01: source weights >= 0 (Breiman 1996); False = plain OLS

    @classmethod
    def fast(cls, seed: int = 0) -> "Config":
        return cls(seed=seed, repeats=2, boots=400, flips=2000, perms=200)


# =====================================================================================================
# FORMAT ADAPTER. The only code that knows the raw folder layout. Adapt here when aas.features lands.
#
# <folder>/study.json              {"extractors": {"<version>": "<sha256>"}, "pilot_ids": [...]}  (frozen at pilot end)
# <folder>/session1/<sid>.json     {"study_id", "session1_at" (ISO), "pilot": bool,
#                                   "ipip": {"answers": {item: 1..5 | null}, "item_ms": {item: ms}},
#                                   "attention": [{"id", "answer", "expected"}],     2 or 3 instructed items
#                                   "use_answers": "yes" | "no",
#                                   "oejts": {"I/E": "I"|"E", ...} | null,          scored by the study server
#                                   "retest_assigned": bool}
# <folder>/payloads/<sid>.json     {"study_id", "submitted_at", "extractor_version", "extractor_hash",
#                                   "payload": <an aas.features.build() record (has "schema_version"; validated
#                                               with features.validate, then converted by _from_features_record)>
#                                   OR the simple shape below (what scripts/simulate_study.py writes):
#                                   "payload": {
#                                     "sources": {"text": bool, "google": bool, "spotify": bool},
#                                     "floors": {"text"|"google"|"spotify": {"passed": bool, "reason": str|null}},
#                                     "text": {"words": int (nearest 100), "platforms": ["reddit"|"x"],
#                                              "estimates": [aas.person.Estimate fields, text-bigfive + text-mbti],
#                                              "head_scores": {trait: float},
#                                              "word_rates": {word: per-1000} | null},     opt-in block
#                                     "google": {"search_hour_shares": [24] | null, "watch_hour_shares": [24] | null,
#                                                "days_active_share", "query_mean_words", "query_question_share",
#                                                "channel_entropy" | null, "top10_channel_share" | null,
#                                                "repeat_watch_share" | null, "topic_shares": {topic: share}} | null,
#                                     "spotify": {"export": "extended"|"basic",
#                                                 "traits": {name: {"value": float, "percentile": float|null}}} | null}}
# <folder>/retest/<sid>.json       {"study_id", "taken_at", "ipip": {"answers", "item_ms"}}
# <folder>/withdrawn.json          ["<sid>", ...]
# =====================================================================================================

@dataclass
class Participant:
    sid: str
    session1_at: datetime | None = None
    pilot: bool = False
    answers: dict = field(default_factory=dict)       # IPIP item -> 1..5 or None
    item_ms: dict = field(default_factory=dict)
    attention_failed: int = 0
    attention_n: int = 0
    use_answers: bool = True
    oejts: dict | None = None                        # axis -> True if FIRST letter
    retest_assigned: bool = False
    retest_answers: dict | None = None
    retest_item_ms: dict | None = None
    retest_days: float | None = None
    withdrawn: bool = False
    # Session 2 (None / empty when no payload)
    has_payload: bool = False
    extractor_version: str | None = None
    extractor_hash: str | None = None
    submitted_at: datetime | None = None
    payload_key: str | None = None                   # canonical hash of the feature content (duplicates)
    problems: list = field(default_factory=list)     # impossible values found by the validator
    floors: dict = field(default_factory=dict)       # source -> passed
    words: float | None = None
    platforms: tuple = ()
    estimates: list = field(default_factory=list)    # aas.person.Estimate (text-bigfive, text-mbti)
    head: dict = field(default_factory=dict)
    word_rates: dict | None = None
    google: dict | None = None                       # GOOGLE_DIRECT -> float (NaN if absent)
    topics: dict | None = None
    spotify: dict | None = None                      # SPOTIFY_FEATURES -> float
    spotify_basic: bool = False


def _dt(s):
    return datetime.fromisoformat(s) if s else None


def _num(x):
    return float("nan") if x is None else float(x)


def _hour_moments(*hists):
    hs = [np.asarray(h, float) for h in hists if h is not None]
    if not hs:
        return float("nan"), float("nan")
    h = np.mean([x / x.sum() if x.sum() > 0 else x for x in hs], axis=0)
    ang = 2 * np.pi * np.arange(24) / 24
    return float(h @ np.cos(ang)), float(h @ np.sin(ang))


def _payload_problems(p: dict) -> list[str]:
    """Impossible values (rule 6). Format-specific, so it lives in the adapter."""
    bad = []

    def chk(cond, msg):
        if not cond:
            bad.append(msg)

    def finite(x):
        return x is None or (isinstance(x, (int, float)) and math.isfinite(x))

    t = p.get("text") or {}
    chk(isinstance(t.get("words", 0), (int, float)) and t.get("words", 0) >= 0, "text.words negative or not a number")
    for e in t.get("estimates", []):
        v, lo, hi = e.get("value"), e.get("low"), e.get("high")
        if not all(finite(x) for x in (v, lo, hi, e.get("validity"))):
            bad.append(f"{e.get('reader')} {e.get('trait')}: non-finite number")
            continue
        if e.get("usable", True) and v is not None:
            if e.get("kind") == "mbti":
                chk(0 <= v <= 1, f"{e.get('reader')} {e.get('trait')}: probability {v} outside [0, 1]")
            elif e.get("kind") == "bigfive":
                chk(-5 <= v <= 5, f"{e.get('reader')} {e.get('trait')}: z {v} outside [-5, 5]")
            if lo is not None and hi is not None:
                chk(lo <= v <= hi, f"{e.get('reader')} {e.get('trait')}: value outside its own interval")
        val, kind = e.get("validity"), e.get("validity_kind")
        if val is not None:
            chk((0 <= val <= 1) if kind == "auc" else (-1 <= val <= 1), f"{e.get('reader')} {e.get('trait')}: validity {val} impossible")
    g = p.get("google")
    if g:
        for k in ("days_active_share", "query_question_share", "top10_channel_share", "repeat_watch_share"):
            x = g.get(k)
            chk(x is None or (finite(x) and 0 <= x <= 1), f"google.{k} = {x} not a share")
        for k in ("query_mean_words", "channel_entropy"):
            x = g.get(k)
            chk(x is None or (finite(x) and x >= 0), f"google.{k} = {x} negative")
        for k in ("search_hour_shares", "watch_hour_shares"):
            h = g.get(k)
            if h is not None:
                chk(len(h) == 24 and all(finite(x) and 0 <= x <= 1 for x in h) and abs(sum(h) - 1) <= 0.05, f"google.{k} not 24 shares summing to 1")
        ts = list((g.get("topic_shares") or {}).values())
        chk(all(finite(x) and 0 <= x <= 1 for x in ts) and (not ts or abs(sum(ts) - 1) <= 0.05), "google.topic_shares not shares summing to 1")
    s = p.get("spotify")
    if s:
        for k, d in (s.get("traits") or {}).items():
            chk(finite(d.get("value")), f"spotify.{k} non-finite")
            pc = d.get("percentile")
            chk(pc is None or (finite(pc) and 0 <= pc <= 100), f"spotify.{k} percentile {pc} outside 0..100")
    return bad


def _mean_hist(*hs):
    hs = [h for h in hs if h]
    return [sum(x) / len(hs) for x in zip(*hs)] if hs else None


def _mean_shares(*ds):
    ds = [d for d in ds if d]
    if not ds:
        return {}
    keys = sorted({k for d in ds for k in d})
    return {k: sum((d.get(k) or 0.0) for d in ds) / len(ds) for k in keys}


def _from_features_record(r: dict) -> dict:
    """aas.features.build() record (schema_version 1.x) -> the payload shape used here.

    Google: rhythm and day share from activity.google_search + activity.youtube_watch (mean of those
    present); query form from search_form.google_search; viewing form from viewing_form; topic shares =
    mean of the google_search and youtube_watch topic shares present. Spotify: the music reader's 9 values."""
    rd = r.get("readers") or {}
    src = r.get("sources") or {}
    act = r.get("activity") or {}

    def ests(name, kind):
        out = []
        for t, e in ((rd.get(name) or {}).get("estimates") or {}).items():
            out.append({"reader": name, "trait": t, "kind": kind, **{k: e.get(k) for k in (
                "value", "low", "high", "percentile", "validity", "validity_kind", "usable", "interval_level")}})
        return out

    gs, yw = act.get("google_search") or {}, act.get("youtube_watch") or {}
    sf = (r.get("search_form") or {}).get("google_search") or {}
    vf = r.get("viewing_form") or {}
    tp = r.get("topics") or {}
    has_g = bool(src.get("google_search") or src.get("youtube_watch") or src.get("youtube_search"))
    days = [x for x in (gs.get("active_day_share"), yw.get("active_day_share")) if x is not None]
    google = None if not has_g else {
        "search_hour_shares": gs.get("hours_utc"), "watch_hour_shares": yw.get("hours_utc"),
        "days_active_share": sum(days) / len(days) if days else None,
        "query_mean_words": sf.get("mean_words"), "query_question_share": sf.get("question_share"),
        "channel_entropy": vf.get("channel_entropy_bits"), "top10_channel_share": vf.get("top10_channel_share"),
        "repeat_watch_share": vf.get("repeat_watch_share"),
        "topic_shares": _mean_shares(tp.get("google_search"), tp.get("youtube_watch"))}
    music = (rd.get("music") or {}).get("estimates") or {}
    spot = None if not src.get("spotify") else {
        "export": (r.get("spotify") or {}).get("export"),
        "traits": {k: {"value": e.get("value"), "percentile": e.get("percentile")} for k, e in music.items() if e.get("usable")}}
    wr = r.get("word_rates")
    return {"sources": {"text": bool(src.get("reddit") or src.get("x")), "google": has_g, "spotify": bool(src.get("spotify"))},
            "floors": r.get("floors") or {},
            "text": {"words": (r.get("writing") or {}).get("words_rounded"),
                     "platforms": [k for k in ("reddit", "x") if src.get(k)],
                     "estimates": ests("text-bigfive", "bigfive") + ests("text-mbti", "mbti"),
                     "head_scores": (rd.get("text-bigfive") or {}).get("raw_scores") or {},
                     "word_rates": (wr or {}).get("per_1k")},
            "google": google, "spotify": spot}


def _parse_payload(rec: dict, P: Participant) -> None:
    p = rec.get("payload") or {}
    P.has_payload = True
    P.extractor_version, P.extractor_hash = rec.get("extractor_version"), rec.get("extractor_hash")
    P.submitted_at = _dt(rec.get("submitted_at"))
    P.payload_key = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    schema_problems = []
    if "schema_version" in p:            # an aas.features record: check it against its own schema, then convert
        try:
            from aas import features
            features.validate(p)
        except ValueError as ex:
            schema_problems.append(f"feature schema: {str(ex)[:120]}")
        except ImportError:
            pass
        ex = p.get("extractor") or {}
        P.extractor_version = P.extractor_version or ex.get("version")
        P.extractor_hash = P.extractor_hash or ex.get("hash")
        p = _from_features_record(p)
    P.problems = schema_problems + _payload_problems(p)
    src = p.get("sources") or {}
    fl = p.get("floors") or {}
    P.floors = {k: bool(src.get(k)) and bool((fl.get(k) or {}).get("passed")) for k in ("text", "google", "spotify")}
    t = p.get("text") or {}
    P.words = t.get("words")
    P.platforms = tuple(sorted(t.get("platforms") or []))
    keep = {f for f in Estimate.__dataclass_fields__}
    P.estimates = [Estimate(**{k: v for k, v in e.items() if k in keep}) for e in t.get("estimates", [])]
    P.head = dict(t.get("head_scores") or {})
    P.word_rates = t.get("word_rates")
    g = p.get("google")
    if g and P.floors["google"]:
        hc, hs = _hour_moments(g.get("search_hour_shares"), g.get("watch_hour_shares"))
        P.google = {"hour_cos": hc, "hour_sin": hs, **{k: _num(g.get(k)) for k in GOOGLE_DIRECT[2:]}}
        P.topics = dict(g.get("topic_shares") or {})
    s = p.get("spotify")
    if s and P.floors["spotify"]:
        tr = s.get("traits") or {}
        P.spotify = {k: _num((tr.get(k) or {}).get("value")) for k in SPOTIFY_FEATURES}
        P.spotify_basic = s.get("export") == "basic"


def load_study(folder: str | Path) -> tuple[list[Participant], dict]:
    """Read a study folder into Participants (sorted by study ID) and the study manifest."""
    folder = Path(folder)
    meta = json.loads((folder / "study.json").read_text()) if (folder / "study.json").exists() else {}
    pilot_ids = set(meta.get("pilot_ids", []))
    withdrawn = set(json.loads((folder / "withdrawn.json").read_text())) if (folder / "withdrawn.json").exists() else set()
    people: dict[str, Participant] = {}
    for f in sorted((folder / "session1").glob("*.json")):
        r = json.loads(f.read_text())
        sid = r["study_id"]
        att = r.get("attention") or []
        oe = r.get("oejts")
        people[sid] = Participant(
            sid=sid, session1_at=_dt(r.get("session1_at")), pilot=bool(r.get("pilot")) or sid in pilot_ids,
            answers=dict((r.get("ipip") or {}).get("answers") or {}), item_ms=dict((r.get("ipip") or {}).get("item_ms") or {}),
            attention_failed=sum(1 for a in att if a.get("answer") != a.get("expected")), attention_n=len(att),
            use_answers=str(r.get("use_answers", "yes")).lower() != "no",
            oejts=None if not oe else {ax: oe[ax] == ax[0] for ax in MBTI_AXES if oe.get(ax) in (ax[0], ax[2])},
            retest_assigned=bool(r.get("retest_assigned")), withdrawn=sid in withdrawn)
    for f in sorted((folder / "payloads").glob("*.json")) if (folder / "payloads").exists() else []:
        r = json.loads(f.read_text())
        P = people.get(r["study_id"])
        if P is None:   # a payload without a Session 1 record cannot be analysed (no quiz); counted in meta
            meta.setdefault("orphan_payloads", []).append(r["study_id"])
            continue
        _parse_payload(r, P)
    for f in sorted((folder / "retest").glob("*.json")) if (folder / "retest").exists() else []:
        r = json.loads(f.read_text())
        P = people.get(r["study_id"])
        if P is None:
            continue
        P.retest_answers = dict(r["ipip"]["answers"])
        P.retest_item_ms = dict(r["ipip"].get("item_ms") or {})
        if P.session1_at and r.get("taken_at"):
            P.retest_days = (_dt(r["taken_at"]) - P.session1_at).total_seconds() / 86400
    return [people[k] for k in sorted(people)], meta

# ============================================================================ end of FORMAT ADAPTER


# ---------------------------------------------------------------- quiz scoring and quality (rule 3)

def quiz_quality(answers: dict, item_ms: dict) -> str | None:
    """Rule 3 (quiz.clean rules) over the answered IPIP items in page order (P15). None = passes."""
    seq = [answers.get(c) for c in quiz.PAGE_ORDER]
    seq = [a for a in seq if a in (1, 2, 3, 4, 5)]
    times = [item_ms[c] for c in quiz.PAGE_ORDER if answers.get(c) in (1, 2, 3, 4, 5) and item_ms.get(c) is not None]
    if times and float(np.median(times)) < quiz.MIN_MEDIAN_ITEM_MS:
        return f"median item time under {quiz.MIN_MEDIAN_ITEM_MS} ms"
    if len(seq) >= 2 and len(set(seq)) == 1:
        return "all answers identical"
    if seq and int(quiz._longest_run(np.array([seq]))[0]) >= quiz.MAX_RUN:
        return f"{quiz.MAX_RUN} or more identical answers in a row"
    return None


def quiz_z(answers: dict, min_items: int = 8) -> dict[str, float]:
    """Quiz z per trait; a trait with fewer than min_items of 10 answered is missing (PREREG section 5)."""
    ans = {c: int(a) for c, a in answers.items() if a in (1, 2, 3, 4, 5) and c in quiz.KEY}
    sc = quiz.score(ans)
    return {t: sc[t]["z"] for t in TRAITS if t in sc and sc[t]["n_items"] >= min_items}


# ---------------------------------------------------------------- exclusions (PREREG section 7)

RULES = [
    ("1", "pilot participant"),
    ("2", "failed 2 or more instructed-response (attention) items"),
    ("3", "quiz quality (quiz.clean rules)"),
    ("4", "answered 'no' to 'should we use your answers?'"),
    ("5a", "no Session 2 payload (did not return)"),
    ("5b", "writing below 100 words, or no second source above its floor (extractor floor flags)"),
    ("6a", "payload integrity: unknown extractor version or hash"),
    ("6b", "payload integrity: impossible values"),
    ("6c", "payload integrity: duplicate payload (kept the earliest)"),
    ("7", "withdrawn"),
]


def apply_exclusions(people: list[Participant], meta: dict) -> tuple[list[Participant], list[dict], list[dict]]:
    """Rules in PREREG order. Returns (usable, flow table rows, per-person log)."""
    known = meta.get("extractors") or {}
    log, flow = [], []
    remaining = list(people)
    flow.append({"rule": "0", "label": "Session 1 records", "excluded": 0, "remaining": len(remaining)})

    def step(code, label, why_fn):
        nonlocal remaining
        keep = []
        for P in remaining:
            why = why_fn(P)
            if why:
                log.append({"study_id": P.sid, "rule": code, "reason": why})
            else:
                keep.append(P)
        flow.append({"rule": code, "label": label, "excluded": len(remaining) - len(keep), "remaining": len(keep)})
        remaining = keep

    lab = dict(RULES)
    step("1", lab["1"], lambda P: "pilot" if P.pilot else None)
    step("2", lab["2"], lambda P: f"failed {P.attention_failed} of {P.attention_n}" if P.attention_failed >= 2 else None)
    step("3", lab["3"], lambda P: quiz_quality(P.answers, P.item_ms))
    step("4", lab["4"], lambda P: None if P.use_answers else "said no")
    step("5a", lab["5a"], lambda P: None if P.has_payload else "no payload")
    step("5b", lab["5b"], lambda P: ("text below floor" if not P.floors.get("text") else
                                     "no second source above its floor" if not (P.floors.get("google") or P.floors.get("spotify")) else None))
    step("6a", lab["6a"], lambda P: None if known.get(P.extractor_version) == P.extractor_hash and P.extractor_version
         else f"extractor {P.extractor_version} / {str(P.extractor_hash)[:12]} not in the frozen list")
    step("6b", lab["6b"], lambda P: "; ".join(P.problems[:3]) if P.problems else None)
    first: dict[str, str] = {}   # over ALL payloads: a copy is a copy even if the original was excluded earlier
    for P in sorted((P for P in people if P.has_payload), key=lambda P: (P.submitted_at or datetime.max, P.sid)):
        first.setdefault(P.payload_key, P.sid)
    step("6c", lab["6c"], lambda P: None if first[P.payload_key] == P.sid else f"same payload as {first[P.payload_key]}")
    step("7", lab["7"], lambda P: "withdrew" if P.withdrawn else None)
    return remaining, flow, log


# ---------------------------------------------------------------- statistics helpers

def _boot_weights(n: int, B: int, rng) -> np.ndarray:
    """Person bootstrap as multinomial counts (B, n): resampling people, keeping all their traits."""
    return rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)


def wr(x, y, W):
    """Weighted Pearson r for each row of weights W (B, n)."""
    sw = W.sum(1)
    mx, my = W @ x / sw, W @ y / sw
    cxy = W @ (x * y) / sw - mx * my
    vx, vy = W @ (x * x) / sw - mx ** 2, W @ (y * y) / sw - my ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return cxy / np.sqrt(vx * vy)


def r_(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rep_r(P, y) -> float:
    """r of a learned model under repeated CV = mean over repeats of each repeat's out-of-fold r (P21).
    NOT r of the repeat-averaged prediction: for a model that learned nothing that prediction is the
    training mean minus a multiple of y_i, so its r runs towards -1 as repeats grow."""
    return float(np.nanmean([r_(p, y) for p in np.atleast_2d(P)]))


def rep_wr(P, y, W):
    return np.nanmean([wr(p, y, W) for p in np.atleast_2d(P)], axis=0)


def fisher_p_greater(r: float, n: int) -> float:
    if not np.isfinite(r) or n < 4:
        return float("nan")
    return float(stats.norm.sf(np.arctanh(np.clip(r, -0.999999, 0.999999)) * math.sqrt(n - 3)))


def fisher_ci(r: float, n: int, level=0.95):
    if not np.isfinite(r) or n < 4:
        return (float("nan"), float("nan"))
    z, h = np.arctanh(np.clip(r, -0.999999, 0.999999)), stats.norm.ppf(0.5 + level / 2) / math.sqrt(n - 3)
    return (float(np.tanh(z - h)), float(np.tanh(z + h)))


def mde_r(n: int) -> float:
    """Smallest r a one-sided alpha .05 test detects with 80% power at this n (Fisher z)."""
    return float(np.tanh((Z95 + Z80POW) / math.sqrt(n - 3))) if n > 3 else float("nan")


def paired_test(d: np.ndarray, flips: int, rng, alternative="greater") -> dict:
    """Mean of per-person differences: one-sample t-test and sign-flip permutation (PREREG H2/H4).

    If the two disagree at alpha .05, the permutation p is the one reported (PREREG)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"n": n, "mean": float("nan"), "p": float("nan")}
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
    p_t = float(stats.ttest_1samp(d, 0.0, alternative=alternative).pvalue) if se > 0 else float("nan")
    S = rng.choice([-1.0, 1.0], size=(flips, n))
    null = S @ d / n
    if alternative == "greater":
        p_perm = float((1 + np.sum(null >= m - 1e-15)) / (1 + flips))
    else:
        p_perm = float((1 + np.sum(np.abs(null) >= abs(m) - 1e-15)) / (1 + flips))
    tq = stats.t.ppf(0.975, n - 1)
    disagree = (p_t < 0.05) != (p_perm < 0.05)
    return {"n": n, "mean": m, "ci95": [m - tq * se, m + tq * se], "p_t": p_t, "p_perm": p_perm,
            "p": p_perm if disagree else p_t, "p_source": "permutation (disagreed with t)" if disagree else "t-test (permutation agrees)"}


def holm_into(rows: list[dict], key="p", out="p_holm"):
    ps = np.array([r.get(key, np.nan) for r in rows], float)
    ok = np.isfinite(ps)
    adj = np.full(len(ps), np.nan)
    if ok.any():
        adj[ok] = holm(ps[ok])
    for r, a in zip(rows, adj):
        r[out] = float(a)
        r["supported"] = bool(np.isfinite(a) and a < 0.05)


def pool_traits(avail: np.ndarray, n: int, min_share: float) -> dict:
    """P22: which traits enter the pooled-over-traits tests, decided from availability (never outcomes).
    avail: (n, 5) bool, the method has a usable estimate AND the person has a quiz z for that trait."""
    cnt = avail.sum(0)
    share = cnt / max(n, 1)
    pool = [t for j, t in enumerate(TRAITS) if share[j] >= min_share]
    left = {t: f"usable for {int(cnt[j])} of {n} ({share[j]:.0%}); below {min_share:.0%}, analysed per trait only"
            for j, t in enumerate(TRAITS) if t not in pool}
    J = [TRAITS.index(t) for t in pool]
    k = avail[:, J].sum(1) if J else np.zeros(n)
    return {"traits": pool, "excluded": left, "n_by_trait": {t: int(cnt[j]) for j, t in enumerate(TRAITS)},
            "people_with_all_pooled_traits": int((k == len(J)).sum()),
            "people_with_some_pooled_traits": int(((k > 0) & (k < len(J))).sum()),
            "people_with_none": int((k == 0).sum()),
            "rule": "pooled sums run over these traits only; a person missing one of them (quiz < 8 items or a "
                    "per-person refusal) contributes the sum over the pooled traits they have (P7, P22)"}


def coverage_stats(inside: np.ndarray, width: np.ndarray, pid: np.ndarray, B: int, rng, level=LEVEL) -> dict:
    """Pooled coverage over person-traits with a 90% person-cluster bootstrap CI and the TOST rule."""
    ok = np.isfinite(inside)
    inside, width, pid = inside[ok], width[ok], pid[ok]
    if len(inside) == 0:
        return {"n_person_traits": 0}
    ids, inv = np.unique(pid, return_inverse=True)
    num = np.bincount(inv, weights=inside, minlength=len(ids))
    den = np.bincount(inv, minlength=len(ids)).astype(float)
    W = _boot_weights(len(ids), B, rng)
    bs = (W @ num) / (W @ den)
    lo, hi = np.percentile(bs, [5, 95])
    cov = float(inside.mean())
    return {"n_people": int(len(ids)), "n_person_traits": int(len(inside)), "coverage": cov, "ci90": [float(lo), float(hi)],
            "stated": level, "equivalent_0.75_0.85": bool(lo >= level - 0.05 and hi <= level + 0.05),
            "mean_width": float(np.nanmean(width))}


def cluster_ols(y, X, groups):
    """OLS with CR1 person-clustered standard errors. Returns (beta, se, df = G - 1)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    ids, inv = np.unique(groups, return_inverse=True)
    G, N, K = len(ids), len(y), X.shape[1]
    S = np.zeros((G, K))
    np.add.at(S, inv, X * e[:, None])
    meat = S.T @ S
    c = G / (G - 1) * (N - 1) / (N - K)
    V = c * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V)), G - 1


def delong_auc(score_pos, score_neg):
    """AUC and its DeLong variance for one classifier."""
    X, Y = np.asarray(score_pos, float), np.asarray(score_neg, float)
    m, n = len(X), len(Y)
    psi = (X[:, None] > Y[None, :]).astype(float) + 0.5 * (X[:, None] == Y[None, :])
    auc = float(psi.mean())
    v10, v01 = psi.mean(1), psi.mean(0)
    var = v10.var(ddof=1) / m + v01.var(ddof=1) / n
    return auc, float(var)


def icc21(a, b) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater (McGraw & Wong A,1)."""
    Y = np.c_[a, b]
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


# ---------------------------------------------------------------- ridge, numpy, many targets at once

def _col_nanmedian(X):
    """Column medians ignoring NaN (0 for an all-NaN column); much faster than np.nanmedian here."""
    if X.shape[0] == 0:
        return np.zeros(X.shape[1])
    S = np.sort(X, 0)                                  # NaN sorts last
    c = (~np.isnan(X)).sum(0)
    lo, hi = np.maximum((c - 1) // 2, 0), np.maximum(c // 2, 0)
    cols = np.arange(X.shape[1])
    med = (S[np.minimum(lo, len(S) - 1), cols] + S[np.minimum(hi, len(S) - 1), cols]) / 2
    return np.where(c > 0, med, 0.0)


def _prep(Xtr, Xte):
    """Median imputation and standardisation fitted on training rows only (as evaluate.ridge)."""
    med = _col_nanmedian(Xtr)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _ridge_path(Ztr, Ytr, Zte, alphas):
    """Predictions on Zte for every alpha and target: (A, m, P)."""
    ybar = Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Ztr, full_matrices=False)
    UtY = U.T @ (Ytr - ybar)                                    # (r, P)
    d = s[None, :] / (s[None, :] ** 2 + alphas[:, None])         # (A, r)
    ZV = Zte @ Vt.T                                             # (m, r)
    return (ZV[None, :, :] * d[:, None, :]) @ UtY + ybar


def ridge_cv_predict(Xtr, Ytr, Xte, rng, inner=5, alphas=ALPHAS):
    """Ridge with alpha chosen per target by inner k-fold CV on the fixed grid (PREREG H2). Y is (n, P)."""
    Ytr = np.asarray(Ytr, float)
    one = Ytr.ndim == 1
    Ytr = Ytr[:, None] if one else Ytr
    n, P = Ytr.shape
    f = rng.permutation(n) % min(inner, n)
    sse = np.zeros((len(alphas), P))
    for k in range(f.max() + 1):
        tr, te = f != k, f == k
        Ztr, Zte = _prep(Xtr[tr], Xtr[te])
        sse += ((_ridge_path(Ztr, Ytr[tr], Zte, alphas) - Ytr[te][None]) ** 2).sum(1)
    best = np.argmin(sse[::-1], axis=0)
    best = len(alphas) - 1 - best                               # ties -> the larger penalty
    Ztr, Zte = _prep(Xtr, Xte)
    path = _ridge_path(Ztr, Ytr, Zte, alphas)                   # (A, m, P)
    out = path[best, :, np.arange(P)].T                         # (m, P)
    return out[:, 0] if one else out


# ---------------------------------------------------------------- the analysis data set

@dataclass
class Data:
    ids: np.ndarray
    q: np.ndarray            # (n, 5) quiz z, NaN where < 8 items
    tz: np.ndarray           # (n, 5) text-bigfive z as shipped
    tlo: np.ndarray
    thi: np.ndarray
    tlevel: np.ndarray
    head: np.ndarray         # (n, 5) raw head scores
    logw: np.ndarray
    platform: np.ndarray     # "reddit", "x", "both", ""
    has_g: np.ndarray
    has_s: np.ndarray
    G: np.ndarray            # (n, 8) Google direct features
    T: np.ndarray            # (n, n_topics) topic shares
    S: np.ndarray            # (n, 9) Spotify
    pattern: np.ndarray
    estimates: list
    mbti_p: np.ndarray       # (n, 4) P(first letter)
    oejts: np.ndarray        # (n, 4) 1/0/NaN
    basic: np.ndarray
    word_rates: np.ndarray | None
    has_wr: np.ndarray


def build_data(usable: list[Participant], cfg: Config) -> Data:
    n = len(usable)
    q = np.full((n, 5), np.nan)
    tz, tlo, thi, tlev, head = (np.full((n, 5), np.nan) for _ in range(5))
    mbti_p, oe = np.full((n, 4), np.nan), np.full((n, 4), np.nan)
    topics = sorted({k for P in usable if P.topics for k in P.topics})
    G, T, S = np.full((n, len(GOOGLE_DIRECT)), np.nan), np.full((n, len(topics)), np.nan), np.full((n, 9), np.nan)
    vocab = sorted({w for P in usable if P.word_rates for w in P.word_rates})
    WR = np.full((n, len(vocab)), np.nan) if vocab else None
    for i, P in enumerate(usable):
        z = quiz_z(P.answers, cfg.min_quiz_items)
        for j, t in enumerate(TRAITS):
            q[i, j] = z.get(t, np.nan)
        for e in P.estimates:
            if e.reader == "text-bigfive" and e.trait in TRAITS and e.usable:
                j = TRAITS.index(e.trait)
                tz[i, j], tlev[i, j] = e.value, e.interval_level if e.interval_level is not None else np.nan
                tlo[i, j] = np.nan if e.low is None else e.low
                thi[i, j] = np.nan if e.high is None else e.high
            if e.reader == "text-mbti" and e.trait in MBTI_AXES and e.usable:
                mbti_p[i, MBTI_AXES.index(e.trait)] = e.value
        for j, t in enumerate(TRAITS):
            head[i, j] = P.head.get(t, np.nan)
        if P.oejts:
            for a, v in P.oejts.items():
                oe[i, MBTI_AXES.index(a)] = float(v)
        if P.google:
            G[i] = [P.google[k] for k in GOOGLE_DIRECT]
            T[i] = [P.topics.get(k, 0.0) for k in topics]
        if P.spotify:
            S[i] = [P.spotify[k] for k in SPOTIFY_FEATURES]
        if WR is not None and P.word_rates:
            WR[i] = [P.word_rates.get(w, 0.0) for w in vocab]
    has_g = np.array([P.google is not None for P in usable])
    has_s = np.array([P.spotify is not None for P in usable])
    plat = np.array(["both" if len(P.platforms) > 1 else (P.platforms[0] if P.platforms else "") for P in usable])
    words = np.array([P.words if P.words else np.nan for P in usable], float)
    return Data(np.array([P.sid for P in usable]), q, tz, tlo, thi, tlev, head, np.log(np.maximum(words, 1)), plat,
                has_g, has_s, G, T, S, has_g.astype(int) * 2 + has_s.astype(int), [P.estimates for P in usable],
                mbti_p, oe, np.array([P.spotify_basic for P in usable]), WR,
                np.array([P.word_rates is not None for P in usable]))


def stratified_folds(pattern: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k folds, each source pattern spread evenly over folds (PREREG H2)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(pattern), int)
    off = 0
    for code in np.unique(pattern):
        idx = rng.permutation(np.nonzero(pattern == code)[0])
        fold[idx] = (np.arange(len(idx)) + off) % k
        off += len(idx)
    return fold


def google_matrix(D: Data, tr: np.ndarray) -> np.ndarray:
    """Google features for everyone: 8 direct + 2 topic PCs fitted on the training people's shares (P9)."""
    X = np.full((len(D.ids), len(GOOGLE_DIRECT) + N_TOPIC_PCS), np.nan)
    X[:, : len(GOOGLE_DIRECT)] = D.G
    fit = tr & D.has_g
    if D.T.shape[1] >= N_TOPIC_PCS and fit.sum() > N_TOPIC_PCS + 1:
        Tm = D.T[fit]
        mu = Tm.mean(0)
        _, _, Vt = np.linalg.svd(Tm - mu, full_matrices=False)
        comps = Vt[:N_TOPIC_PCS]
        comps *= np.sign(comps[np.arange(N_TOPIC_PCS), np.abs(comps).argmax(1)])[:, None]   # fixed sign
        X[D.has_g, len(GOOGLE_DIRECT):] = (D.T[D.has_g] - mu) @ comps.T
    return X


# ---------------------------------------------------------------- the combined model (H2)

def _fit_combined(tr, te, y, text, blocks, cfg, rng, want_best=False):
    """Text z stacked by OLS with inner out-of-fold predictions of a ridge per non-text source.

    tr, te: integer indices into y/text/blocks; blocks: [(name, X (n, p), has (n,))].
    Missing source: indicator + training median of its OOF prediction (PREREG section 5)."""
    ctr, cte = [np.ones(len(tr)), text[tr]], [np.ones(len(te)), text[te]]
    single_src = {}
    for name, X, has in blocks:
        htr, hte = has[tr], has[te]
        if htr.sum() < cfg.min_source_n:
            continue
        Xh, yh = X[tr][htr], y[tr][htr]
        f = rng.permutation(len(yh)) % cfg.inner
        oof = np.empty(len(yh))
        for k in range(cfg.inner):
            oof[f == k] = ridge_cv_predict(Xh[f != k], yh[f != k], Xh[f == k], rng, cfg.inner)
        pte = ridge_cv_predict(Xh, yh, X[te][hte], rng, cfg.inner) if hte.any() else np.empty(0)
        c = float(np.median(oof))
        a_tr, a_te = np.full(len(tr), c), np.full(len(te), c)
        a_tr[htr], a_te[hte] = oof, pte
        ctr += [a_tr, (~htr).astype(float)]
        cte += [a_te, (~hte).astype(float)]
        if want_best:
            single_src[name] = (r_(oof, yh), hte, pte)
    A, Ate = np.column_stack(ctr), np.column_stack(cte)
    keep = np.ones(A.shape[1], bool)
    while True:   # source weights constrained >= 0 (cfg.stack_nonneg): drop negative ones and refit
        beta = np.zeros(A.shape[1])
        beta[keep] = np.linalg.lstsq(A[:, keep], y[tr], rcond=None)[0]
        neg = [c for c in range(2, A.shape[1], 2) if keep[c] and beta[c] < 0]
        if not cfg.stack_nonneg or not neg:
            break
        keep[neg] = False
    return Ate @ beta, single_src


def _fit_single(tr, te, y, text):
    b, a = np.polyfit(text[tr], y[tr], 1)
    return a + b * text[te]


def run_cv(D: Data, j: int, cfg: Config, blocks_fn, rows: np.ndarray | None = None, intervals=True, best=True):
    """H2/H3(b)/H4 for trait j: 10 x 10-fold CV. Returns per-person repeat-averaged quantities."""
    ok = np.isfinite(D.q[:, j]) & np.isfinite(D.tz[:, j])
    if rows is not None:
        ok &= rows
    idx = np.nonzero(ok)[0]
    y, text = D.q[idx, j], D.tz[idx, j]
    n = len(idx)
    R = cfg.repeats
    single, comb, bests = np.zeros((R, n)), np.zeros((R, n)), np.zeros((R, n))
    inside, width = np.full((R, n), np.nan), np.full((R, n), np.nan)
    pattern = D.pattern[idx]
    for r in range(R):
        fold = stratified_folds(pattern, cfg.folds, seed=r)        # PREREG: seeds 0..9
        rng = np.random.default_rng([cfg.seed, 1000 + j, r])
        for k in range(cfg.folds):
            te, tr = np.nonzero(fold == k)[0], np.nonzero(fold != k)[0]
            trmask = np.zeros(len(D.ids), bool)
            trmask[idx[tr]] = True
            blocks = blocks_fn(D, trmask, idx)
            single[r, te] = _fit_single(tr, te, y, text)
            comb[r, te], src = _fit_combined(tr, te, y, text, blocks, cfg, rng, want_best=best)
            if best:   # sensitivity: best single source chosen inside the training fold by inner OOF r
                r_text = r_(_inner_oof_single(tr, y, text, cfg, rng), y[tr])
                bests[r, te] = single[r, te]
                cand = [(v[0], nm) for nm, v in src.items() if np.isfinite(v[0])]
                if cand and max(cand)[0] > r_text:
                    _, hte, pte = src[max(cand)[1]]
                    bests[r, te[hte]] = pte
            if intervals:   # H3(b): 10th/90th percentile of training-fold OOF residuals of the combined model
                fi = rng.permutation(len(tr)) % cfg.inner
                res = np.empty(len(tr))
                for kk in range(cfg.inner):
                    itr, ite = tr[fi != kk], tr[fi == kk]
                    res[fi == kk] = y[ite] - _fit_combined(itr, ite, y, text, blocks, cfg, rng)[0]
                q10, q90 = np.quantile(res, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])
                lo, hi = comb[r, te] + q10, comb[r, te] + q90
                inside[r, te] = (y[te] >= lo) & (y[te] <= hi)
                width[r, te] = hi - lo
    d = ((y - single) ** 2 - (y - comb) ** 2).mean(0)
    d_best = ((y - bests) ** 2 - (y - comb) ** 2).mean(0) if best else None
    m_skill = (np.abs(y) - np.abs(y - comb)).mean(0)
    return {"idx": idx, "y": y, "text": text, "single": single.mean(0), "comb": comb.mean(0), "single_R": single, "comb_R": comb,
            "d": d, "d_best": d_best,
            "m": m_skill, "abs_err_comb": np.abs(y - comb).mean(0), "abs_err_single": np.abs(y - single).mean(0),
            "inside": np.nanmean(inside, 0) if intervals else None, "width": np.nanmean(width, 0) if intervals else None}


def _inner_oof_single(tr, y, text, cfg, rng):
    f = rng.permutation(len(tr)) % cfg.inner
    out = np.empty(len(tr))
    for k in range(cfg.inner):
        a, b = tr[f != k], tr[f == k]
        out[f == k] = _fit_single(a, b, y, text)
    return out


def blocks_all(D: Data, trmask, idx):
    return [("google", google_matrix(D, trmask)[idx], D.has_g[idx]), ("spotify", D.S[idx], D.has_s[idx])]


def blocks_only(name):
    def f(D, trmask, idx):
        return [b for b in blocks_all(D, trmask, idx) if b[0] == name]
    return f


# ---------------------------------------------------------------- analyses

def h1(D: Data, cfg: Config) -> dict:
    rows = []
    rng = np.random.default_rng([cfg.seed, 1])
    for j, t in enumerate(TRAITS):
        ok = np.isfinite(D.q[:, j]) & np.isfinite(D.tz[:, j])
        x, y = D.tz[ok, j], D.q[ok, j]
        n = int(ok.sum())
        if n < 10:
            rows.append({"trait": t, "n": n, "skipped": f"text reader usable for only {n} people"})
            continue
        r = r_(x, y)
        bs = wr(x, y, _boot_weights(n, cfg.boots, rng))
        rows.append({"trait": t, "n": n, "r": r, "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                     "p": fisher_p_greater(r, n), "mde_r_80pct": mde_r(n)})
    holm_into(rows)
    return {"family": "H1, Holm over 5 traits", "traits": rows}


def h2_h3b_h4(D: Data, cfg: Config) -> tuple[dict, dict, dict, dict]:
    rng = np.random.default_rng([cfg.seed, 2])
    n = len(D.ids)
    avail = np.isfinite(D.q) & np.isfinite(D.tz)
    pool = pool_traits(avail, n, cfg.pool_min_share)
    per = {t: run_cv(D, j, cfg, blocks_all) for j, t in enumerate(TRAITS) if avail[:, j].sum() >= cfg.min_trait_n}
    skipped = {t: f"text reader usable for {int(avail[:, j].sum())} people (< {cfg.min_trait_n}); not analysed"
               for j, t in enumerate(TRAITS) if t not in per}
    Dsum, Msum, have = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    trows, drs = [], []
    B = cfg.boots
    W = _boot_weights(n, B, rng)
    dr_boot = np.zeros(B)
    for t in TRAITS:
        if t not in per:
            trows.append({"trait": t, "n": int(avail[:, TRAITS.index(t)].sum()), "skipped": skipped[t]})
            continue
        o = per[t]
        pooled = t in pool["traits"]
        if pooled:
            Dsum[o["idx"]] += o["d"]
            Msum[o["idx"]] += o["m"]
            have[o["idx"]] = True
        pt = paired_test(o["d"], cfg.flips, rng)
        w = W[:, o["idx"]]
        rc, rs = rep_r(o["comb_R"], o["y"]), rep_r(o["single_R"], o["y"])
        bs = rep_wr(o["comb_R"], o["y"], w) - rep_wr(o["single_R"], o["y"], w)
        if pooled:
            dr_boot += bs / len(pool["traits"])
            drs.append(rc - rs)
        trows.append({"trait": t, **pt, "in_pool": pooled, "r_combined": rc, "r_single": rs, "delta_r": rc - rs,
                      "delta_r_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                      "best_single_in_cv": {"mean_d": float(np.mean(o["d_best"])), "ci95": _tci(o["d_best"])}})
    holm_into(trows)
    primary = paired_test(Dsum[have], cfg.flips, rng)
    primary["supported"] = bool(np.isfinite(primary["p"]) and primary["p"] < 0.05)
    h2 = {"pool": pool, "primary_pooled": primary, "traits": trows, "family": "H2 per trait, Holm over the traits analysed",
          "delta_r_pooled_mean": float(np.mean(drs)) if drs else None,
          "delta_r_pooled_ci95": [float(np.percentile(dr_boot, 2.5)), float(np.percentile(dr_boot, 97.5))] if drs else None,
          "note": "best single source fixed in advance as text; best-single-in-CV is a sensitivity check, not a test"}
    # H3(b), pooled over the same traits
    pin = [per[t] for t in pool["traits"] if t in per]
    h3b = coverage_stats(np.concatenate([o["inside"] for o in pin]) if pin else np.array([]),
                         np.concatenate([o["width"] for o in pin]) if pin else np.array([]),
                         np.concatenate([o["idx"] for o in pin]) if pin else np.array([]), cfg.boots, rng)
    h3b["pooled_traits"] = pool["traits"]
    h3b["traits"] = [{"trait": t, "n": len(o["idx"]), "in_pool": t in pool["traits"], "coverage": float(np.nanmean(o["inside"])),
                      "mean_width": float(np.nanmean(o["width"]))} for t, o in per.items()]
    # H4
    h4p = paired_test(Msum[have], cfg.flips, rng)
    h4p["supported"] = bool(np.isfinite(h4p["p"]) and h4p["p"] < 0.05)
    h4rows = []
    for t, o in per.items():
        w = W[:, o["idx"]]
        base = np.abs(o["y"])
        sk = {}
        for name, err in (("combined", o["abs_err_comb"]), ("single_calibrated", o["abs_err_single"]),
                          ("text_as_shipped", np.abs(o["y"] - o["text"]))):
            v = 1 - err.mean() / base.mean()
            bs = 1 - (w @ err) / (w @ base)
            sk[name] = {"skill": float(v), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        h4rows.append({"trait": t, "n": len(o["idx"]), "in_pool": t in pool["traits"], **sk})
    ts = np.zeros(n)
    for t in pool["traits"]:
        j = TRAITS.index(t)
        ok = avail[:, j]
        ts[ok] += np.abs(D.q[ok, j]) - np.abs(D.q[ok, j] - D.tz[ok, j])
    h4 = {"pooled_traits": pool["traits"], "primary_pooled_combined": h4p,
          "text_as_shipped_pooled": paired_test(ts[have], cfg.flips, rng), "traits": h4rows}
    return h2, h3b, h4, per


def _tci(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 3:
        return [float("nan")] * 2
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
    return [float(d.mean() - h), float(d.mean() + h)]


def h2_pairs(D: Data, cfg: Config) -> dict:
    out = {}
    rng = np.random.default_rng([cfg.seed, 3])
    for name, has in (("text+google", D.has_g), ("text+spotify", D.has_s)):
        rows = []
        for j, t in enumerate(TRAITS):
            nav = int((has & np.isfinite(D.q[:, j]) & np.isfinite(D.tz[:, j])).sum())
            if has.sum() < 2 * cfg.min_source_n or nav < cfg.min_trait_n:
                rows.append({"trait": t, "n": nav, "skipped": "too few people with this source and a usable text estimate"})
                continue
            o = run_cv(D, j, cfg, blocks_only(name.split("+")[1]), rows=has, intervals=False, best=False)
            pt = paired_test(o["d"], cfg.flips, rng)
            rows.append({"trait": t, **pt, "delta_r": rep_r(o["comb_R"], o["y"]) - rep_r(o["single_R"], o["y"])})
        holm_into(rows)
        out[name] = {"family": f"H2 sub-analysis {name}, people with both, Holm over 5", "traits": rows}
    return out


def h3a(D: Data, cfg: Config) -> dict:
    rng = np.random.default_rng([cfg.seed, 4])
    n = len(D.ids)
    ok = np.isfinite(D.q) & np.isfinite(D.tlo) & np.isfinite(D.thi)
    inside = np.where(ok, (D.q >= D.tlo) & (D.q <= D.thi), np.nan)
    width = np.where(ok, D.thi - D.tlo, np.nan)
    pid = np.repeat(np.arange(n)[:, None], 5, 1)
    levels = sorted({float(x) for x in D.tlevel[ok] if np.isfinite(x)})
    pool = pool_traits(ok, n, cfg.pool_min_share)
    J = [TRAITS.index(t) for t in pool["traits"]]
    out = coverage_stats(inside[:, J].ravel(), width[:, J].ravel(), pid[:, J].ravel(), cfg.boots, rng)
    out["pooled_traits"] = pool["traits"]
    out["stated_levels_seen"] = levels
    out["traits"] = [{"trait": t, "n": int(ok[:, j].sum()), "in_pool": t in pool["traits"],
                      "coverage": float(np.nanmean(inside[:, j])) if ok[:, j].any() else None,
                      "mean_width": float(np.nanmean(width[:, j])) if ok[:, j].any() else None} for j, t in enumerate(TRAITS)]
    lw = D.logw
    cuts = np.nanpercentile(lw, [100 / 3, 200 / 3])
    terc = np.digitize(lw, cuts)
    out["by_word_tercile"] = [{"tercile": k + 1, "n_people": int((terc == k).sum()),
                               "words_range": [float(np.exp(np.nanmin(lw[terc == k]))), float(np.exp(np.nanmax(lw[terc == k])))] if (terc == k).any() else None,
                               "coverage": float(np.nanmean(inside[terc == k][:, J]))} for k in range(3)]
    return out


def s1(D: Data, cfg: Config) -> dict:
    out = {}
    for name, has, Xfn in (("google", D.has_g, lambda tr: google_matrix(D, tr)), ("spotify", D.has_s, lambda tr: D.S)):
        rows = []
        for j, t in enumerate(TRAITS):
            ok = has & np.isfinite(D.q[:, j])
            idx = np.nonzero(ok)[0]
            n = len(idx)
            if n < 2 * cfg.min_source_n:
                rows.append({"trait": t, "n": n, "skipped": "too few people"})
                continue
            y = D.q[idx, j]
            rng = np.random.default_rng([cfg.seed, 5, j, len(name)])
            Y = np.c_[y, np.stack([rng.permutation(y) for _ in range(cfg.perms)], 1)]
            pred = np.zeros((cfg.repeats, n))
            null_pred = np.zeros((n, cfg.perms))
            for r in range(cfg.repeats):
                fold = np.random.default_rng(r).permutation(n) % cfg.folds
                for k in range(cfg.folds):
                    te, tr = fold == k, fold != k
                    trmask = np.zeros(len(D.ids), bool)
                    trmask[idx[tr]] = True
                    X = Xfn(trmask)[idx]
                    if r == 0:   # permutations: 1 repeat, as evaluate.predict
                        p = ridge_cv_predict(X[tr], Y[tr], X[te], rng, cfg.inner)
                        pred[r, te] = p[:, 0]
                        null_pred[te] = p[:, 1:]
                    else:
                        pred[r, te] = ridge_cv_predict(X[tr], y[tr], X[te], rng, cfg.inner)
            r = rep_r(pred, y)
            null = np.array([r_(null_pred[:, b], Y[:, b + 1]) for b in range(cfg.perms)])
            null = np.where(np.isfinite(null), null, 0.0)
            bs = rep_wr(pred, y, _boot_weights(n, cfg.boots, rng))
            rows.append({"trait": t, "n": n, "r": r, "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                         "p": float((1 + np.sum(null >= r)) / (1 + cfg.perms)), "null_mean": float(null.mean())})
        holm_into(rows)
        out[name] = {"family": f"S1 {name} alone, Holm over 5; read r against null_mean", "traits": rows,
                     "n_basic_export": int(D.basic[has].sum()) if name == "spotify" else None}
    return out


def s2(D: Data, cfg: Config) -> dict:
    rows = []
    for a, ax in enumerate(MBTI_AXES):
        ok = np.isfinite(D.oejts[:, a]) & np.isfinite(D.mbti_p[:, a])
        pos, neg = D.mbti_p[ok & (D.oejts[:, a] == 1), a], D.mbti_p[ok & (D.oejts[:, a] == 0), a]
        if len(pos) < 5 or len(neg) < 5:
            rows.append({"axis": ax, "n": int(ok.sum()), "skipped": "too few of one letter"})
            continue
        auc, var = delong_auc(pos, neg)
        se = math.sqrt(max(var, 1e-12))
        rows.append({"axis": ax, "n": int(ok.sum()), "n_first_letter": len(pos), "auc": auc,
                     "ci95": [auc - 1.96 * se, auc + 1.96 * se], "p": float(stats.norm.sf((auc - 0.5) / se))})
    holm_into(rows)
    return {"family": "S2, Holm over 4 letters (one-sided AUC > 0.5)", "axes": rows}


def s3(D: Data, cfg: Config) -> dict:
    ys, Xs, gs = [], [], []
    lw_c = D.logw - np.nanmean(D.logw)
    for j in range(5):
        ok = np.isfinite(D.q[:, j]) & np.isfinite(D.tz[:, j]) & np.isfinite(lw_c)
        fe = np.zeros((ok.sum(), 5))
        fe[:, j] = 1
        Xs.append(np.c_[fe, D.tz[ok, j], lw_c[ok], D.tz[ok, j] * lw_c[ok]])
        ys.append(D.q[ok, j])
        gs.append(np.nonzero(ok)[0])
    y, X, g = np.concatenate(ys), np.vstack(Xs), np.concatenate(gs)
    beta, se, df = cluster_ols(y, X, g)
    t = beta[-1] / se[-1]
    return {"n_person_traits": int(len(y)), "n_people": int(len(np.unique(g))), "beta_reader_z": float(beta[5]),
            "beta_log_words": float(beta[6]), "beta_product": float(beta[7]), "se_product": float(se[7]),
            "ci95_product": [float(beta[7] - stats.t.ppf(0.975, df) * se[7]), float(beta[7] + stats.t.ppf(0.975, df) * se[7])],
            "p_one_sided": float(stats.t.sf(t, df)), "supported": bool(stats.t.sf(t, df) < 0.05),
            "model": "q ~ trait FE + z + logw_c + z*logw_c, CR1 person-clustered SEs"}


def s4(D: Data, cfg: Config) -> dict:
    rng = np.random.default_rng([cfg.seed, 6])
    out = {}
    for label, rowsel, Xfn, alt in (
            ("calibration_refit_all", np.ones(len(D.ids), bool), lambda j: D.head[:, [j]], "greater"),
            ("weights_refit_word_subset", D.has_wr, lambda j: D.word_rates, "two-sided")):
        rows = []
        for j, t in enumerate(TRAITS):
            X = Xfn(j)
            okq = rowsel & np.isfinite(D.q[:, j])
            shipped_share = float(np.isfinite(D.tz[okq, j]).mean()) if okq.any() else 0.0
            # P22: where the reader refuses the trait for most people, the shipped answer is "no number",
            # i.e. the population average, so the refit is compared with z = 0 for everyone instead.
            vs_zero = shipped_share < cfg.pool_min_share
            ok = okq if vs_zero else okq & np.isfinite(D.tz[:, j])
            if X is None or ok.sum() < 30 or (X.ndim == 2 and X.shape[1] == 0):
                rows.append({"trait": t, "n": int(ok.sum()), "skipped": "fewer than 30 people (or no word block)"})
                continue
            idx = np.nonzero(ok)[0]
            Xi = X[idx]
            if label.startswith("calibration") and not np.isfinite(Xi).all():
                keep = np.isfinite(Xi[:, 0])
                idx, Xi = idx[keep], Xi[keep]
            y = D.q[idx, j]
            shipped = np.zeros(len(idx)) if vs_zero else D.tz[idx, j]
            pred = np.zeros((cfg.repeats, len(idx)))
            for r in range(cfg.repeats):
                fold = stratified_folds(D.pattern[idx], cfg.folds, seed=r)
                for k in range(cfg.folds):
                    te, tr = fold == k, fold != k
                    if label.startswith("calibration"):
                        b, a = np.polyfit(Xi[tr, 0], y[tr], 1)
                        pred[r, te] = a + b * Xi[te, 0]
                    else:
                        pred[r, te] = ridge_cv_predict(Xi[tr], y[tr], Xi[te], rng, cfg.inner)
            d = ((y - shipped) ** 2 - (y - pred) ** 2).mean(0)
            pt = paired_test(d, cfg.flips, rng, alternative=alt)
            rows.append({"trait": t, **pt, "r_refit": rep_r(pred, y), "r_shipped": None if vs_zero else r_(shipped, y),
                         "compared_with": (f"z = 0 (shipped reader gives a number for only {shipped_share:.0%})" if vs_zero
                                           else "shipped z")})
        holm_into(rows)
        out[label] = {"family": f"S4 {label}, Holm over 5 ({alt})", "traits": rows}
    return out


def s5(people: list[Participant], flow_log: list[dict], cfg: Config) -> dict:
    s1_fail = {e["study_id"] for e in flow_log if e["rule"] in ("1", "2", "3", "4", "7")}
    sel, why = [], {"assigned": 0, "completed": 0, "excluded_session1_rules": 0, "retest_quality": 0, "outside_window": 0}
    for P in people:
        if not P.retest_assigned:
            continue
        why["assigned"] += 1
        if P.retest_answers is None:
            continue
        why["completed"] += 1
        if P.sid in s1_fail:
            why["excluded_session1_rules"] += 1
            continue
        if quiz_quality(P.retest_answers, P.retest_item_ms or {}):
            why["retest_quality"] += 1
            continue
        lo, hi = cfg.retest_window
        if P.retest_days is None or not (lo <= P.retest_days <= hi + 1):   # day 20 inclusive (fractional days)
            why["outside_window"] += 1
            continue
        sel.append(P)
    rng = np.random.default_rng([cfg.seed, 7])
    rows = []
    for t in TRAITS:
        a, b = [], []
        for P in sel:
            z1, z2 = quiz_z(P.answers, cfg.min_quiz_items), quiz_z(P.retest_answers, cfg.min_quiz_items)
            if t in z1 and t in z2:
                a.append(z1[t])
                b.append(z2[t])
        a, b = np.array(a), np.array(b)
        n = len(a)
        if n < 5:
            rows.append({"trait": t, "n": n, "skipped": "fewer than 5"})
            continue
        r = r_(a, b)
        ib = np.array([icc21(a[i], b[i]) for i in rng.integers(0, n, (cfg.boots, n))])
        rows.append({"trait": t, "n": n, "r": r, "r_ci95": list(fisher_ci(r, n)), "icc21": icc21(a, b),
                     "icc21_ci95": [float(np.nanpercentile(ib, 2.5)), float(np.nanpercentile(ib, 97.5))],
                     "validity_ceiling_sqrt_r": float(math.sqrt(max(r, 0)))})
    return {"selection": why, "n_used": len(sel), "traits": rows, "note": "no hypothesis (PREREG S5)"}


def s6(D: Data, cfg: Config) -> dict:
    rng = np.random.default_rng([cfg.seed, 8])
    rows = []
    for j, t in enumerate(TRAITS):
        ok = np.isfinite(D.q[:, j]) & np.isfinite(D.tz[:, j])
        g = {p: ok & (D.platform == p) for p in ("reddit", "x")}
        n1, n2 = int(g["reddit"].sum()), int(g["x"].sum())
        if n1 < 10 or n2 < 10:
            rows.append({"trait": t, "n_reddit": n1, "n_x": n2, "skipped": "fewer than 10 in a group"})
            continue
        r1, r2 = r_(D.tz[g["reddit"], j], D.q[g["reddit"], j]), r_(D.tz[g["x"], j], D.q[g["x"], j])
        b1 = wr(D.tz[g["reddit"], j], D.q[g["reddit"], j], _boot_weights(n1, cfg.boots, rng))
        b2 = wr(D.tz[g["x"], j], D.q[g["x"], j], _boot_weights(n2, cfg.boots, rng))
        z = (np.arctanh(r1) - np.arctanh(r2)) / math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        rows.append({"trait": t, "n_reddit": n1, "n_x": n2, "r_reddit": r1, "r_x": r2, "diff": r1 - r2,
                     "ci95": [float(np.nanpercentile(b1 - b2, 2.5)), float(np.nanpercentile(b1 - b2, 97.5))],
                     "p": float(2 * stats.norm.sf(abs(z)))})
    holm_into(rows)
    return {"family": "S6, two-sided, exploratory, Holm over 5", "traits": rows}


# ---------------------------------------------------------------- S7 fusion (added 2026-11-01)

def _readings(D: Data):
    return [readings_from_estimates(e)[0] for e in D.estimates]


def _fusion_metrics(name, D, mu, lo, hi, cfg, rng, compare_to, compare_label, mu_reps=None, inside=None):
    """Validity, coverage, skill and paired error for one fusion method. Pooled tests run over the traits
    the method speaks for in >= 90% of people (P22); the paired error over traits both methods speak for.
    inside: optional (n, 5) per person-trait coverage already averaged over CV repeats (P11)."""
    n = len(D.ids)
    W = _boot_weights(n, cfg.boots, rng)
    okall = np.isfinite(D.q) & np.isfinite(mu)
    if inside is None:
        inside = np.where(okall, (D.q >= lo) & (D.q <= hi), np.nan)
    width = np.where(okall, hi - lo, np.nan)
    pool = pool_traits(okall, n, cfg.pool_min_share)
    pool_cmp = pool_traits(okall & np.isfinite(compare_to), n, cfg.pool_min_share)
    rows = []
    Dsum, Msum, haveD, haveM = np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    for j, t in enumerate(TRAITS):
        ok = okall[:, j]
        if ok.sum() < 10:
            rows.append({"trait": t, "n": int(ok.sum()), "skipped": "fewer than 10 people with this estimate"})
            continue
        x, y = mu[ok, j], D.q[ok, j]
        if mu_reps is None:
            r, bs = r_(x, y), wr(x, y, W[:, ok])
        else:   # learned in CV: mean of each repeat's r (P21)
            r, bs = rep_r(mu_reps[:, ok, j], y), rep_wr(mu_reps[:, ok, j], y, W[:, ok])
        base, err = np.abs(y), np.abs(y - x)
        sk_bs = 1 - (W[:, ok] @ err) / (W[:, ok] @ base)
        rows.append({"trait": t, "n": int(ok.sum()), "in_pool": t in pool["traits"], "r": r,
                     "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                     "p": fisher_p_greater(r, int(ok.sum())), "skill": float(1 - err.mean() / base.mean()),
                     "skill_ci95": [float(np.percentile(sk_bs, 2.5)), float(np.percentile(sk_bs, 97.5))],
                     "coverage": float(np.nanmean(inside[ok, j])), "mean_width": float(np.mean(width[ok, j]))})
        if t in pool["traits"]:
            Msum[ok] += base - err
            haveM |= ok
        if t in pool_cmp["traits"]:
            okc = ok & np.isfinite(compare_to[:, j])
            Dsum[okc] += (D.q[okc, j] - compare_to[okc, j]) ** 2 - (D.q[okc, j] - mu[okc, j]) ** 2
            haveD |= okc
    holm_into(rows)
    J = [TRAITS.index(t) for t in pool["traits"]]
    pid = np.repeat(np.arange(n)[:, None], 5, 1)
    return {"family": f"S7 {name} validity, Holm over the traits analysed (one-sided r > 0)", "traits": rows,
            "pool": pool, "pool_for_paired_error": pool_cmp["traits"],
            "coverage_pooled": coverage_stats(inside[:, J].ravel(), width[:, J].ravel(), pid[:, J].ravel(), cfg.boots, rng),
            "skill_pooled_test": paired_test(Msum[haveM], cfg.flips, rng),
            f"paired_error_vs_{compare_label}": paired_test(Dsum[haveD], cfg.flips, rng)}


def s7(D: Data, cfg: Config, per_h2: dict) -> dict:
    rng = np.random.default_rng([cfg.seed, 9])
    n = len(D.ids)
    readings = _readings(D)
    # (a) bayes, zero data: exactly what FusedReader shows a person
    fuser = Fuser("bayes", level=LEVEL)
    mu, lo, hi = (np.full((n, 5), np.nan) for _ in range(3))
    for i, rs in enumerate(readings):
        for t, f in fuser.predict(rs).items():
            j = TRAITS.index(t)
            mu[i, j], lo[i, j], hi[i, j] = f.value, f.low, f.high
    out = {"bayes": _fusion_metrics("bayes (n = 0, as shipped)", D, mu, lo, hi, cfg, rng, D.tz, "text_as_shipped")}
    # (b) bayes_eb_cp fitted inside the same CV
    if n >= cfg.eb_min_n:
        single = np.full((n, 5), np.nan)
        for j, t in enumerate(TRAITS):
            if t in (per_h2 or {}):
                o = per_h2[t]
                single[o["idx"], j] = o["single"]
        mu2, lo2, hi2 = (np.full((cfg.repeats, n, 5), np.nan) for _ in range(3))
        for j, t in enumerate(TRAITS):
            has_t = np.array([any(r.trait == t for r in rs) for rs in readings])
            idx = np.nonzero(np.isfinite(D.q[:, j]) & has_t)[0]
            if len(idx) < cfg.min_trait_n:
                continue
            chans = sorted({r.channel for i in idx for r in readings[i] if r.trait == t})
            groups = [next(r.group for i in idx for r in readings[i] if r.trait == t and r.channel == c) for c in chans]
            U, V, Vsd = Fuser._arrays([readings[i] for i in idx], t, chans)
            y = D.q[idx, j]
            for r in range(cfg.repeats):
                fold = stratified_folds(D.pattern[idx], cfg.folds, seed=r)
                for k in range(cfg.folds):
                    te, tr = fold == k, fold != k
                    m = make_method("bayes_eb_cp", int(tr.sum()), LEVEL, seed=cfg.seed).fit(U[tr], y[tr], V[tr], Vsd[tr], groups)
                    a, b, c = m.predict(U[te], V[te], Vsd[te])
                    mu2[r, idx[te], j], lo2[r, idx[te], j], hi2[r, idx[te], j] = a, b, c
        with np.errstate(invalid="ignore"):
            ins = np.where(np.isfinite(mu2), (D.q[None] >= lo2) & (D.q[None] <= hi2), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN person-traits (no reading): stay NaN
            res = _fusion_metrics("bayes_eb_cp (fitted in CV)", D, np.nanmean(mu2, 0), np.nanmean(lo2, 0),
                                  np.nanmean(hi2, 0), cfg, rng, single, "text_single_calibrated",
                                  mu_reps=mu2, inside=np.nanmean(ins, 0))
        res["label"] = "registered secondary" if n >= cfg.eb_confirm_n else f"exploratory (n = {n} < {cfg.eb_confirm_n}, fusion.md)"
        out["bayes_eb_cp"] = res
    else:
        out["bayes_eb_cp"] = {"skipped": f"n = {n} < {cfg.eb_min_n}"}
    out["checks"] = fusion_checks(D, readings, cfg, rng)
    return out


def fusion_checks(D: Data, readings, cfg, rng) -> dict:
    """docs/fusion.md pilot checks: each channel's real validity vs stated, and the text/MBTI error correlation."""
    n = len(D.ids)
    W = _boot_weights(n, cfg.boots, rng)
    chans = []
    for j, t in enumerate(TRAITS):
        by: dict[str, list] = {}
        for i, rs in enumerate(readings):
            if not np.isfinite(D.q[i, j]):
                continue
            for r in rs:
                if r.trait == t:
                    by.setdefault(r.channel, []).append((i, r.u, r.validity))
        u_by = {}
        for c, lst in sorted(by.items()):
            i = np.array([a for a, _, _ in lst])
            u = np.array([b for _, b, _ in lst])
            v = np.array([x for _, _, x in lst])
            if len(i) < 10:
                chans.append({"trait": t, "channel": c, "n": len(i), "skipped": "fewer than 10 people"})
                continue
            rr = r_(u, D.q[i, j])
            bs = wr(u, D.q[i, j], W[:, i])
            chans.append({"trait": t, "channel": c, "n": len(i), "real_r": rr,
                          "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                          "stated_median": float(np.median(v)), "stated_x0.75": float(np.median(v) * 0.75)})
            u_by[c] = (i, u, rr)
        if len(u_by) == 2:   # text-bigfive and text-mbti:<axis> for the same trait
            (c1, (i1, u1, v1)), (c2, (i2, u2, v2)) = sorted(u_by.items())
            common = np.intersect1d(i1, i2)
            if len(common) < 10:
                continue
            a = u1[np.searchsorted(i1, common)]
            b = u2[np.searchsorted(i2, common)]
            y = D.q[common, j]
            ea = (a - v1 * y) / math.sqrt(max(1 - v1 ** 2, 1e-6))
            eb = (b - v2 * y) / math.sqrt(max(1 - v2 ** 2, 1e-6))
            rho = r_(ea, eb)
            chans.append({"trait": t, "channel": f"error corr {c1} x {c2}", "n": len(common), "real_r": rho,
                          "ci95": list(fisher_ci(rho, len(common))), "stated_median": 0.30, "stated_x0.75": None})
    return {"note": "estimates only (fusion.md pilot checks); prior error corr within text = 0.30", "rows": chans}


# ---------------------------------------------------------------- driver

def analyse(folder: str | Path, cfg: Config, only: set | None = None) -> dict:
    people, meta = load_study(folder)
    usable, flow, log = apply_exclusions(people, meta)
    D = build_data(usable, cfg)
    res = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
           "input": {"folder_digest": _digest(folder), "n_session1": len(people), "orphan_payloads": len(meta.get("orphan_payloads", []))},
           "exclusions": {"flow": flow, "log": log},
           "sample": {"n_usable": len(usable), "n_google": int(D.has_g.sum()), "n_spotify": int(D.has_s.sum()),
                      "n_both": int((D.has_g & D.has_s).sum()), "n_word_block": int(D.has_wr.sum()),
                      "n_oejts": int(np.isfinite(D.oejts).any(1).sum()),
                      "quiz_traits_missing": {t: int(np.isnan(D.q[:, j]).sum()) for j, t in enumerate(TRAITS)},
                      "text_reader_usable": {t: int(np.isfinite(D.tz[:, j]).sum()) for j, t in enumerate(TRAITS)},
                      "text_pool": pool_traits(np.isfinite(D.q) & np.isfinite(D.tz), len(usable), cfg.pool_min_share),
                      "platform": {p: int((D.platform == p).sum()) for p in ("reddit", "x", "both")}}}
    want = (lambda k: only is None or k in only)
    if len(usable) < 20:
        res["error"] = f"only {len(usable)} usable participants; nothing analysed"
        return res
    if want("H1"):
        res["H1"] = h1(D, cfg)
    per = None
    if want("H2") or want("H3") or want("H4") or want("S7"):
        res["H2"], res["H3b"], res["H4"], per = h2_h3b_h4(D, cfg)
        res["H2"]["pairs"] = h2_pairs(D, cfg) if want("H2") else None
    if want("H3"):
        res["H3a"] = h3a(D, cfg)
    if want("S1"):
        res["S1"] = s1(D, cfg)
    if want("S2"):
        res["S2"] = s2(D, cfg)
    if want("S3"):
        res["S3"] = s3(D, cfg)
    if want("S4"):
        res["S4"] = s4(D, cfg)
    if want("S5"):
        res["S5"] = s5(people, log, cfg)
    if want("S6"):
        res["S6"] = s6(D, cfg)
    if want("S7"):
        res["S7"] = s7(D, cfg, per)
    return _clean(res)


def _digest(folder) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(folder).rglob("*.json")):
        rel = f.relative_to(folder).as_posix()
        if rel.startswith("results/"):
            continue
        h.update(rel.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _clean(o):
    """JSON-safe, rounded floats (determinism across platforms at the 1e-9 level)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if not math.isfinite(x) else round(x, 9)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


# ---------------------------------------------------------------- plain-text report

def _f(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}" if isinstance(x, float) else str(x)


def _ci(c, d=3):
    return "n/a" if not c or c[0] is None else f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def _p(p):
    return "n/a" if p is None else ("<0.0001" if p < 1e-4 else f"{p:.4f}")


def _h2_mde(n: int) -> str:
    """POWER.md section 9 (measured text r about 0.10) in one line, for 'not detected at this n'."""
    if n < 150:
        return "nothing (Delta r 0.10: power 0.03 per trait, 0.02 omnibus at n = 60)"
    if n < 400:
        return "almost nothing: Delta r 0.10 has power 0.11 per trait and 0.11 omnibus at n = 250 (measured r1 0.10)"
    if n < 900:
        return "Delta r 0.10 on average (omnibus power 0.83 at 600); per trait 0.43; Delta r 0.05 not (0.06)"
    return "Delta r 0.10 per trait becomes likely; Delta r 0.05 still needs more (POWER.md section 9)"


def report(res: dict) -> str:
    L = []
    w = L.append
    w("AAS whole-person study: pre-registered analysis (scripts/study_analysis.py)")
    w(f"input digest {res['input']['folder_digest'][:16]}  seed {res['config']['seed']}  repeats {res['config']['repeats']}")
    w("")
    w("EXCLUSIONS (PREREG section 7, applied in order)")
    for r in res["exclusions"]["flow"]:
        w(f"  {r['rule']:<3} {r['label']:<86} -{r['excluded']:<4} {r['remaining']}")
    s = res["sample"]
    w(f"  usable n = {s['n_usable']}; Google {s['n_google']}, Spotify {s['n_spotify']}, both {s['n_both']}; "
      f"word block {s['n_word_block']}; OEJTS {s['n_oejts']}; platforms {s['platform']}")
    w(f"  quiz traits missing (< 8 of 10 items): {s['quiz_traits_missing']}")
    w(f"  text reader usable (a number, not refused) per trait: {s['text_reader_usable']}")
    tp = s["text_pool"]
    w(f"  POOLED TESTS over {tp['traits']} (P22: the text reader speaks for >= 90% of people). Left out: "
      + ("; ".join(f"{t}: {why}" for t, why in tp["excluded"].items()) or "none"))
    w(f"  people with all pooled traits {tp['people_with_all_pooled_traits']}, some {tp['people_with_some_pooled_traits']}, none {tp['people_with_none']}")
    if "error" in res:
        w(res["error"])
        return "\n".join(L)

    def fam(title, rows, key="trait", cols=("r", "ci95", "p", "p_holm")):
        w(title)
        for r in rows:
            if "skipped" in r:
                w(f"  {r.get(key)}: skipped ({r['skipped']}, n {r.get('n')})")
                continue
            parts = [f"{r.get(key)}", f"n {r.get('n', r.get('n_reddit', ''))}"]
            for c in cols:
                v = r.get(c)
                parts.append(f"{c} {_ci(v) if isinstance(v, list) else _p(v) if c.startswith('p') else _f(v)}")
            parts.append("SUPPORTED" if r.get("supported") else "not detected at this n")
            w("  " + "  ".join(parts))

    if "H1" in res:
        fam("\nH1  text reader validity (no fitting): Pearson r, one-sided Fisher p, 95% person bootstrap CI, Holm/5",
            res["H1"]["traits"], cols=("r", "ci95", "p", "p_holm", "mde_r_80pct"))
    if "H2" in res:
        h = res["H2"]
        p = h["primary_pooled"]
        w("\nH2  increment, combined (text + learned Google/Spotify, stacked) vs text calibrated, 10 x 10-fold CV")
        w(f"  PRIMARY pooled D_i = sum_t d_it: n {p['n']}, mean {_f(p['mean'], 4)} {_ci(p['ci95'], 4)}, p_t {_p(p['p_t'])}, "
          f"p_perm {_p(p['p_perm'])} -> p {_p(p['p'])} ({p['p_source']}): {'SUPPORTED' if p['supported'] else 'not detected at this n'}")
        fam("  per trait (mean d_it, Holm/5):", h["traits"], cols=("mean", "ci95", "p", "p_holm", "delta_r", "delta_r_ci95"))
        w(f"  Delta r pooled mean over {h['pool']['traits']}: {_f(h['delta_r_pooled_mean'])} {_ci(h['delta_r_pooled_ci95'])}")
        w(f"  detectable at n = {p['n']} (POWER.md section 3): {_h2_mde(p['n'])}")
        w("  sensitivity, best single source chosen in CV (mean d vs combined): " +
          ", ".join(f"{r['trait']} {_f(r['best_single_in_cv']['mean_d'], 4)}" for r in h["traits"] if "best_single_in_cv" in r))
        for name, sub in (h.get("pairs") or {}).items():
            fam(f"  sub-analysis {name} (people with both):", sub["traits"], cols=("mean", "ci95", "p", "p_holm", "delta_r"))
    if "H3a" in res:
        for lab, c in (("H3(a) text reader's shipped 80% intervals", res["H3a"]), ("H3(b) combined model's in-CV 80% intervals", res["H3b"])):
            w(f"\n{lab}: coverage {c['coverage']:.3f}, 90% cluster CI [{c['ci90'][0]:.3f}, {c['ci90'][1]:.3f}], "
              f"mean width {c['mean_width']:.2f} z; equivalence to 0.80 +- 0.05: {'SUPPORTED' if c['equivalent_0.75_0.85'] else 'not supported'}")
            w(f"  pooled over {c.get('pooled_traits')}; per trait: " + ", ".join(
                f"{r['trait']} {r['coverage']:.3f} (w {r['mean_width']:.2f}, n {r['n']})" if r.get("coverage") is not None
                else f"{r['trait']} n/a (n {r['n']})" for r in c["traits"]))
        w("  text coverage by word tercile: " + ", ".join(f"T{r['tercile']} {r['coverage']:.3f} (n {r['n_people']})" for r in res["H3a"]["by_word_tercile"]))
    if "H4" in res:
        h = res["H4"]
        p = h["primary_pooled_combined"]
        w(f"\nH4  skill vs guessing z = 0, pooled over {h['pooled_traits']}. PRIMARY pooled M_i: mean {_f(p['mean'], 4)} {_ci(p['ci95'], 4)}, p {_p(p['p'])} "
          f"({p['p_source']}): {'SUPPORTED' if p['supported'] else 'not detected at this n'}")
        w("  detectable (POWER.md section 9): at r 0.10 the best possible skill is 0.005 (power 0.16 at n = 250, 0.25 at 600); "
          "at r 0.16, 0.013 (0.29 and 0.49). Pooled over traits is stronger but still weak")
        tp = h["text_as_shipped_pooled"]
        w(f"  text as shipped, pooled: mean {_f(tp['mean'], 4)}, p {_p(tp['p'])}")
        for r in h["traits"]:
            w(f"  {r['trait']}: combined {_f(r['combined']['skill'])} {_ci(r['combined']['ci95'])}; text shipped "
              f"{_f(r['text_as_shipped']['skill'])} {_ci(r['text_as_shipped']['ci95'])}; text calibrated {_f(r['single_calibrated']['skill'])}")
    if "S1" in res:
        for name, sub in res["S1"].items():
            fam(f"\nS1  {name} alone (learned in CV), permutation p, Holm/5; read r against null_mean", sub["traits"],
                cols=("r", "ci95", "null_mean", "p", "p_holm"))
    if "S2" in res:
        fam("\nS2  MBTI reader vs OEJTS letters, AUC with DeLong CI, Holm/4", res["S2"]["axes"], key="axis", cols=("auc", "ci95", "p", "p_holm"))
    if "S3" in res:
        s3r = res["S3"]
        w(f"\nS3  dose-response: product z x log words {_f(s3r['beta_product'], 4)} {_ci(s3r['ci95_product'], 4)}, one-sided p "
          f"{_p(s3r['p_one_sided'])}: {'SUPPORTED' if s3r['supported'] else 'not detected at this n'} (n {s3r['n_people']} people)")
    if "S4" in res:
        for name, sub in res["S4"].items():
            fam(f"\nS4  {name}: mean d (shipped error - refit error)", sub["traits"], cols=("mean", "ci95", "p", "p_holm", "r_refit", "r_shipped"))
    if "S5" in res:
        s5r = res["S5"]
        w(f"\nS5  retest reliability (no hypothesis): selection {s5r['selection']}, used {s5r['n_used']}")
        for r in s5r["traits"]:
            if "skipped" in r:
                w(f"  {r['trait']}: skipped ({r['skipped']})")
            else:
                w(f"  {r['trait']}: n {r['n']}  r {r['r']:.3f} {_ci(r['r_ci95'])}  ICC(2,1) {r['icc21']:.3f} {_ci(r['icc21_ci95'])}  "
                  f"ceiling sqrt(r) {r['validity_ceiling_sqrt_r']:.2f}")
    if "S6" in res:
        fam("\nS6  Reddit-only minus X-only text validity (two-sided, exploratory)", res["S6"]["traits"], cols=("r_reddit", "r_x", "diff", "ci95", "p", "p_holm"))
    if "S7" in res:
        for name in ("bayes", "bayes_eb_cp"):
            sub = res["S7"][name]
            if "skipped" in sub:
                w(f"\nS7  {name}: skipped ({sub['skipped']})")
                continue
            fam(f"\nS7  fusion {name} {sub.get('label', '')}: validity, coverage, skill", sub["traits"],
                cols=("r", "ci95", "p_holm", "coverage", "skill"))
            c = sub["coverage_pooled"]
            w(f"  pooled over {sub['pool']['traits']} (paired error over {sub['pool_for_paired_error']})")
            w(f"  pooled coverage {c['coverage']:.3f} [{c['ci90'][0]:.3f}, {c['ci90'][1]:.3f}] (90%), equivalence: "
              f"{'yes' if c['equivalent_0.75_0.85'] else 'no'}; mean width {c['mean_width']:.2f}")
            k = [x for x in sub if x.startswith("paired_error_vs_")][0]
            pe = sub[k]
            w(f"  {k}: mean {_f(pe['mean'], 4)} {_ci(pe['ci95'], 4)}, one-sided p {_p(pe['p'])}")
        w("  checks (fusion.md): real validity vs stated")
        for r in res["S7"]["checks"]["rows"]:
            if "skipped" in r:
                w(f"    {r['trait']} {r['channel']:<40} n {r['n']:<4} skipped ({r['skipped']})")
                continue
            w(f"    {r['trait']} {r['channel']:<40} n {r['n']:<4} real {_f(r['real_r'])} {_ci(r['ci95'])}  stated {_f(r['stated_median'])}"
              + (f" (x0.75 {_f(r['stated_x0.75'])})" if r["stated_x0.75"] is not None else ""))
    w("\nWording: 'SUPPORTED' only when the registered test passes; otherwise 'not detected at this n' (PREREG section 6).")
    w(f"H2 stack: {'non-negative source weights (registered, P20)' if res['config'].get('stack_nonneg') else 'plain OLS (sensitivity run, not the registered method)'}")
    w("\nPREREG ISSUES (decided before data; see PREREG.md 'Changes before data collection')")
    L.extend(prereg_issues())
    return "\n".join(L)


def prereg_issues() -> list[str]:
    """The PREREG_ISSUES comment block at the top of this file, so the report and the code cannot drift."""
    lines = Path(__file__).read_text().splitlines()
    a = next(i for i, x in enumerate(lines) if x.startswith("# PREREG_ISSUES"))
    b = next(i for i in range(a + 1, len(lines)) if lines[i].startswith("# ====="))
    return ["  " + x[2:].rstrip() for x in lines[a + 1: b]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None, help="output folder (default <folder>/results)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="2 repeats, fewer resamples: a quick look, NOT the registered run")
    ap.add_argument("--only", default=None, help="comma list, e.g. H1,H3 (debugging only)")
    ap.add_argument("--stack-ols", action="store_true", help="sensitivity run: plain OLS stack instead of the registered non-negative one (P20)")
    a = ap.parse_args(argv)
    cfg = Config.fast(a.seed) if a.fast else Config(seed=a.seed)
    cfg.stack_nonneg = not a.stack_ols
    res = analyse(a.folder, cfg, set(a.only.split(",")) if a.only else None)
    out = Path(a.out or Path(a.folder) / "results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    txt = report(res)
    (out / "report.txt").write_text(txt + "\n")
    print(txt)
    return res


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

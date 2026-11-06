"""A listening card for one real person: three sentences chosen with the identity round's joint rule,
the explore or stay dial and the old favourites line. Used by scripts/my_card.py and the recognition
test (py/docs/recognition_test.md). Everything runs on the person's own machine; nothing is uploaded
and nothing but the card text is written anywhere.

Reference. The joint rule (statements/identity.py, `joint_select`) needs, for every statement, whether
it is true of each reference person and each reference person's chance that it holds next time. The
636 reference listeners cannot be shipped person by person, so the card ships aggregates
(data/card_norms.json, built by scripts/build_card_norms.py) and rebuilds a stand-in population from
them: REF_SIZE synthetic people drawn from a Gaussian copula with the 57 x 57 correlation of the real
normal scores, each family missing at its real rate, and each statement true with exactly the real
share among those measured. A synthetic person's hold probability is the identity round's formula with
their own noise set to the family's 1 - r_f (the bivariate normal case), because a synthetic person
has no quarters of their own. The real person gets the full formula, their own two halves standing in
for H1a and H1b. This is an approximation of the frozen rule, not the rule itself: dependence beyond
pairwise correlation and the link between missing values and history length are lost. Shares printed
on the card ("true of X% of listeners") are the exact counts in the 636, not the stand-in's.

The card's window is the whole export (H1 = everything, its two halves = H1a and H1b), as the round 2
cards were made on whole training weeks. D = 4 decoys and k = 3, the identity round's lineup of five.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm

from ..person import Person, Play
from ..readers import music as mr
from .engine import BARNUM, top_k
from .families import Family
from .families_v2 import FAMILIES_V2, measure_v2
from .history import History, from_rows

# The joint rule, imported read-only from the identity round. If identity.py ever stops exporting
# these, copy them here and reconcile with identity.py afterwards.
from .identity import KAPPA, LAMBDA, all_statements, joint_select

DATA = Path(__file__).resolve().parents[1] / "data"
K = 3
DECOYS = 4
REF_SIZE = 2000
REF_SEED = 20261106
MIN_PLAYS = 2_000
MIN_MS = 30_000          # a Spotify stream counts from 30 seconds (Spotify's own threshold)


# ------------------------------------------------------------------------------------- history

def history_from_plays(plays: list[Play]) -> tuple[History, int]:
    """Plays -> History on start times. Spotify plays under 30 seconds are dropped (ListenBrainz
    only records listens). Returns the history and how many plays were dropped."""
    rows, dropped = [], 0
    for p in plays:
        if p.ms_played is not None and p.ms_played < MIN_MS:
            dropped += 1
            continue
        end = int(round(p.when.timestamp() * 1000))
        start = end - int(p.ms_played) if p.ms_played is not None else end
        a = p.artist.strip().lower()
        rel = f"{a}\x1f{p.album.strip().lower()}" if p.album else None
        rows.append((start, a, p.track, rel))
    return from_rows(rows), dropped


# ------------------------------------------------------------------------------------- norms

@dataclass
class Norms:
    n: int
    knots: np.ndarray                 # percentiles of the knots, 1..99
    measured: np.ndarray              # per family
    r: np.ndarray                     # per family split-half r of normal scores
    full: list[np.ndarray]            # per family knot values on whole histories
    half: list[np.ndarray]            # per family knot values on halves
    grid: list[np.ndarray]
    pct_th: list[np.ndarray]          # per family, mid-rank percentile (0-1) of each threshold
    true: dict                        # (family index, "hi"/"lo") -> counts of the n true, per threshold
    corr: np.ndarray


@lru_cache(maxsize=1)
def norms() -> Norms:
    d = json.loads((DATA / "card_norms.json").read_text())
    fams = d["families"]
    if [f["key"] for f in fams] != [f.key for f in FAMILIES_V2]:
        raise RuntimeError("card_norms.json was built for a different family list; rerun scripts/build_card_norms.py")
    true = {}
    for k, f in enumerate(fams):
        for dname in ("hi", "lo"):
            if f[f"true_{dname}"] is not None:
                true[(k, dname)] = np.asarray(f[f"true_{dname}"], float)
    return Norms(d["n"], np.asarray(d["knots"], float), np.array([f["measured"] for f in fams], float),
                 np.array([f["r"] for f in fams], float), [np.asarray(f["full"], float) for f in fams],
                 [np.asarray(f["half"], float) for f in fams], [np.asarray(f["grid"], float) for f in fams],
                 [np.asarray(f["pct_th"], float) for f in fams], true, np.asarray(d["corr"], float))


def pct_of(x: float, knots: np.ndarray, vals: np.ndarray) -> float:
    """Percentile (0-1) of x on a knot table, ties at the middle of their range, beyond the end knots
    half way to 0 or 1 (as the music reader does). NaN for NaN."""
    if not math.isfinite(x) or not len(vals):
        return math.nan
    p = mr.percentile_of(x, {"p": list(knots), "v": list(vals)})
    return math.nan if p is None else p / 100


def zscore(p: float, m: float) -> float:
    if not math.isfinite(p):
        return math.nan
    m = max(m, 1.0)
    return float(ndtri(min(max(p, 0.5 / m), 1 - 0.5 / m)))


# ------------------------------------------------------------------------------------- the stand-in reference

class Ref:
    """Statement truth and hold probabilities for the synthetic reference, plus the statement list."""

    def __init__(self, size: int = REF_SIZE, seed: int = REF_SEED) -> None:
        N = norms()
        self.cs = all_statements(FAMILIES_V2)
        self.groups = np.array([FAMILIES_V2[k].group for k in self.cs.fam])
        F = len(FAMILIES_V2)
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(N.corr + 1e-9 * np.eye(F))
        z = rng.standard_normal((size, F)) @ L.T
        u = norm.cdf(z)
        miss = rng.random((size, F)) >= (N.measured / N.n)[None, :]
        S = len(self.cs)
        T = np.zeros((size, S), bool)
        Q = np.zeros((size, S))
        zth = np.zeros(S)
        share = np.zeros(S)
        for s in range(S):
            k, sign, gi = int(self.cs.fam[s]), int(self.cs.sign[s]), int(self.cs.gidx[s])
            d = "hi" if sign > 0 else "lo"
            c = N.true[(k, d)][gi]
            share[s] = c / N.n
            m = max(N.measured[k], 1.0)
            cond = min(c / m, 1.0)                     # share among those measured
            zth[s] = zscore(N.pct_th[k][gi], m)
            truth = (u[:, k] >= 1 - cond) if sign > 0 else (u[:, k] <= cond)
            T[:, s] = truth & ~miss[:, k]
            r = N.r[k]
            q = norm.cdf(sign * (r * z[:, k] - zth[s]) / math.sqrt(KAPPA * (1 - r * r)))
            Q[:, s] = np.where(miss[:, k], share[s], q)
        self.T1, self.Qall, self.zth, self.share = T, Q, zth, share


@lru_cache(maxsize=1)
def reference() -> Ref:
    return Ref()


# ------------------------------------------------------------------------------------- the card

@dataclass
class Line:
    family: Family
    direction: str
    th: float
    value: float
    share: float          # of the 636 reference listeners it is true of
    hold: float           # chance it still holds on a new stretch of the person's listening

    @property
    def text(self) -> str:
        return self.family.text(self.direction, self.th)


@dataclass
class Card:
    lines: list[Line]
    dial: tuple[float, float] | None          # (score, percentile 0-100) or None
    old_favourites: tuple[float, float | None] | None   # (share, percentile 0-100) or None
    plays: int
    days: float
    notes: list[str] = field(default_factory=list)
    values: np.ndarray | None = None          # the 57 measures on the whole history (kept in memory only)
    percentiles: np.ndarray | None = None


def _person_rows(full: np.ndarray, a: np.ndarray, b: np.ndarray, ref: Ref) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The person's truth, hold probability and whole-history percentiles for every statement."""
    N = norms()
    F = len(FAMILIES_V2)
    p1 = np.array([pct_of(full[k], N.knots, N.full[k]) for k in range(F)])
    z1 = np.array([zscore(p1[k], N.measured[k]) for k in range(F)])
    za = np.array([zscore(pct_of(a[k], N.knots, N.half[k]), 2 * N.measured[k]) for k in range(F)])
    zb = np.array([zscore(pct_of(b[k], N.knots, N.half[k]), 2 * N.measured[k]) for k in range(F)])
    cs = ref.cs
    v = full[cs.fam]
    with np.errstate(invalid="ignore"):
        truth = np.where(cs.sign > 0, v >= cs.th, v <= cs.th) & np.isfinite(v)
    R = N.r[cs.fam]
    dd = (za - zb)[cs.fam] ** 2 / 2
    tau2 = np.where(np.isfinite(dd), (LAMBDA * (1 - R) + dd) / (LAMBDA + 1), 1 - R)
    zz = z1[cs.fam]
    mu = zz * R / (R + tau2)
    var = KAPPA * (R * tau2 / (R + tau2) + tau2)
    q = norm.cdf(cs.sign * (mu - ref.zth) / np.sqrt(var))
    q = np.where(np.isfinite(zz), q, ref.share)
    return truth, np.where(truth, q, 0.0), p1


class _Pop:
    """The duck-typed population identity.joint_select reads: row 0 is the person."""

    def __init__(self, ref: Ref, truth: np.ndarray, q: np.ndarray) -> None:
        self.cs, self.groups = ref.cs, ref.groups
        self.T1 = np.vstack([truth[None, :], ref.T1])
        self.Q = np.vstack([q[None, :], np.where(ref.T1, ref.Qall, 0.0)])
        self.Qall = np.vstack([q[None, :], ref.Qall])


def select(full: np.ndarray, a: np.ndarray, b: np.ndarray, k: int = K, D: int = DECOYS) -> list[Line]:
    ref = reference()
    truth, q, _ = _person_rows(full, a, b, ref)
    pop = _Pop(ref, truth, q)
    chosen = joint_select(pop, 0, np.arange(1, pop.T1.shape[0]), k, D)
    out = []
    for s in chosen:
        f = FAMILIES_V2[int(ref.cs.fam[s])]
        out.append(Line(f, "hi" if ref.cs.sign[s] > 0 else "lo", float(ref.cs.th[s]),
                        float(full[int(ref.cs.fam[s])]), float(ref.share[s]), float(q[s])))
    return out


def barnum_lines(full: np.ndarray, k: int = K) -> list[Line]:
    """A control card: statements true of the person and of 50% to 90% of the reference listeners, the
    most widely shared first, one per family, at most two per group (round 1's Barnum band)."""
    ref = reference()
    cs = ref.cs
    v = full[cs.fam]
    with np.errstate(invalid="ignore"):
        truth = np.where(cs.sign > 0, v >= cs.th, v <= cs.th) & np.isfinite(v)
    ok = np.nonzero(truth & (ref.share >= BARNUM[0]) & (ref.share <= BARNUM[1]))[0]
    best: dict[int, int] = {}
    for s in ok[np.argsort(-ref.share[ok], kind="stable")]:
        best.setdefault(int(cs.fam[s]), int(s))

    @dataclass
    class _S:
        family: Family
        s: int

    cands = sorted((_S(FAMILIES_V2[f], s) for f, s in best.items()), key=lambda x: -ref.share[x.s])
    out = []
    for c in top_k(cands, k):
        s = c.s
        out.append(Line(c.family, "hi" if cs.sign[s] > 0 else "lo", float(cs.th[s]), float(full[int(cs.fam[s])]),
                        float(ref.share[s]), math.nan))
    return out


def _dial_and_favourites(plays: list[Play]) -> tuple[tuple[float, float] | None, tuple[float, float | None] | None]:
    reader = mr.MusicReader()
    if not reader.norms or not reader.dials:
        return None, None
    tr = mr.measure(mr.arrays_from_plays(plays))
    vals = {k: t.value for k, t in tr.items()}
    sc = mr.dial_scores(vals, reader.norms, reader.dials)
    dial = None
    if "dial_explore" in sc:
        v = sc["dial_explore"]
        dial = (v, mr.percentile_of(v, reader.norms["dials"]["dial_explore"]))
    of = None
    if math.isfinite(vals.get("old_friends", math.nan)):
        of = (vals["old_friends"], mr.percentile_of(vals["old_friends"], reader.norms["traits"]["old_friends"]))
    return dial, of


def make_card(person: Person, k: int = K) -> Card:
    plays = [p for p in person.plays if p.artist and p.track]
    h, dropped = history_from_plays(plays)
    notes = list(person.notes)
    if dropped:
        notes.append(f"{dropped:,} plays under 30 seconds left out")
    n = len(h.t)
    if n == 0:
        return Card([], None, None, 0, 0.0, notes + ["no plays found"])
    full = h.window()
    a, b = h.halves()
    fv, av, bv = measure_v2(full), measure_v2(a), measure_v2(b)
    if n < MIN_PLAYS:
        notes.append(f"only {n:,} plays; the reference listeners all had 2,000 or more, so this card is shaky")
    _, cnt = np.unique(h.artist, return_counts=True)
    if cnt.max() / n > 0.75 or len(cnt) < 50:
        notes.append("one artist dominates this history (the reference left such accounts out)")
    lines = select(fv, av, bv, k)
    dial, of = _dial_and_favourites(plays)
    N = norms()
    pcts = np.array([pct_of(fv[j], N.knots, N.full[j]) for j in range(len(FAMILIES_V2))])
    return Card(lines, dial, of, n, full.days, notes, fv, pcts)


def make_barnum(card: Card, k: int = K) -> Card:
    """The Barnum control card for a person whose card was made (same dial and favourites lines)."""
    return Card(barnum_lines(card.values, k) if card.values is not None else [], card.dial, card.old_favourites,
                card.plays, card.days)


# ------------------------------------------------------------------------------------- text

def share_text(x: float) -> str:
    v = 100 * x
    if v < 1:
        return "under 1%" if v < 0.5 else f"{v:.1f}%"
    return f"{round(v)}%"


def _pc(p: float) -> int:
    """A percentile for reading: whole numbers from 1 to 99 (the norms do not resolve the extremes)."""
    return int(min(max(round(p), 1), 99))


def render(card: Card) -> list[str]:
    """The card body, as plain lines. Same text in a person's own run and on a lineup sheet."""
    out = []
    if not card.lines:
        out.append("No sentence could be made: the history is too short for the measures to say anything.")
    for j, ln in enumerate(card.lines, 1):
        out.append(f"{j}. {ln.text}")
        out.append(f"   Yours: {ln.family.show(ln.value)}. True of {share_text(ln.share)} of listeners.")
    out.append("")
    if card.dial is not None:
        v, p = card.dial
        if v >= 0:
            side, more = "exploring new artists", f"Further toward exploring than {_pc(p)}% of listeners."
        else:
            side, more = "staying with old favourites", f"Further toward staying than {_pc(100 - p)}% of listeners."
        out.append(f"Explore or stay: you lean toward {side} ({v:+.2f}, where 0 is the typical listener). {more}")
    else:
        out.append("Explore or stay: not enough history yet (it needs about nine months).")
    if card.old_favourites is not None:
        v, p = card.old_favourites
        tail = f" Higher than {_pc(p)}% of listeners." if p is not None else ""
        out.append(f"Old favourites: {round(100 * v)}% of your plays go to artists who were already in your top 50 "
                   f"four to six months before.{tail}")
    else:
        out.append("Old favourites: not enough history yet (it needs about eight months).")
    return out

"""The "only you" filter: from a person's family values, the threshold statements that are true of
them and of at most 10% of a reference population, ranked by rarity times confidence.

Population thresholds are always counted with the person left out (and, in the lineup test, the
decoys too). A statement's threshold comes from its family's fixed grid of round numbers, never
from the person's own value, so every statement is a one-sided pattern ("at least 60% of ...") that
anyone could satisfy.

Confidence asks whether the statement would survive a different stretch of the same history: the
window is cut into four quarters, the family is measured on each, and confidence is
Phi((value - threshold) / sd of the quarter values) (sign flipped for "no more than"). The sd is
floored at a tenth of the family's population sd, and when fewer than two quarters can be measured
it is set to half the population sd, which penalises statements that only long windows support.
Rarity is the surprisal -log2((c + 1) / (n + 1)) of the c of n reference people the statement is
true of. Score = confidence x surprisal.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .families import FAMILIES, Family

ONLY_YOU = 0.10          # true of at most this share of the reference population
PER_GROUP = 2            # at most this many statements from one group in a top-k
BARNUM = (0.5, 0.9)      # base-rate band for the Barnum control


@dataclass
class Statement:
    family: Family
    direction: str       # "hi" (at least) or "lo" (no more than)
    th: float
    value: float         # the person's own measured value
    conf: float
    c: int               # reference people it is also true of
    n: int               # reference people
    score: float

    @property
    def share(self) -> float:
        return self.c / self.n if self.n else math.nan

    @property
    def text(self) -> str:
        return self.family.text(self.direction, self.th)

    def card(self) -> str:
        return (f"{self.text} Yours: {self.family.show(self.value)}. True of {self.c} of the {self.n} other "
                f"listeners ({100 * self.share:.1f}%).")

    def key(self) -> tuple[str, str, float]:
        return (self.family.key, self.direction, self.th)


class Table:
    """Family values for a population on one kind of window, with the truth of every grid statement."""

    def __init__(self, values: np.ndarray, quarter_sd: np.ndarray | None = None,
                 pop_sd: np.ndarray | None = None, families: list[Family] | None = None) -> None:
        self.families = FAMILIES if families is None else families
        self.V = np.asarray(values, float)                  # people x families
        self.P = self.V.shape[0]
        self.sdq = np.full_like(self.V, np.nan) if quarter_sd is None else np.asarray(quarter_sd, float)
        if pop_sd is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pop_sd = np.nanstd(self.V, axis=0)
        self.pop_sd = np.nan_to_num(np.asarray(pop_sd, float))
        self.grid = [np.asarray(f.grid, float) for f in self.families]
        with np.errstate(invalid="ignore"):
            self.T = {(k, d): (self.V[:, k, None] >= g) if d == "hi" else (self.V[:, k, None] <= g)
                      for k, (f, g) in enumerate(zip(self.families, self.grid)) for d in f.dirs}
        self.tot = {kd: t.sum(0) for kd, t in self.T.items()}

    def truth(self, st: Statement, people: np.ndarray | slice = slice(None)) -> np.ndarray:
        k = self.families.index(st.family)
        g = int(np.nonzero(self.grid[k] == st.th)[0][0])
        return self.T[(k, st.direction)][people, g]

    def _spread(self, i: int, k: int) -> float:
        s = self.sdq[i, k]
        floor = 0.1 * self.pop_sd[k]
        if not math.isfinite(s):
            return max(0.5 * self.pop_sd[k], 1e-12)
        return max(s, floor, 1e-12)

    def candidates(self, i: int, exclude: list[int] | np.ndarray, mode: str = "only") -> list[Statement]:
        """The best statement per family and direction for person i (true of i), against everyone
        except `exclude` (which should contain i). mode "only": share <= 10%, ranked by score;
        mode "barnum": share inside the Barnum band, ranked by confidence."""
        ex = np.unique(np.asarray(exclude, dtype=np.int64))
        n = self.P - len(ex)
        out = []
        for (k, d), T in self.T.items():
            v = self.V[i, k]
            if not math.isfinite(v):
                continue
            mine = T[i]
            if not mine.any():
                continue
            c = self.tot[(k, d)] - T[ex].sum(0)
            p = c / n
            g = self.grid[k]
            z = (v - g) / self._spread(i, k) * (1 if d == "hi" else -1)
            conf = norm.cdf(z)
            if mode == "only":
                ok = mine & (p <= ONLY_YOU)
                sc = conf * -np.log2((c + 1) / (n + 1))
            else:
                ok = mine & (p >= BARNUM[0]) & (p <= BARNUM[1])
                sc = conf
            if not ok.any():
                continue
            j = int(np.argmax(np.where(ok, sc, -np.inf)))
            out.append(Statement(self.families[k], d, float(g[j]), float(v), float(conf[j]), int(c[j]), n, float(sc[j])))
        out.sort(key=lambda s: -s.score)
        # one statement per family (the better direction)
        seen, uniq = set(), []
        for s in out:
            if s.family.key not in seen:
                seen.add(s.family.key)
                uniq.append(s)
        return uniq


def top_k(statements: list[Statement], k: int, per_group: int = PER_GROUP) -> list[Statement]:
    """Highest-scoring first, at most `per_group` from one group of families."""
    used: dict[str, int] = {}
    out = []
    for s in statements:
        if used.get(s.family.group, 0) >= per_group:
            continue
        used[s.family.group] = used.get(s.family.group, 0) + 1
        out.append(s)
        if len(out) == k:
            break
    return out


def quarter_sd(quarter_values: np.ndarray) -> np.ndarray:
    """people x 4 x families -> people x families sd (ddof 1) over the quarters that could be measured."""
    q = np.asarray(quarter_values, float)
    ok = np.isfinite(q).sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(q, axis=1, ddof=1)
    sd[ok < 2] = np.nan
    return sd

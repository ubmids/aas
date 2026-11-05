"""Evaluation core: run the readers on one person, check the guesses against the quiz,
and score the readers across many people.

    prof  = build(person, readers)             Estimates + the reasons a reader couldn't read
    comp  = compare(prof, quiz_result)         one person: predicted vs their own quiz, inside the interval?
    ev    = evaluate([(prof, quiz_result)...]) many people: interval coverage, error vs predict-the-average
    print(dump(prof, comp))                    bare plain text, for debugging

Rules enforced here:
  - an Estimate with usable=False never becomes a claim, only a reason
  - two readers on the same Big Five trait are kept side by side and this module never
    averages them; a combination appears only when the `fused` reader (aas.fusion.FusedReader)
    is in the readers list, as one more estimate under its own name
No UI lives here on purpose: the model gets finished and scored first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import erf, sqrt

from .person import Estimate, Person, Reader
from .schema import TRAITS

# Loose MBTI -> Big Five links (McCrae & Costa 1989, |r| about 0.4 to 0.7).
# axis -> (Big Five trait, sign of that trait when the FIRST letter is true).
from .fusion import MBTI_LINKS  # one home for the MBTI to Big Five links

MBTI_TO_BIGFIVE = {axis: (trait, sign) for axis, (trait, sign, _r) in MBTI_LINKS.items()}
MBTI_ORDER = ["I/E", "N/S", "T/F", "J/P"]


def phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


# ---------------------------------------------------------------- build
@dataclass
class Unread:
    """Something a reader could not say, and why."""
    reader: str
    reason: str
    trait: str | None = None


@dataclass
class Profile:
    sources: list[str]
    notes: list[str]
    bigfive: dict[str, list[Estimate]] = field(default_factory=dict)  # trait -> one per reader (fused is one more reader)
    mbti: list[Estimate] = field(default_factory=list)
    measured: list[Estimate] = field(default_factory=list)
    unread: list[Unread] = field(default_factory=list)
    readers_run: list[str] = field(default_factory=list)

    @property
    def estimates(self) -> list[Estimate]:
        return [e for t in TRAITS for e in self.bigfive.get(t, [])] + self.mbti + self.measured

    def to_dict(self) -> dict:
        return {
            "sources": self.sources,
            "notes": self.notes,
            "readers_run": self.readers_run,
            "estimates": [asdict(e) for e in self.estimates],
            "unread": [asdict(u) for u in self.unread],
        }


def build(person: Person, readers: list[Reader]) -> Profile:
    """Run every reader that can read this person; keep its usable Estimates and every reason it couldn't.

    A reader that raises is recorded as a reason, not allowed to take the profile down.
    """
    prof = Profile(sources=list(person.sources), notes=list(person.notes))
    for r in readers:
        name = getattr(r, "name", type(r).__name__)
        try:
            ok, why = r.can_read(person)
        except Exception as ex:
            prof.unread.append(Unread(name, f"failed while checking the data ({type(ex).__name__}: {ex})"))
            continue
        if not ok:
            prof.unread.append(Unread(name, why or "not enough data"))
            continue
        try:
            ests = r.read(person)
        except Exception as ex:
            prof.unread.append(Unread(name, f"failed while reading ({type(ex).__name__}: {ex})"))
            continue
        prof.readers_run.append(name)
        for e in ests:
            if not e.usable:
                prof.unread.append(Unread(e.reader or name, e.evidence or "too little data", e.trait))
            elif e.kind == "bigfive":
                prof.bigfive.setdefault(e.trait, []).append(e)
            elif e.kind == "mbti":
                prof.mbti.append(e)
            else:
                prof.measured.append(e)
    prof.mbti.sort(key=lambda e: (MBTI_ORDER.index(e.trait) if e.trait in MBTI_ORDER else 9, e.reader))
    return prof


# ---------------------------------------------------------------- one person
@dataclass
class Check:
    """One prediction next to the person's own quiz answer."""
    reader: str
    trait: str                    # Big Five letter
    via: str                      # "bigfive" (direct) or an MBTI axis such as "I/E" (loose)
    predicted_z: float | None     # direct only
    low: float | None
    high: float | None
    quiz_z: float
    quiz_percentile: float
    inside: bool | None           # quiz z inside the predicted interval (None: no interval / loose)
    same_side: bool | None        # prediction and quiz on the same side of average
    validity: float | None
    validity_kind: str | None
    level: float | None = None    # how often the interval claims to contain the truth

    @property
    def error(self) -> float | None:
        return None if self.predicted_z is None else self.predicted_z - self.quiz_z


@dataclass
class Comparison:
    checks: list[Check]
    overall: str
    missing: list[str]   # Big Five letters the quiz has but no reader spoke to


def _quiz_traits(quiz_result: dict) -> dict:
    return {t: v for t, v in quiz_result.items() if t in TRAITS and isinstance(v, dict) and "z" in v}


def compare(profile: Profile, quiz_result: dict) -> Comparison:
    """Every Big Five claim vs the person's own quiz score (aas.quiz.score() output).

    MBTI estimates get a loose same-side-of-average check on their nearest Big Five trait
    (not a test: the two systems only partly overlap)."""
    q = _quiz_traits(quiz_result)
    checks: list[Check] = []
    for t in TRAITS:
        if t not in q:
            continue
        qz, qp = float(q[t]["z"]), float(q[t]["percentile"])
        for e in profile.bigfive.get(t, []):
            has = e.low is not None and e.high is not None
            checks.append(Check(e.reader, t, "bigfive", e.value, e.low, e.high, qz, qp,
                                (e.low <= qz <= e.high) if has else None,
                                (e.value >= 0) == (qz >= 0), e.validity, e.validity_kind, e.interval_level))
    for e in profile.mbti:
        if e.trait not in MBTI_TO_BIGFIVE:
            continue
        t, sign = MBTI_TO_BIGFIVE[e.trait]
        if t not in q:
            continue
        qz, qp = float(q[t]["z"]), float(q[t]["percentile"])
        expect_high = (e.value >= 0.5) == (sign > 0)
        checks.append(Check(e.reader, t, e.trait, None, None, None, qz, qp, None,
                            expect_high == (qz >= 0), e.validity, e.validity_kind))

    direct = [c for c in checks if c.via == "bigfive" and c.inside is not None]
    loose = [c for c in checks if c.via != "bigfive"]
    parts = []
    if direct:
        parts.append(f"{sum(c.inside for c in direct)} of {len(direct)} Big Five guesses had the quiz score inside their interval.")
    else:
        parts.append("No direct Big Five guess with an interval, nothing to score strictly.")
    if loose:
        parts.append(f"Loose MBTI checks: {sum(bool(c.same_side) for c in loose)} of {len(loose)} on the same side of average.")
    spoken = {c.trait for c in checks}
    return Comparison(checks, " ".join(parts), [t for t in TRAITS if t in q and t not in spoken])


# ---------------------------------------------------------------- many people
@dataclass
class ReaderTraitScore:
    """How one reader did on one Big Five trait across many people."""
    reader: str
    trait: str
    n: int                         # people with both a usable guess and a quiz score
    n_with_interval: int
    coverage: float | None         # share of quiz scores inside the stated interval
    stated_level: float            # what coverage the interval claims (e.g. 0.80)
    mean_width: float | None       # average interval width in z
    mae: float                     # mean |guess - quiz| in z
    mae_baseline: float            # same, guessing average (z = 0) for everyone
    skill: float                   # 1 - mae / mae_baseline; > 0 beats the baseline
    r: float | None                # correlation guess vs quiz (None if n < 3 or no spread)
    same_side: float               # share on the same side of average

    def line(self) -> str:
        cov = "n/a" if self.coverage is None else f"{self.coverage:.2f} (stated {self.stated_level:.2f}, n={self.n_with_interval})"
        r = "n/a" if self.r is None else f"{self.r:+.2f}"
        return (f"{self.reader:<16} {self.trait}  n={self.n:<4} coverage {cov}  MAE {self.mae:.2f} vs "
                f"baseline {self.mae_baseline:.2f} (skill {self.skill:+.2f})  r {r}  same side {self.same_side:.2f}")


@dataclass
class Evaluation:
    scores: list[ReaderTraitScore]
    n_people: int
    skipped: dict[str, int]        # reader -> how many people it couldn't read (wholly or a Big Five trait)

    def lines(self) -> list[str]:
        out = [f"{self.n_people} people"]
        out += [s.line() for s in self.scores]
        out += [f"{k}: couldn't read {v} of {self.n_people}" for k, v in sorted(self.skipped.items())]
        return out


def _corr(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    sa = sqrt(sum((x - ma) ** 2 for x in a))
    sb = sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def _stated(checks: list[Check], fallback: float) -> float:
    levels = {c.level for c in checks if c.level is not None}
    if len(levels) > 1:
        raise ValueError(f"one reader x trait states several interval levels: {sorted(levels)}")
    return levels.pop() if levels else fallback


def evaluate(pairs: list[tuple[Profile, dict]], level: float = 0.80) -> Evaluation:
    """Score each reader x Big Five trait across many (profile, quiz_result) pairs.

    coverage: do the stated intervals contain the quiz score about `level` of the time?
      Much lower = overconfident intervals; much higher = intervals wider than they need be.
    mae vs mae_baseline: is the guess closer to the truth than guessing average (z = 0) for all?
      The baseline is z = 0 because quiz z is already relative to the ~600k-person norms.
    The stated level comes from each Estimate's interval_level; `level` is the fallback when unset.
    """
    by: dict[tuple[str, str], list[Check]] = {}
    skipped: dict[str, int] = {}
    for prof, qr in pairs:
        for reader in {u.reader for u in prof.unread if u.trait is None or u.trait in TRAITS}:
            skipped[reader] = skipped.get(reader, 0) + 1
        for c in compare(prof, qr).checks:
            if c.via == "bigfive":
                by.setdefault((c.reader, c.trait), []).append(c)
    scores = []
    for (reader, t), cs in sorted(by.items(), key=lambda kv: (kv[0][0], TRAITS.index(kv[0][1]))):
        n = len(cs)
        wi = [c for c in cs if c.inside is not None]
        pred = [c.predicted_z for c in cs]
        truth = [c.quiz_z for c in cs]
        mae = sum(abs(p - q) for p, q in zip(pred, truth)) / n
        base = sum(abs(q) for q in truth) / n
        scores.append(ReaderTraitScore(
            reader, t, n, len(wi),
            sum(c.inside for c in wi) / len(wi) if wi else None,
            _stated(wi, level),
            sum(c.high - c.low for c in wi) / len(wi) if wi else None,
            mae, base, 1 - mae / base if base > 0 else 0.0,
            _corr(pred, truth),
            sum(bool(c.same_side) for c in cs) / n,
        ))
    return Evaluation(scores, len(pairs), skipped)


# ---------------------------------------------------------------- debug dump
def _est_line(e: Estimate) -> str:
    rng = "no interval" if e.low is None or e.high is None else f"[{e.low:+.2f}, {e.high:+.2f}]"
    pct = "" if e.percentile is None else f" pct {e.percentile:.0f}"
    val = "untested" if e.validity is None else f"{e.validity_kind} {e.validity:.2f}"
    ev = f"  ({e.evidence})" if e.evidence else ""
    return f"  {e.reader:<16} {e.trait:<18} {e.value:+.3f} {rng}{pct}  validity {val}{ev}"


def dump(profile: Profile, comparison: Comparison | None = None) -> str:
    """Bare plain text of everything in the profile, for debugging. Not a user-facing view."""
    L = [f"sources: {', '.join(profile.sources) or '-'}", f"readers run: {', '.join(profile.readers_run) or '-'}"]
    L += [f"note: {n}" for n in profile.notes]
    for t in TRAITS:
        ests = profile.bigfive.get(t, [])
        if ests:
            L.append(f"bigfive {t} (z)" + ("  several readers, shown separately (the fused reader, if run, is the combination)" if len(ests) > 1 else ""))
            L += [_est_line(e) for e in ests]
    if profile.mbti:
        L.append("mbti (value = P(first letter))")
        L += [_est_line(e) for e in profile.mbti]
    if profile.measured:
        L.append("measured")
        L += [_est_line(e) for e in profile.measured]
    if profile.unread:
        L.append("unread")
        L += [f"  {u.reader}{' ' + u.trait if u.trait else ''}: {u.reason}" for u in profile.unread]
    if comparison is not None:
        L.append("vs quiz")
        for c in comparison.checks:
            if c.via == "bigfive":
                inside = "-" if c.inside is None else ("inside" if c.inside else "OUTSIDE")
                L.append(f"  {c.reader:<16} {c.trait} guess {c.predicted_z:+.2f} quiz {c.quiz_z:+.2f} "
                         f"err {c.error:+.2f} {inside}")
            else:
                L.append(f"  {c.reader:<16} {c.trait} loose via {c.via}: quiz {c.quiz_z:+.2f} "
                         f"{'same side' if c.same_side else 'other side'}")
        if comparison.missing:
            L.append(f"  no reader for: {' '.join(comparison.missing)}")
        L.append(f"  {comparison.overall}")
    return "\n".join(L) + "\n"

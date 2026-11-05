"""Text readers: what a person writes -> MBTI letters and Big Five, from saved models.

    TextMBTIReader      reader "text-mbti"      P(first letter) for I/E, N/S, T/F, J/P
    TextBigFiveReader   reader "text-bigfive"   Big Five as a z-score with an 80% interval

Both load artifacts from py/models/<name>/ (model.joblib + card.json), built by
scripts/train_text_readers.py. See docs/readers_text.md for what the numbers mean.

How one person is read
  1. keep only things they wrote as prose (kind "search" is left out by default: queries
     are 2-4 word keyword strings, nothing like the forum posts / essays the models learned)
  2. strip leaks from every item, the same way training text was stripped (people on
     Reddit write "INTJ here" and "I scored high on openness" too)
  3. count words; under the floor -> usable=False with the reason
  4. score: word counts -> TF-IDF -> logistic (or ridge) head -> length-aware calibration.
     Very long histories are cut into training-sized chunks and the chunk scores averaged,
     because the model was never tested on more text than a training person had.
  5. interval: bootstrap over the person's own items (how much the answer moves if their
     texts had been a different sample), plus the model's tested error at that length.
  6. validity by KIND of writing (card "validity_by_source", 2026-11-01): each item is sorted into
     tweets / essay-like long prose / other prose (Reddit comments and the rest). The stated
     accuracy is the measured accuracy on that kind of writing (in-domain only for essay-like
     prose), mixed by words when a person brings several kinds, and the estimate is shrunk and its
     interval widened to match. A trait never shown to work outside the training data (openness
     for Big Five) is refused (usable=False) unless most of the words are essay-like.

Nothing here touches the network. Inference needs numpy, scipy and scikit-learn only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from math import erf, log, sqrt
from pathlib import Path

import numpy as np

from ..person import Estimate, Person, TextItem
from ..slices.text import strip_leaks
from ..slices.text_pandora import BIG5_LEAK

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
EXCLUDED_KINDS = frozenset({"search"})
INTERVAL_LEVEL = 0.80
Z80 = 1.2815515655446004  # standard normal 90th percentile: m +- Z80*sd is an 80% interval
CHUNK_WORDS = 20          # bootstrap / subsampling unit when a person has few long items
MIN_UNITS = 20

# Kinds of writing, for validity by source (card "validity_by_source"). Defaults; a card may override.
TWEET_SOURCES = frozenset({"x", "twitter"})
TWEET_KINDS = frozenset({"tweet", "reply", "retweet", "quote"})
ESSAY_KINDS = frozenset({"essay", "journal", "diary", "blog", "letter"})
ESSAY_MIN_WORDS = 300     # one item this long is essay-like prose (about the 5th percentile of Essays)
WRITING_KINDS = ("essay_like", "tweets", "other_prose")
WRITING_LABEL = {"essay_like": "essay-like long prose", "tweets": "tweets", "other_prose": "other prose (e.g. Reddit comments)"}


def kind_of_writing(item: TextItem, words: int, essay_min_words: int = ESSAY_MIN_WORDS) -> str:
    """Which measured kind of writing one item is: "tweets", "essay_like" or "other_prose"."""
    if (item.source or "").lower() in TWEET_SOURCES or (item.kind or "").lower() in TWEET_KINDS:
        return "tweets"
    if (item.kind or "").lower() in ESSAY_KINDS or words >= essay_min_words:
        return "essay_like"
    return "other_prose"


# ---------------------------------------------------------------- shared text handling

def clean(text: str) -> str:
    """The one leak stripper for training AND inference: MBTI strict strip + Big Five self-report words."""
    return re.sub(r"\s+", " ", BIG5_LEAK.sub(" ", strip_leaks(text, strict=True))).strip()


def n_words(text: str) -> int:
    return len(text.split())


def units(texts: list[str], min_units: int = MIN_UNITS, chunk: int = CHUNK_WORDS) -> list[str]:
    """Bootstrap/subsampling units: the items themselves, or ~20-word pieces when there are
    too few items to resample (one long essay, three long posts)."""
    texts = [t for t in texts if t]
    if len(texts) >= min_units:
        return texts
    out = []
    for t in texts:
        w = t.split()
        out += [" ".join(w[i : i + chunk]) for i in range(0, len(w), chunk)]
    return out


def own_prose(person: Person, exclude_kinds=EXCLUDED_KINDS, essay_min_words: int = ESSAY_MIN_WORDS) -> tuple[list[str], dict]:
    """The person's own writing, leak-stripped, minus excluded kinds. Returns (texts, stats).
    stats["words_by_writing"]: words per kind of writing (tweets / essay_like / other_prose)."""
    kept, n_items, n_excl, by_kind, by_writing = [], 0, 0, {}, {}
    for it in person.texts:
        if it.kind in exclude_kinds:
            n_excl += 1
            continue
        c = clean(it.text or "")
        if not c:
            continue
        n_items += 1
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
        w = n_words(c)
        k = kind_of_writing(it, w, essay_min_words)
        by_writing[k] = by_writing.get(k, 0) + w
        kept.append(c)
    return kept, {"items": n_items, "excluded": n_excl, "by_kind": by_kind, "words": sum(map(n_words, kept)),
                  "words_by_writing": by_writing}


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else ("es" if word.endswith(("s", "sh", "ch", "x")) else "s"))


def phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def probit(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(min(max(p, 1e-6), 1 - 1e-6)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def auc_to_r(auc: float) -> float:
    """r of a score with a continuous trait given its AUC against the median split (bivariate normal);
    the same formula as aas.fusion.auc_to_r, kept here so readers don't import the fuser."""
    from math import pi, sin

    return sqrt(2) * sin(pi * (min(max(float(auc), 0.0), 1.0) - 0.5) / 2)


def interp_log(k: float, ks: list[float], vals: list[float]) -> float:
    """Linear interpolation in log(words); flat beyond the ends."""
    return float(np.interp(log(max(k, 1)), np.log(ks), vals))


# ---------------------------------------------------------------- the scoring core (shared with training)

@dataclass
class Head:
    """One trait: linear head on TF-IDF, then length-aware recalibration.

    binary:      logit p = a*z + b*z*lw + c*lw + d        (lw = log(words / ref_words))
    continuous:  zhat    = same form, fitted by least squares on held-out predictions
    """
    coef: np.ndarray
    intercept: float
    cal: np.ndarray          # (a, b, c, d)
    base_rate: float | None  # binary only: share of positives in training (the prior to remove)

    def raw(self, X) -> np.ndarray:
        return np.asarray(X @ self.coef).ravel() + self.intercept

    def recal(self, z: np.ndarray, words: np.ndarray, ref_words: float) -> np.ndarray:
        lw = np.log(np.maximum(words, 1) / ref_words)
        a, b, c, d = self.cal
        return a * z + b * z * lw + c * lw + d


def tfidf(counts, idf: np.ndarray):
    """Sublinear TF-IDF + L2 norm on a (n x V) count matrix (sparse or dense). Same as sklearn's
    TfidfTransformer(sublinear_tf=True), written out so bootstrap resamples can reuse it."""
    import scipy.sparse as sp

    X = sp.csr_matrix(counts, dtype=np.float64, copy=True)
    X.data = (1 + np.log(X.data)) if X.nnz else X.data
    X = X @ sp.diags(idf)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    norms[norms == 0] = 1
    return sp.diags(1 / norms) @ X


def build_vectorizer(vocab: dict[str, int], ngram_range=(1, 1)):
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(vocabulary=vocab, ngram_range=tuple(ngram_range), lowercase=True)


# ---------------------------------------------------------------- reader base

class _TextReader:
    name = "text"
    kind = "mbti"

    def __init__(self, model_dir: Path | str | None = None, exclude_kinds=EXCLUDED_KINDS, boots: int = 200, seed: int = 0):
        self.dir = Path(model_dir) if model_dir else MODELS_DIR / self.name.replace("-", "_")
        self.card, self.m = _load(str(self.dir))
        self.exclude_kinds = frozenset(exclude_kinds)
        self.boots = boots
        self.seed = seed
        self.floor = int(self.card["min_text"]["floor_words"])
        self.recommended = int(self.card["min_text"]["recommended_words"])
        self.vec = build_vectorizer(self.m["vocab"], self.m["ngram_range"])
        self.vbs = self.card.get("validity_by_source")      # None: in-domain validity only (older / PANDORA cards)
        self.essay_min_words = int((self.vbs or {}).get("essay_min_words", ESSAY_MIN_WORDS))

    def _prose(self, person: Person):
        return own_prose(person, self.exclude_kinds, self.essay_min_words)

    # -- Reader protocol
    def can_read(self, person: Person) -> tuple[bool, str]:
        texts, st = self._prose(person)
        if st["words"] < self.floor:
            why = (f"too little of your own writing: {st['words']:,} words in {st['items']:,} items; this reader needs "
                   f"at least {self.floor:,} (accuracy falls to about half its best below that)")
            if st["excluded"]:
                why += f"; {_plural(st['excluded'], 'search')} left out because queries are not prose"
            return False, why
        if st["words"] < self.recommended:
            return True, f"{st['words']:,} words: readable, but below the {self.recommended:,} where accuracy levels off"
        return True, f"{st['words']:,} words of your own writing"

    def read(self, person: Person) -> list[Estimate]:
        texts, st = self._prose(person)
        traits = list(self.card["traits"])
        if st["words"] < self.floor:
            ok, why = self.can_read(person)
            return [Estimate(reader=self.name, trait=t, kind=self.kind, value=float("nan"), usable=False, evidence=why, interval_level=None)
                    for t in traits]
        z_point, z_boot, words_used, chunked = self._scores(texts, st["words"])
        out = []
        for t in traits:
            out.append(self._estimate(t, z_point[t], z_boot[t], words_used, st, chunked))
        return out

    # -- scoring
    def _scores(self, texts: list[str], words: int):
        """Point and bootstrap scores per trait, on the calibrated scale (logit or z)."""
        u = units(texts)
        C = self.vec.transform(u)                       # (units x V) counts
        uw = np.array([n_words(x) for x in u], float)
        cap = float(self.card["features"]["long_text_cap_words"])
        ref = float(self.card["features"]["ref_words"])
        idf = self.m["idf"]
        rng = np.random.default_rng(self.seed)
        heads: dict[str, Head] = self.m["heads"]
        chunked = words > cap * 1.5
        if chunked:
            # Partition units into training-sized chunks; the score is the mean chunk score.
            order = rng.permutation(len(u))
            groups, cur, cw = [], [], 0.0
            for i in order:
                cur.append(i)
                cw += uw[i]
                if cw >= cap:
                    groups.append(cur)
                    cur, cw = [], 0.0
            if cur and cw >= cap / 2:
                groups.append(cur)
            import scipy.sparse as sp

            G = sp.csr_matrix((np.ones(sum(map(len, groups))), (np.repeat(np.arange(len(groups)), [len(g) for g in groups]), np.concatenate(groups))), shape=(len(groups), len(u)))
            Xg = tfidf(G @ C, idf)
            gw = np.asarray(G @ uw).ravel()
            point, boot = {}, {}
            for t, h in heads.items():
                s = h.recal(h.raw(Xg), gw, ref)
                point[t] = float(s.mean())
                idx = rng.integers(0, len(s), (self.boots, len(s)))
                boot[t] = s[idx].mean(1)
            return point, boot, cap, len(groups)
        X = tfidf(C.sum(0), idf)
        W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=self.boots).astype(float)
        Xb = tfidf(W @ C, idf)
        point, boot = {}, {}
        for t, h in heads.items():
            point[t] = float(h.recal(h.raw(X), np.array([words]), ref)[0])
            boot[t] = h.recal(h.raw(Xb), np.full(self.boots, words, float), ref)
        return point, boot, words, 0

    def _curve(self, t: str, key: str, words: float) -> float:
        c = self.card["min_text"]["curve"]
        ks = [row["words"] for row in c]
        return interp_log(words, ks, [row[key][t] for row in c])

    def _full_auc(self, t: str) -> float:
        return float(self.card["test"]["full"][t]["score"])

    def validity_mix(self, t: str, words: float, st: dict) -> dict:
        """Stated accuracy for this person's mix of kinds of writing (card "validity_by_source").

        Returns auc (what to state), auc_in (in-domain curve at this length), shares {kind: word share},
        shown_share (share of words on which this trait has been shown to work), refuse (bool),
        parts [(kind, share, auc, row)]. Without a validity_by_source block: in-domain, as before."""
        auc_in = self._curve(t, "auc", words)
        vbs = self.vbs
        byw = {k: v for k, v in (st.get("words_by_writing") or {}).items() if v > 0}
        if not vbs or not byw:
            return {"auc": auc_in, "auc_in": auc_in, "shares": {"in_domain": 1.0}, "shown_share": 1.0, "refuse": False, "parts": []}
        full = self._full_auc(t)
        lift_share = min(max((auc_in - 0.5) / (full - 0.5), 0.0), 1.0) if full > 0.5 else 1.0
        total = float(sum(byw.values()))
        auc, shown, parts = 0.0, 0.0, []
        for k in WRITING_KINDS:
            if k not in byw:
                continue
            f = byw[k] / total
            src = vbs["sources"][k]
            if src.get("in_domain"):
                a, row, ok = auc_in, None, True
            else:
                row = src["traits"][t]
                ok = bool(row.get("shown", True))
                a = 0.5 + (float(row["auc"]) - 0.5) * lift_share if ok else 0.5
            auc += f * a
            shown += f * ok
            parts.append((k, f, a, row))
        return {"auc": auc, "auc_in": auc_in, "shares": {k: f for k, f, _, _ in parts}, "shown_share": shown,
                "refuse": shown < float(vbs.get("refuse_below_shown_share", 0.5)), "parts": parts}

    def _mix_text(self, V: dict) -> str:
        """Plain words on which kinds of writing the stated accuracy comes from."""
        if not V["parts"]:
            return ""
        kinds = ", ".join(f"{f:.0%} {WRITING_LABEL[k]}" for k, f, _, _ in V["parts"])
        bits = []
        for k, f, a, row in V["parts"]:
            if row is None:
                bits.append(f"{WRITING_LABEL[k]}: tested in domain, AUC {a:.2f}")
            elif not row.get("shown", True):
                bits.append(f"{WRITING_LABEL[k]}: not shown to work, counted as chance")
            else:
                ci = f" [95% range {row['auc_lo']:.2f} to {row['auc_hi']:.2f}]" if "auc_lo" in row else ""
                chance = "" if row.get("clear_of_chance", True) else ", not clearly better than chance"
                bits.append(f"{WRITING_LABEL[k]}: AUC {a:.2f} at this length{ci}{chance}")
        return f" Your writing is {kinds}. Accuracy by kind: " + "; ".join(bits) + "."

    def _evidence_head(self, st: dict, chunked: int) -> str:
        kinds = ", ".join(_plural(v, k) for k, v in sorted(st["by_kind"].items(), key=lambda kv: -kv[1]))
        s = f"from {kinds} you wrote, {st['words']:,} words"
        if st["excluded"]:
            s += f" ({_plural(st['excluded'], 'search')} left out: queries are not prose)"
        if chunked:
            s += f"; read as {chunked} chunks of about {int(self.card['features']['long_text_cap_words']):,} words (training-sized: the model was never tested on more per person) and averaged"
        if st["words"] < self.recommended:
            s += f". Below {self.recommended:,} words accuracy is still dropping, so trust this less"
        return s


@lru_cache(maxsize=4)
def _load(model_dir: str):
    import joblib

    d = Path(model_dir)
    card_p, model_p = d / "card.json", d / "model.joblib"
    if not card_p.exists() or not model_p.exists():
        only = {"text_mbti": " --only mbti", "text_bigfive": " --only bigfive"}.get(d.name, "")
        raise FileNotFoundError(
            f"no trained model in {d} ({'card.json' if not card_p.exists() else 'model.joblib'} missing). "
            f"The weights are not shipped with the public source; build them with: "
            f"cd py && uv run python scripts/train_text_readers.py{only} (datasets: see DATA.md)")
    return json.loads(card_p.read_text()), joblib.load(model_p)


class TextMBTIReader(_TextReader):
    """P(first letter) per MBTI axis from the person's own writing.

    value = P(I) for I/E, P(N) for N/S, P(T) for T/F, P(J) for J/P, with the training forum's
    lopsided base rates removed (a person with no telling words gets 0.5, not "77% I").
    low/high: 80% bootstrap range of that probability over the person's own items."""
    name = "text-mbti"
    kind = "mbti"

    def _estimate(self, t, s_point, s_boot, words_used, st, chunked) -> Estimate:
        h: Head = self.m["heads"][t]
        prior = logit(h.base_rate) if self.card["output_prior"] == "balanced" else 0.0
        p = float(sigmoid(s_point - prior))
        # Resampling duplicates items, and sublinear TF squashes duplicates, so the bootstrap cloud is
        # shifted; centre it on the point estimate (in logit space) and keep only its spread.
        s_boot = s_boot - np.median(s_boot) + s_point
        lo, hi = np.quantile(sigmoid(s_boot - prior), [(1 - INTERVAL_LEVEL) / 2, (1 + INTERVAL_LEVEL) / 2])
        V = self.validity_mix(t, words_used, st)
        auc, auc_in = V["auc"], V["auc_in"]
        # Off the forum the letter is predicted less well, so the calibrated forum probability is too bold:
        # keep the person's latent score and re-express it at the lower accuracy (same model the fuser uses).
        p, lo, hi = (shrink_p(x, auc_to_r(auc_in), auc_to_r(auc)) for x in (p, float(lo), float(hi)))
        first = t[0]
        ev = (f"{self._evidence_head(st, chunked)}. Chance you are {first} rather than {t[2]}: {p:.0%} "
              f"(80% range {lo:.0%} to {hi:.0%} if your texts were resampled). Tested on {self.card['data']['short']}: "
              f"AUC {auc_in:.2f} at this length (0.5 is a coin flip, 1.0 perfect).")
        if self.vbs and not self.vbs.get("tested_off_forum_with_mbti_labels", True):
            ev += (f" Untested off the forum: no MBTI-labelled writing from elsewhere is openly available, so the accuracy stated "
                   f"for your writing, AUC {auc:.2f}, is inferred from how the model relates to Big Five on student essays, "
                   f"and the chance above is pulled toward 50% to match.")
        return Estimate(reader=self.name, trait=t, kind="mbti", value=float(p), low=float(lo), high=float(hi),
                        validity=round(auc, 4), validity_kind="auc", evidence=ev, usable=True, interval_level=INTERVAL_LEVEL)


class TextBigFiveReader(_TextReader):
    """Big Five z-score from the person's own writing, with an 80% predictive interval.

    Essays model (binary median-split labels): the model gives P(above median). Assuming the
    trait and the model's score are jointly normal with correlation r (r implied by the tested
    AUC at this text length), the expected z is m = sqrt(1-r^2) * probit(P) and the person's
    true z lies in m +- 1.28 * sqrt(1 - r^2 + bootstrap variance) 80% of the time.
    PANDORA model (continuous labels): the head predicts z directly; the interval comes from the
    held-out residual quantiles at this length, widened by the bootstrap spread."""
    name = "text-bigfive"
    kind = "bigfive"

    def _estimate(self, t, s_point, s_boot, words_used, st, chunked) -> Estimate:
        h: Head = self.m["heads"][t]
        target = self.card["target"]
        tname = self.card["trait_names"][t]
        if target == "binary_median":
            V = self.validity_mix(t, words_used, st)
            if V["refuse"]:
                return self._refusal(t, tname, V, st, chunked)
            p_in = float(sigmoid(s_point - logit(h.base_rate)))
            auc, auc_in = V["auc"], V["auc_in"]
            # r_in comes from the closed-form auc_to_r of the in-domain AUC at this length, the same conversion
            # used for the ratio below and by aas.fusion when it turns the stated AUC into r. The card's
            # r_implied curve is a 200,000-draw simulation of the same relation and sits about 0.001 above it;
            # using one formula everywhere makes the r this reader states equal the r fusion reads back.
            r_in = auc_to_r(auc_in)
            # ratio < 1 when the person's kinds of writing are read less well than the essays: the expected z
            # shrinks toward 0 by it and the interval widens to the accuracy actually measured on that writing.
            ratio = min(max(auc_to_r(auc) / auc_to_r(auc_in), 0.0), 1.0) if auc_in > 0.5 else 1.0
            r = r_in * ratio
            s_in = sqrt(max(1 - r_in * r_in, 1e-6))
            m = ratio * s_in * probit(p_in)
            mb = ratio * s_in * np.array([probit(x) for x in sigmoid(s_boot - logit(h.base_rate))])
            s = sqrt(max(1 - r * r, 1e-6))
            sd = sqrt(s * s + float(np.var(mb)))
            lo, hi = m - Z80 * sd, m + Z80 * sd
            p = phi(m / s)
            ev = (f"{self._evidence_head(st, chunked)}. Chance your {tname} is above the median: {p:.0%}. "
                  f"The model was trained on yes/no (above/below median) labels from {self.card['data']['short']}, so the "
                  f"z shown is derived from that chance, not measured. Accuracy stated for your writing: AUC {auc:.2f} "
                  f"(about r = {r:.2f}); on student essays at this length it would be {auc_in:.2f}.{self._mix_text(V)}")
            if V["parts"] and any(k == "other_prose" for k, *_ in V["parts"]):
                ev += " Reddit-style prose has not been measured directly yet; its number is the lowest seen on other kinds of writing."
            validity, vkind = auc, "auc"
        else:  # continuous_z (PANDORA)
            m = float(s_point)
            r = self._curve(t, "r", words_used)
            q_lo = self._curve(t, "resid_q10", words_used)
            q_hi = self._curve(t, "resid_q90", words_used)
            spread = float(np.std(s_boot))
            lo, hi = m + q_lo - Z80 * spread, m + q_hi + Z80 * spread
            ev = (f"{self._evidence_head(st, chunked)}. Tested on {self.card['data']['short']}: r = {r:.2f} with "
                  f"questionnaire scores at this length.")
            validity, vkind = r, "r"
        return Estimate(reader=self.name, trait=t, kind="bigfive", value=float(m), low=float(lo), high=float(hi),
                        percentile=round(100 * phi(m), 1), validity=round(float(validity), 4), validity_kind=vkind,
                        evidence=ev, usable=True, interval_level=INTERVAL_LEVEL)


    def _refusal(self, t: str, tname: str, V: dict, st: dict, chunked: int) -> Estimate:
        ess = V["shares"].get("essay_like", 0.0)
        ev = (f"{tname}: not shown to work on this kind of writing, so no number is given. "
              f"{self._evidence_head(st, chunked)}.{self._mix_text(V)} On every other kind of writing we tested "
              f"(tweets, spoken vlogs, sitcom lines) the model's {tname} score was no better than chance (every 95% range "
              f"includes 0.5); it has only been shown to work on essay-like long prose, which is {ess:.0%} of your words "
              f"(at least {float(self.vbs.get('refuse_below_shown_share', 0.5)):.0%} needed).")
        return Estimate(reader=self.name, trait=t, kind="bigfive", value=float("nan"), validity=0.0, validity_kind="r",
                        evidence=ev, usable=False, interval_level=None)


def shrink_p(p: float, rho_in: float, rho: float) -> float:
    """Re-express a probability calibrated at accuracy rho_in (letter r) at a lower accuracy rho, keeping the
    person's latent score: P = Phi(rho u / sqrt(1 - rho^2)) with u = sqrt(1 - rho_in^2) probit(P_in) / rho_in."""
    if rho_in <= 0 or rho >= rho_in:
        return float(p)
    u = sqrt(max(1 - rho_in * rho_in, 1e-9)) * probit(p) / rho_in
    return phi(max(rho, 0.0) * u / sqrt(max(1 - rho * rho, 1e-9)))


READERS = {"text-mbti": TextMBTIReader, "text-bigfive": TextBigFiveReader}

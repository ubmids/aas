"""TextDistilledReader ("text-distilled"): Big Five from a person's writing, by a small model trained to give the
ratings Claude Haiku 4.5 gives, run entirely on this machine.

Built by scripts/distill.py (see docs/distill.md for the experiment, the pass rule and the result). The student
is TF-IDF and/or all-MiniLM-L6-v2 sentence embeddings (fastembed ONNX, on device) with a ridge head per trait.
No network call is made: the teacher was only ever asked about other people's public forum posts.

How one person is read
  1. own prose only, leak-stripped (readers.text.own_prose, the same as text-bigfive; searches left out)
  2. under the floor (card min_text.floor_words) -> usable=False with the reason
  3. the text is cut the way the teacher's input was cut (~1,500 tokens: a seeded random sample of whole items, or
     the first N words of one long text). A longer history is read as several such samples (different seeds) and
     the scores averaged
  4. score -> z against the student's own score distribution on the training pool (no labels involved)
  5. stated accuracy = the AUC measured on the kind of writing the person brought (card validity_by_source,
     word-weighted over kinds); the expected z is r * z_score and the 80% interval is +- 1.28*sqrt(1 - r^2), r implied
     by that AUC. Traits listed as refused in the card, or with a stated AUC at or below 0.5, are usable=False.
"""

from __future__ import annotations

import json
from functools import lru_cache
from math import sqrt
from pathlib import Path

import numpy as np

from ..person import Estimate, Person
from . import text as R

MODEL_DIR = R.MODELS_DIR / "text_distilled"


@lru_cache(maxsize=4)
def _load(model_dir: str) -> tuple[dict, dict]:
    import joblib

    d = Path(model_dir)
    if not (d / "card.json").exists() or not (d / "model.joblib").exists():
        raise FileNotFoundError(
            f"no trained model in {d} (model.joblib is not shipped with the public source). Build it with: "
            "cd py && uv run python scripts/distill.py --rate (needs ANTHROPIC_API_KEY and AAS_LLM_ALLOW_SPEND=1), "
            "then uv run python scripts/distill.py --evaluate. See docs/distill.md and DATA.md")
    return json.loads((d / "card.json").read_text()), joblib.load(d / "model.joblib")


def budget_samples(texts: list[str], budget_tokens: int, n: int) -> list[str]:
    """The teacher's cut (llm_rater.budget_text, seed 0), plus seeds 1..n-1 when the text is longer than one cut."""
    from .llm_rater import budget_text, estimate_tokens

    joined = "\n".join(t for t in texts if t)
    first, info = budget_text(joined, budget_tokens, 0)
    if not info["cut"]:
        return [first]
    k = min(n, max(1, int(estimate_tokens(joined) // budget_tokens)))
    return [first] + [budget_text(joined, budget_tokens, s)[0] for s in range(1, k)]


class TextDistilledReader:
    name = "text-distilled"
    kind = "bigfive"

    def __init__(self, model_dir: Path | str | None = None, exclude_kinds=R.EXCLUDED_KINDS, embedder=None, samples: int = 5):
        self.dir = Path(model_dir) if model_dir else MODEL_DIR
        self.card, self.m = _load(str(self.dir))
        self.exclude_kinds = frozenset(exclude_kinds)
        self.samples = samples
        self.floor = int(self.card["min_text"]["floor_words"])
        self.vbs = self.card["validity_by_source"]
        self._embedder = embedder
        self.vec = R.build_vectorizer(self.m["bow"]["vocab"], tuple(self.m["bow"]["ngram_range"])) if "bow" in self.m else None

    # -- features
    @property
    def embedder(self):
        if self._embedder is None:
            from ..slices.text import Embedder

            self._embedder = Embedder()
        return self._embedder

    def _embed(self, docs: list[str]) -> np.ndarray:
        c, mx = int(self.m["emb_chunk"]), int(self.m["emb_max_chunks"])
        out = []
        for t in docs:
            w = t.split()
            pieces = [" ".join(w[s:s + c]) for s in range(0, len(w), c)][:mx] or [""]
            out.append(np.asarray(self.embedder.embed(pieces), np.float32).mean(0))
        return np.stack(out)

    def _part(self, part: str, docs: list[str]) -> np.ndarray:
        p = self.m[part]
        if part == "bow":
            X = R.tfidf(self.vec.transform(docs), p["idf"])
            return np.asarray(X @ np.asarray(p["coef"], float).T) + p["intercept"]
        X = (self._embed(docs) - p["mean"]) / p["scale"]
        return X @ np.asarray(p["coef"], float).T + p["intercept"]

    def scores(self, docs: list[str]) -> np.ndarray:
        """Raw student scores (docs x 5), on the teacher's 1-5 scale for bow/emb, or a z-mix for mix."""
        if self.m["recipe"] == "mix":
            ref = self.m["mix_ref"]
            parts = [(self._part(k, docs) - ref[k]["mean"]) / ref[k]["sd"] for k in ("bow", "emb")]
            return (parts[0] + parts[1]) / 2
        return self._part(self.m["recipe"], docs)

    # -- validity
    def stated_auc(self, t: str, st: dict) -> tuple[float, dict]:
        wbw = {k: v for k, v in st.get("words_by_writing", {}).items() if v > 0} or {"other_prose": 1}
        tot = sum(wbw.values())
        shares = {k: v / tot for k, v in wbw.items()}
        auc = sum(s * float(self.vbs["kinds"][k][t]["auc"]) for k, s in shares.items())
        return auc, shares

    # -- Reader protocol
    def can_read(self, person: Person) -> tuple[bool, str]:
        _, st = R.own_prose(person, self.exclude_kinds)
        if st["words"] < self.floor:
            return False, (f"too little of your own writing: {st['words']:,} words; this reader needs at least {self.floor:,}")
        return True, f"{st['words']:,} words of your own writing"

    def read(self, person: Person) -> list[Estimate]:
        texts, st = R.own_prose(person, self.exclude_kinds)
        traits = list(self.card["traits"])
        if st["words"] < self.floor:
            why = self.can_read(person)[1]
            return [Estimate(reader=self.name, trait=t, kind="bigfive", value=float("nan"), usable=False, evidence=why,
                             interval_level=None) for t in traits]
        docs = budget_samples(texts, int(self.m["budget_tokens"]), self.samples)
        S = self.scores(docs)
        s = S.mean(0)
        z = (s - np.asarray(self.m["ref"]["mean"])) / np.asarray(self.m["ref"]["sd"])
        status = "" if self.card["verdict"]["pass"] else " This reader failed its pre-registered test and is not the default."
        out = []
        for j, t in enumerate(traits):
            auc, shares = self.stated_auc(t, st)
            mix = ", ".join(f"{R.WRITING_LABEL[k]} {v:.0%}" for k, v in sorted(shares.items(), key=lambda kv: -kv[1]))
            head = (f"from {st['words']:,} words of your own writing ({mix}), read as {len(docs)} sample(s) of about "
                    f"{int(self.m['budget_tokens'] / 1.33):,} words by a small model on this device that imitates an AI rater. ")
            if t in self.vbs.get("refused_traits", []) or auc <= 0.5:
                out.append(Estimate(reader=self.name, trait=t, kind="bigfive", value=float("nan"), usable=False,
                                    validity=0.0, validity_kind="r", interval_level=None,
                                    evidence=head + "Not shown to work on this kind of writing." + status))
                continue
            r = R.auc_to_r(auc)
            m = float(r * z[j])
            half = R.Z80 * sqrt(max(1 - r * r, 0.0))
            out.append(Estimate(reader=self.name, trait=t, kind="bigfive", value=m, low=m - half, high=m + half,
                                percentile=100 * R.phi(m), validity=round(auc, 4), validity_kind="auc",
                                interval_level=R.INTERVAL_LEVEL,
                                evidence=head + f"Tested AUC {auc:.2f} on this kind of writing (implied r {r:.2f}); "
                                                f"z derived for display." + status))
        return out

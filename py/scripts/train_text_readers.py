"""Rebuild the text readers from scratch: py/models/text_mbti/ and py/models/text_bigfive/.

    cd py && uv run python scripts/train_text_readers.py              # both
    uv run python scripts/train_text_readers.py --only mbti
    uv run python scripts/train_text_readers.py --only bigfive
    uv run python scripts/train_text_readers.py --only bigfive --bigfive-data pandora   # once PANDORA lands
    uv run python scripts/train_text_readers.py --only bigfive --bigfive-data pooled    # Essays + Friends + PAN15 (lost; see docs/text_crossdomain.md)

Data needed (both already on this machine):
  data/raw/mbti/*.parquet        see aas.slices.text.load_mbti for the download line
  data/raw/essays/essays.csv     curl -L -o data/raw/essays/essays.csv \
                                   https://raw.githubusercontent.com/SenticNet/personality-detection/master/essays.csv
  data/raw/models/               fastembed MiniLM cache (only for the embedding challenger in selection)

About 25 min on an M2 Pro (most of it tokenising 420k forum posts several times). Deterministic (seed 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aas.readers import text as R
from aas.readers import text_train as TT
from aas.schema import RAW

MODELS = Path(__file__).resolve().parents[1] / "models"
MBTI_TO_B5 = {"I/E": ("E", -1), "N/S": ("O", +1), "T/F": ("A", -1), "J/P": ("C", +1)}
B5_TO_MBTI = {"E": ("I/E", -1), "O": ("N/S", +1), "A": ("T/F", -1), "C": ("J/P", +1)}


def mbti_emb(c: TT.Corpus) -> np.ndarray | None:
    """Round 1's cached MiniLM means (20 sampled posts, default strip). A slightly weaker version of the
    family than a full rebuild would give; it lost by ~0.05 AUC in round 1, so this is enough to referee."""
    f = RAW / "mbti" / "cache" / "emb_stripped_k20_n8675.parquet"
    if not f.exists():
        TT.log("no cached MBTI embeddings; skipping the embedding challenger")
        return None
    e = pd.read_parquet(f)
    E = e.reindex(c.ids).to_numpy(np.float32, copy=True)
    bad = np.isnan(E).any(1)
    if bad.any():  # people with no posts left after stripping: give them the average embedding
        TT.log(f"{int(bad.sum())} people have no cached embedding; filled with the mean")
        E[bad] = np.nanmean(E[~bad], 0)
    return E


def essays_emb(c: TT.Corpus) -> np.ndarray | None:
    cache = RAW / "essays" / "cache" / "emb_minilm_chunk100.npy"
    if cache.exists():
        return np.load(cache)
    try:
        from aas.slices.text import Embedder

        emb = Embedder()
    except Exception as ex:  # noqa: BLE001
        TT.log(f"embedder unavailable ({ex}); skipping the embedding challenger")
        return None
    chunks, owner = [], []
    for i, it in enumerate(c.items):
        w = " ".join(it).split()
        for s in range(0, len(w), 100):
            chunks.append(" ".join(w[s : s + 100]))
            owner.append(i)
    TT.log(f"embedding {len(chunks)} essay chunks (MiniLM, ~3 min)")
    V = emb.embed(chunks, batch_size=32)
    owner = np.array(owner)
    E = np.stack([V[owner == i].mean(0) for i in range(len(c.items))]).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, E)
    return E


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["mbti", "bigfive"])
    ap.add_argument("--bigfive-data", choices=["essays", "pandora", "pooled"], default="essays",
                    help="pooled = every shippable Big Five domain (docs/text_crossdomain.md); essays = Essays only")
    ap.add_argument("--out", help="output folder for --bigfive-data pooled")
    ap.add_argument("--no-emb", action="store_true", help="skip the embedding challenger in selection")
    a = ap.parse_args()

    mbti_c = essays_c = None
    m_mbti = m_b5 = None
    if a.only in (None, "mbti"):
        mbti_c = TT.load_mbti_corpus()
        _, m_mbti = TT.train(mbti_c, MODELS / "text_mbti", "text-mbti", None if a.no_emb else mbti_emb(mbti_c))
    if a.only in (None, "bigfive"):
        if a.bigfive_data == "pandora":
            try:
                c = TT.load_pandora_corpus()
            except (FileNotFoundError, ValueError) as e:
                print(f"PANDORA: {e}")
                return 1
            TT.train(c, MODELS / "text_bigfive", "text-bigfive", None)
            return 0
        if a.bigfive_data == "pooled":
            # Not the shipped recipe: it lost the leave-one-domain-out test (docs/text_crossdomain.md). Kept so the
            # comparison can be rebuilt; writes to --out (default models/text_bigfive_pooled), never over the artifact.
            c = TT.load_pooled_corpus()
            TT.train(c, Path(a.out) if a.out else MODELS / "text_bigfive_pooled", "text-bigfive", None)
            return 0
        essays_c = TT.load_essays_corpus()
        _, m_b5 = TT.train(essays_c, MODELS / "text_bigfive", "text-bigfive", None if a.no_emb else essays_emb(essays_c), {
            "z_note": ("Labels are yes/no (above/below the sample median), so the model outputs P(above median). The z shown "
                       "is derived, for display: m = sqrt(1 - r^2) * probit(P), r implied by the tested AUC assuming the trait "
                       "and the model score are jointly normal. It is the expected z given the text, not a measured score."),
        })

    # Domain-shift probe: each model on the other dataset's people, against the loosely related label there.
    if m_mbti is not None and m_b5 is not None:
        TT.log("cross-domain: MBTI model on essays, Big Five model on forum users")
        x1 = TT.cross_domain(m_mbti, essays_c, MBTI_TO_B5)
        x2 = TT.cross_domain(m_b5, mbti_c, B5_TO_MBTI)
        for d, x, other in (("text_mbti", x1, "Essays (Big Five median-split labels)"), ("text_bigfive", x2, "PersonalityCafe (MBTI letters)")):
            p = MODELS / d / "card.json"
            card = json.loads(p.read_text())
            card["cross_domain"] = {"tested_on": other, "note": (
                "Model applied to another dataset and scored against the loosely related trait there "
                "(I~low E, N~O, F~A, J~C; |r| about 0.4-0.7 between the systems, McCrae & Costa 1989). "
                "Both a change of writing genre and a change of construct, so this is a lower bound on transfer."), "auc": x}
            p.write_text(json.dumps(card, indent=1, default=float))
            TT.log(f"{d} on {other}: {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

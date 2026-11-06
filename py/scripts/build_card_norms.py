"""Build src/aas/data/card_norms.json: the aggregate reference that `scripts/my_card.py` reads.

    cd py && uv run python scripts/build_card_norms.py

Reference population: the 636 ListenBrainz listeners of statements round 1 (303) and round 2's
confirmation set (333), the same people the identity round explored on. Their 57 family values on the
whole training weeks and on its two halves come from the per-person caches that
`scripts/run_statements_v2.py` keeps in the system temp dir (rebuilt from the local, gitignored bench if
missing). Nothing per person is written: only, per family, quantile knots (1st to 99th percentile, the
extremes left out as in music_norms.json), how many people could be measured, the split-half
correlation of normal scores, and for every grid threshold how many people each statement is true of
and the threshold's own mid-rank percentile; plus the 57 x 57 correlation of normal scores. See
docs/recognition_test.md, "How the card is made".
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from aas.statements.families_v2 import FAMILIES_V2
from aas.statements.identity import R_CEIL, R_FLOOR, normal_scores

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "aas" / "data" / "card_norms.json"
KNOTS = list(range(1, 100))


def _v2_values(which: str) -> dict:
    spec = importlib.util.spec_from_file_location("run_statements_v2", HERE / "run_statements_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.values(which)


def _r(x: float) -> float:
    return float(f"{x:.6g}")


def nearest_corr(C: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues and rescale to a unit diagonal."""
    C = np.nan_to_num((C + C.T) / 2)
    np.fill_diagonal(C, 1.0)
    w, V = np.linalg.eigh(C)
    C = (V * np.maximum(w, 1e-4)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / d[:, None] / d[None, :]


def main() -> None:
    old, fresh = _v2_values("old"), _v2_values("fresh")
    keys = [f.key for f in FAMILIES_V2]
    for d in (old, fresh):
        assert list(d["families"]) == keys
    full = np.vstack([old["full"], fresh["full"]])
    h1 = np.vstack([old["h1"], fresh["h1"]])
    h2 = np.vstack([old["h2"], fresh["h2"]])
    n = full.shape[0]
    halves = np.vstack([h1, h2])

    z = normal_scores(full, full)
    za, zb = normal_scores(halves, h1), normal_scores(halves, h2)
    fams = []
    for k, f in enumerate(FAMILIES_V2):
        col = full[:, k]
        ok = np.isfinite(col)
        r = col[ok]
        hv = halves[:, k][np.isfinite(halves[:, k])]
        both = np.isfinite(za[:, k]) & np.isfinite(zb[:, k])
        rf = float(np.corrcoef(za[both, k], zb[both, k])[0, 1]) if both.sum() > 2 else 0.0
        g = np.asarray(f.grid, float)
        srt = np.sort(r)
        pct_th = (np.searchsorted(srt, g, "left") + np.searchsorted(srt, g, "right")) / 2 / max(len(srt), 1)
        fams.append({
            "key": f.key,
            "measured": int(ok.sum()),
            "r": _r(float(np.clip(np.nan_to_num(rf), R_FLOOR, R_CEIL))),
            "full": [_r(v) for v in np.percentile(r, KNOTS)] if len(r) else [],
            "half": [_r(v) for v in np.percentile(hv, KNOTS)] if len(hv) else [],
            "grid": [_r(v) for v in g],
            "pct_th": [_r(v) for v in pct_th],
            "true_hi": [int(v) for v in (col[:, None] >= g).sum(0)] if "hi" in f.dirs else None,
            "true_lo": [int(v) for v in (col[:, None] <= g).sum(0)] if "lo" in f.dirs else None,
        })
    C = np.ma.corrcoef(np.ma.masked_invalid(z), rowvar=False).filled(0.0)
    C = nearest_corr(np.asarray(C))
    out = {
        "source": (f"{n} public ListenBrainz listeners (statements round 1, 303, and round 2's confirmation "
                   "set, 333), 57 round 2 families on each person's training weeks and its two halves. "
                   "Aggregates only: no per-person rows, no ids."),
        "n": n,
        "knots": KNOTS,
        "families": fams,
        "corr": [[round(float(v), 4) for v in row] for row in C],
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) from {n} listeners")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Figures for paper/identity_paper.md. Figures 1 to 4 use the exploration CSVs (the 636 seen
listeners); figure 5 uses the confirmation results on the third set (352 new listeners).

Run from the repo root:  uv run paper/figures/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "py" / "docs"
OUT = Path(__file__).resolve().parent

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "lines.linewidth": 2, "legend.frameon": False,
})


def k_curve():
    lu = pd.read_csv(DOCS / "identity_exploration_lineup.csv")
    on = pd.read_csv(DOCS / "identity_exploration_one_of_n.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    for sel, col, lab in (("joint", S1, "chosen together"), ("independent", S2, "round 2 rule")):
        d = lu[lu.selection == sel]
        a.fill_between(d.k, d.ci_lo * 100, d.ci_hi * 100, color=col, alpha=0.15, lw=0)
        a.plot(d.k, d.accuracy * 100, color=col, marker="o", ms=5, label=lab)
        a.annotate(f"{d.accuracy.iloc[-1] * 100:.1f}%", (7, d.accuracy.iloc[-1] * 100),
                   xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    a.axhline(20, color=INK2, lw=1, ls=":")
    a.text(1, 21.5, "chance 20%", color=INK2, fontsize=8)
    a.set(xlabel="sentences", ylabel="right person picked (%)", title="Lineup of five",
          ylim=(0, 100), xticks=range(1, 8), xlim=(0.7, 7.9))
    a.legend(loc="lower right")
    for m, col, lab in (("joint sentences", S1, "chosen together"), ("independent sentences", S2, "round 2 rule")):
        d = on[on.matcher == m].assign(k=lambda x: x.k.astype(int))
        b.plot(d.k, d.top1 * 100, color=col, marker="o", ms=5, label=lab)
    fp = on[on.matcher == "z_mahalanobis"].top1.iloc[0] * 100
    b.axhline(fp, color=S3, lw=1.5, ls="--")
    b.text(1, fp + 1.5, f"learned fingerprint, all 57 measures: {fp:.1f}%", color=INK, fontsize=8)
    b.set(xlabel="sentences", ylabel="right person ranked first (%)", title="One of 636",
          ylim=(0, 60), xticks=range(1, 8))
    b.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_k_curve.png", dpi=200)


def price():
    on = pd.read_csv(DOCS / "identity_exploration_one_of_n.csv")
    row = lambda m, k: on[(on.matcher == m) & (on.k.astype(str) == k)].top1.iloc[0] * 100
    bars = [
        ("1 sentence", row("joint sentences", "1"), S1),
        ("3 sentences, round 2 rule", row("independent sentences", "3"), S2),
        ("3 sentences", row("joint sentences", "3"), S1),
        ("7 sentences", row("joint sentences", "7"), S1),
        ("fingerprint, percentiles", row("percentile_mad", "all"), S3),
        ("fingerprint, learned distance", row("z_mahalanobis", "all"), S3),
        ("artists played (not readable)", row("items_hellinger", "all"), S4),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = range(len(bars))[::-1]
    ax.barh(list(y), [v for _, v, _ in bars], color=[c for *_, c in bars], height=0.62, edgecolor=SURF, lw=2)
    for yi, (_, v, _) in zip(y, bars):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(y), [n for n, *_ in bars])
    ax.set(xlim=(0, 105), xlabel="right person ranked first among 636 (%)",
           title="What readability costs (chance 0.16%)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_price_of_readability.png", dpi=200)


def decay():
    d = pd.read_csv(DOCS / "identity_exploration_decay.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    for col, c, lab in (("items_top1", S4, "artists played"), ("fingerprint_top1", S3, "learned fingerprint"),
                        ("sentences3_top1", S1, "3 sentences")):
        a.plot(d.gap_months, d[col] * 100, color=c, marker="o", ms=5, label=lab)
    a.set(xlabel="gap between windows (months)", ylabel="ranked first among 469 (%)", title="One of 469",
          ylim=(0, 100), xticks=[0, 1, 3, 6, 12])
    a.legend(loc="center right")
    for col, c, lab in (("fingerprint_lineup", S3, "learned fingerprint"), ("sentences3_lineup", S1, "3 sentences")):
        b.plot(d.gap_months, d[col] * 100, color=c, marker="o", ms=5, label=lab)
    b.axhline(20, color=INK2, lw=1, ls=":")
    b.text(0, 21.5, "chance 20%", color=INK2, fontsize=8)
    b.set(xlabel="gap between windows (months)", ylabel="right person picked (%)", title="Lineup of five",
          ylim=(0, 100), xticks=[0, 1, 3, 6, 12])
    b.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_time_decay.png", dpi=200)


def bits():
    d = pd.read_csv(DOCS / "identity_exploration_bits.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for sel, c, lab in (("joint", S1, "chosen together"), ("independent", S2, "round 2 rule")):
        x = d[d.selection == sel]
        ax.plot(x.k, x.bits_first_half, color=c, marker="o", ms=5, label=f"{lab}, when made")
        ax.plot(x.k, x.bits_independent_sum, color=c, ls=":", lw=1.5, label=f"{lab}, if independent")
        ax.plot(x.k, x.bits_surviving, color=c, ls="--", marker="s", ms=4, label=f"{lab}, surviving")
    ax.axhline(d.bits_needed.iloc[0], color=INK2, lw=1)
    ax.text(4.2, d.bits_needed.iloc[0] - 0.9, "needed for 1 of 636: 9.3 bits", color=INK2, fontsize=8)
    ax.set(xlabel="sentences", ylabel="bits", title="Bits of identity per set", xticks=range(1, 8), ylim=(0, 22))
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_bits.png", dpi=200)


def confirmation():
    """Figure 5: the third set. Curves from the confirmation CSVs; printed labels from the JSON."""
    import json
    conf = json.loads((DOCS / "identity_results.json").read_text())["confirmation"]
    lu = pd.read_csv(DOCS / "identity_confirmation_lineup.csv")
    ex = pd.read_csv(DOCS / "identity_exploration_lineup.csv")
    d = pd.read_csv(DOCS / "identity_confirmation_decay.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.6))
    for sel, col, lab in (("joint", S1, "chosen together"), ("independent", S2, "round 2 rule")):
        x = lu[lu.selection == sel]
        a.fill_between(x.k, x.ci_lo * 100, x.ci_hi * 100, color=col, alpha=0.15, lw=0)
        a.plot(x.k, x.accuracy * 100, color=col, marker="o", ms=5, label=lab)
        v = conf["lineup"][f"{sel}_k7"]["accuracy"] * 100
        a.annotate(f"{v:.1f}%", (7, v), xytext=(6, 0), textcoords="offset points", va="center",
                   color=INK, fontsize=9)
    x = ex[ex.selection == "joint"]
    a.plot(x.k, x.accuracy * 100, color=S1, lw=1, ls="--", alpha=0.6, label="chosen together, 636 explored")
    a.axhline(75, color=INK2, lw=1, ls="-.")
    a.text(1, 76.5, "primary bar 75% (3 sentences)", color=INK2, fontsize=8)
    a.axhline(20, color=INK2, lw=1, ls=":")
    a.text(1, 21.5, "chance 20%", color=INK2, fontsize=8)
    a.set(xlabel="sentences", ylabel="right person picked (%)", title="Lineup of five, 352 new listeners",
          ylim=(0, 100), xticks=range(1, 8), xlim=(0.7, 7.9))
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    n = int(d.people.iloc[0])
    for col, c, lab in (("items_top1", S4, "artists played, 1 of N"), ("fingerprint_lineup", S3, "fingerprint, lineup"),
                        ("sentences3_lineup", S1, "3 sentences, lineup"), ("fingerprint_top1", S3, "fingerprint, 1 of N"),
                        ("sentences3_top1", S1, "3 sentences, 1 of N")):
        ls = "-" if "lineup" in col or "items" in col else "--"
        b.plot(d.gap_months, d[col] * 100, color=c, marker="o", ms=4, ls=ls, label=lab)
    b.set(xlabel="gap between windows (months)", ylabel="right person found (%)",
          title=f"Decay, the same {n} people at every gap", ylim=(0, 100), xticks=[0, 1, 3, 6, 12])
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_confirmation.png", dpi=200)


if __name__ == "__main__":
    k_curve(); price(); decay(); bits(); confirmation()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))

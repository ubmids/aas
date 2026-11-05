# Ground truth: the IPIP Big Five quiz

Everything AAS predicts gets checked against a person's own answers to a
personality questionnaire. This layer is that questionnaire: which items, how
to score them, what "average" means, and how short the in-app quiz can be
before it stops being a trustworthy yardstick.

Code: `src/aas/quiz.py` (scoring, norms, `score()`), `scripts/build_ipip.py`
(rebuilds everything below), `scripts/take_quiz.py` (take it yourself),
`tests/test_quiz.py`. Every number here is in `docs/quiz_results.json`.

## Data

Open-Source Psychometrics IPIP-FFM dump (8 Nov 2018): 1,015,341 rows from a
free online test, 2016-2018, people who agreed at the end that their answers
were accurate and could be used. 50 items from Goldberg's "Big-Five Factor
Markers" (https://ipip.ori.org/newBigFive5broadKey.htm), each rated 1 to 5
(disagree, neutral, agree), plus milliseconds spent on each item, time on each
page, screen size, country and a duplicate-IP count. Mostly English-speaking
(US 54%, then GB, CA, AU).

The 151 MB zip was read without unpacking the 416 MB CSV and has been deleted.
What remains is `data/raw/ipip/ipip_clean.parquet` (21 MB, gitignored): the
cleaned answers as int8, eight timing summaries, screen size, country, start
hour and a fixed random train/held-out split. Norms are committed as
`src/aas/data/ipip_norms.json` (8 KB).

## Scoring key

Traits use the project letters. The dataset's EST scale is emotional
stability, so N (neuroticism) is EST with every direction flipped.

| trait | reverse-keyed items (score 6 - answer) |
|---|---|
| E | EXT2, 4, 6, 8, 10 |
| N | EST2 (relaxed), EST4 (seldom blue); the other 8 count toward N as answered |
| A | AGR1, 3, 5, 7 |
| C | CSN2, 4, 6, 8 |
| O | OPN2, 4, 6 |

Trait score = mean of the 10 keyed answers (1 to 5).

Proof the directions are right: for every item, correlate the raw answer with
the keyed total of the other nine items on its scale. All 21 reversed items
come out negative raw (e.g. CSN6 -0.59, OPN2 -0.53) and all 50 come out
positive after keying (weakest: OPN9 "I spend time reflecting on things"
0.30, AGR3 "I insult people" 0.34, CSN3 "I pay attention to details" 0.35).

## Cleaning

Applied in this order:

| rule | dropped | left |
|---|---:|---:|
| raw rows | | 1,015,341 |
| any item missing or 0 (0 = skipped) | 140,907 | 874,434 |
| IPC != 1 (more than one submission from that IP; the codebook's own advice for "max cleanliness") | 271,112 | 603,322 |
| per-item timing missing or not positive, or page time missing | 3,872 | 599,450 |
| median time per item under 1 s (too fast to have read the statement) | 1,713 | 597,737 |
| the same answer to all 50 items | 77 | 597,660 |
| 15 or more identical answers in a row, in the order shown on the page | 295 | 597,365 |

**597,365 kept (58.8%).** IPC is the big cut and the most debatable one: it
also throws out everyone on a shared university or office network. We keep
it because a duplicate person (someone retaking the test) is worse for norms
than losing honest strangers, and 597k is plenty. The speed and straight-line
rules remove few people, which suggests the "I answered accurately" consent
screen already filtered most junk. Very slow responders were kept: being
distracted does not make answers wrong.

## Norms (what "average" is)

Full 10-item scale, on the 1 to 5 mean scale:

| | O | C | E | A | N |
|---|---:|---:|---:|---:|---:|
| mean | 3.94 | 3.34 | 2.91 | 3.76 | 3.08 |
| SD | 0.62 | 0.74 | 0.91 | 0.74 | 0.86 |

People describe themselves as open and agreeable, so answering "3, neutral"
to everything lands well *below* average on O (z -1.53, about the 7th
percentile) and A (z -1.04), but right at average on E and N. Neutral is not
average; only the norms can tell you where you stand.

Trait intercorrelations are small (largest E-A 0.31, C-N -0.24, E-N -0.23),
as they should be for five separate traits.

The norms file holds, per trait: the item key, mean and SD, an exact
percentile for every possible raw sum (mid-rank: % scoring lower plus half of
% scoring the same), the same for the short form, and the item means and
covariance matrix, so z can be computed for any partial set of items.

`score(answers)` takes `{item code: 1..5}` and returns per trait
`{score, z, percentile, n_items, form}`. Form is `full` (all 10 items, exact
table), `short` (exactly the short-form items, exact table) or `partial`
(anything else; z from the stored means/covariances, percentile from the
normal curve). Traits with no answered items are left out.

## Reliability

Cronbach's alpha, full 10-item scales: **O 0.80, C 0.82, E 0.90, A 0.84,
N 0.87.** All good (0.8+ is the usual bar for research use) and in line with
published values for this inventory.

## Factor check

Varimax-rotated 5-factor analysis on a random 50,000 of the cleaned people
(keyed, standardised items). Correlation-matrix eigenvalues: 7.74, 5.03,
4.02, 3.56, 2.92, then a drop to 1.44, 1.29, 1.04: five clear factors, then a
cliff. **All 50 items load highest on their own trait's factor.** Own-factor
loadings run 0.37 to 0.80; the weakest are OPN9 (0.37), AGR10 "I make people
feel at ease" (0.37), CSN3 (0.38), EST4 (0.39); the largest cross-loading is
AGR2 "I am interested in people" at 0.36 on E, which makes sense.

## Short form (the in-app quiz)

Items chosen on the training half (299k people) by corrected item-total r,
then judged on the held-out half (298k people) the selection never saw.
"Balanced" means at least one forward and one reversed item per trait, so a
person who agrees with everything does not score high on everything.

Correlation of short-form score with the full 10-item score, held-out half:

| form | O | C | E | A | N |
|---|---:|---:|---:|---:|---:|
| 2 items, balanced | 0.82 | 0.84 | 0.89 | 0.88 | 0.85 |
| 3 items, balanced | 0.90 | 0.89 | 0.92 | 0.90 | 0.91 |
| **4 items, balanced (chosen)** | **0.91** | **0.92** | **0.95** | **0.92** | **0.94** |
| 4 random items (mean of 50 draws) | 0.88 | 0.89 | 0.93 | 0.90 | 0.92 |

Chosen 20 items: O = OPN10, OPN2, OPN1, OPN5; C = CSN6, CSN1, CSN5, CSN4;
E = EXT4, EXT5, EXT7, EXT2; A = AGR4, AGR9, AGR7, AGR5; N = EST8, EST6, EST1,
EST2. Balance only changed N (EST2 "relaxed" replaced EST7 "I change my mood a lot";
0.939 vs 0.931, so it helped as well).

Two honest caveats:

- These r's are *part-whole*: the 4 items are also 4 of the 10, so some of the
  agreement is the items agreeing with themselves. Against only the 6 items
  not in the short form, the 4-item score correlates 0.72 (O), 0.72 (C),
  0.83 (E), 0.74 (A), 0.81 (N). The truth about the "real trait" sits between.
- Alpha of the 4-item forms is 0.69 to 0.83, lower than the full scales, as it
  must be with fewer items.

**Why it matters for the whole project:** the short quiz is the ruler. If a
model predicted someone's *full-scale* trait perfectly, it could still only
correlate about 0.91 to 0.95 with their 4-item quiz score, because the quiz
itself is that noisy. So any r from a model checked against the short quiz is
capped there, and realistic footprint models (r 0.3 to 0.4 against full
scales) will read about 5 to 10% lower against the short quiz. A 2-item quiz
would cap at 0.82 to 0.89 and shave off more. For a validation study, use the
full 50 items; 20 items is fine for an in-app "how close were we" check.

## Bonus: does how fast you answer reveal who you are?

A first "digital trace to personality" toy. Features are response-time
behaviour only, never the answers: log median ms per item, spread of item
times (SD and IQR of log ms), fraction of items answered under 1.5 s, time on
the first item (includes reading the instructions), total test time, time on
the intro page and on the consent page. Ridge regression scored by
`evaluate.predict()` on a random 20,000 people, 5-fold x 5 repeats,
**100 permutations** (not 200, to keep it to about a minute per run; 50 would
not have been enough, see below), 1,000 bootstraps.

| features, sample | O | C | E | A | N |
|---|---:|---:|---:|---:|---:|
| timing only, everyone | 0.126 | 0.094 | 0.077 | 0.082 | 0.121 |
| timing only, US only | 0.142 | 0.094 | 0.069 | 0.070 | 0.131 |
| timing only, non-US | 0.139 | 0.073 | 0.078 | 0.088 | 0.092 |
| timing + screen size + start hour | 0.139 | 0.124 | 0.091 | 0.107 | 0.162 |

Re-run on 2026-11-05 with the per-repeat r of the 1 Nov `evaluate.predict` fix; every cell moved by 0.001 or
less (timing only, everyone, was 0.127, 0.095, 0.078, 0.083, 0.122) and no verdict changed.

Every cell: 95% CI about +/- 0.014, null_mean (what shuffled labels score)
-0.004 to -0.008, permutation p = 0.01 (the floor for 100 perms), Holm 0.05,
"survives". The signal is real. It is also small.

What carries it (univariate, all 597k people): people who answer faster score
higher on O (median item time r -0.11) and on N (share of sub-1.5 s answers
r +0.11). The O link survives inside the US alone, so it is not just
"non-native speakers read slower"; quick verbal processing going with the
intellect side of Openness is the likely story. Timing comes from taking a
personality test, so it is not an independent footprint: how long you dwell on
"I worry about things" is partly about your answer.

**Significant is not the same as useful.** With 597k people, time on the intro
page correlates r = -0.003 with Openness at p = 0.008: "significant", and
utterly meaningless. With 20k people anything above about r = 0.014 clears
p < 0.05. What matters is the size: r = 0.13 explains 1.7% of the variance
(r squared). The field's bar for digital footprints is r around 0.3 to 0.4
(9 to 16% of variance), and r = 0.1 is "weak". Timing alone is weak: it would
move a guess about one person by almost nothing. Its only use is as one small
input among many.

## Shared code

- `evaluate.predict` now warns when `len(traits) / (perms + 1) >= 0.05`. The smallest possible p is
  1/(perms+1) and Holm multiplies the smallest by 5, so with 5 traits 50 perms gives at best 0.098 and
  nothing can survive; the warning asks for at least 20 perms per trait.
- Still a suggestion, not built: a `ceiling` column (or helper) so slices checked against the short quiz
  can report r / quiz-ceiling alongside raw r.

## Rebuild

```
cd py
uv run python scripts/build_ipip.py --perms 100 --delete-raw   # ~5 min, downloads 151 MB
uv run pytest tests/test_quiz.py
uv run python scripts/take_quiz.py          # 20 items; --full for 50
```

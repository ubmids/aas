# Text slice: what people write

Question: from what someone writes, can we tell their personality, and how right is it really?

Code: `src/aas/slices/text.py`, `src/aas/slices/text_pandora.py`, runner `scripts/run_text.py`.
Numbers: `docs/text_results.csv` (every AUC with 95% CI), `docs/text_leak_stats.csv`,
`docs/text_lang_correlations.csv`. Reproduce with `cd py && uv run python scripts/run_text.py`
(about 20 min on a busy M2 Pro; embeddings are cached after the first run, so a rerun takes about 5 min).

## Data

**MBTI PersonalityCafe.** 8,675 forum users, their last ~50 posts each (422,845 posts), and the
MBTI type they reported. The original is Kaggle `datasnaek/mbti-type` (`mbti_1.csv`). With no Kaggle
login on this machine, we used the Hugging Face mirror `jingjietan/kaggle-mbti` (3 parquet files,
39 MB, Apache-2.0). We checked it is the same data: same 8,675 rows, row 0 is the known INFJ with
the same first posts, and its integer type codes decode to exactly the published per-type counts
(the loader checks this on every run). Its columns named O/C/E/A are really MBTI letters, not Big Five;
we ignore them. Big Five columns in our tables are NaN for this dataset.

Labels: each MBTI letter is a yes/no target, scored with `evaluate.classify()`, which gives
out-of-fold AUC. Share of people with the first letter: I 77%, N 86%, T 46%, J 40%.

**PANDORA** (Reddit, Big Five) needs an access request. The loader is written against the published
format and fails with a clear message until the files arrive (see the end of this doc).

## The leak

This is a forum *about* MBTI. People write their type all the time.

| fact | count |
|---|---|
| posts altered by leak stripping | 229,772 of 422,845 (54%, mostly links) |
| users who name some MBTI type | 8,319 of 8,675 (96%) |
| users who name **their own** type at least once | 7,803 of 8,675 (90%) |
| leak tokens left after stripping | 0 |

What `strip_leaks()` removes: URLs; the 16 types in any case, with plurals or possessives, and
wildcard partials (`INxJ`, `xNTP`, `ENFP-T`); temperament labels (NT, NF, SJ, SP...);
cognitive-function jargon (`Fi`, `Ne-dom`, `Fe/Ni`, `NiTe`); and words that name a letter or the
system (introvert/extrovert and variants, intuitive(s), sensor(s), thinker(s), feeler(s),
judger(s), perceiver(s), MBTI, Myers-Briggs, socionics, enneagram and `4w5`-style codes,
PerC/personalitycafe).

To measure how big the leak was, the "leak tokens only" model sees nothing except 21 counts: how
often a person writes each of the 16 type names, links, functions, and so on. That model alone
gets AUC 0.87 to 0.90. That is the inflated number you get by reading the answer off the page.

**Strict mode check.** After stripping, we fit a word-count model and read which words it leaned on.
Most were real style differences (below), but some theory jargon survived: "inferior", "dominant",
"intuition", "sensing", "type/typed", "sx", "doorslam". `strip_leaks(strict=True)` removes those
too. It changed the bag-of-words AUC by at most 0.007, so that leftover jargon is not what carries
the signal. Embeddings were not recomputed in strict mode (18 more minutes for a change this small).

## Features

**a) Interpretable language measures** (24 columns, `lang_*`, computed on all of a person's posts):

| feature | plain meaning |
|---|---|
| log words, words per post | how much they write |
| I-words, we-words, you-words | focus on self, on a group, on the listener |
| negation (not, never, n't) | disagreeing, denying |
| swear words | informality, low restraint |
| ? per post, ! per post | asking questions; enthusiasm |
| type-token ratio (100-word windows) | vocabulary variety (windowed because raw TTR falls the more you write) |
| average word length | formal or abstract vocabulary |
| tentative, certain, and their difference | hedging ("maybe, probably") versus sureness ("definitely, clearly") |
| 10 NRC EmoLex rates | share of words tagged anger, anticipation, disgust, fear, joy, sadness, surprise, trust, positive, negative |

The emotion lexicon is the NRC Word-Emotion Association Lexicon v0.92 (Mohammad and Turney), the
official file, 2.6 MB. It has no stemming, so "loved" can be missed. **Its licence is free for
non-commercial research only**: commercial use needs a licence from NRC.

**b) Sentence embeddings** (384 columns, `emb_*`). We run `all-MiniLM-L6-v2` through `fastembed`
(ONNX runtime, CPU, no torch). The model is 87 MB and the dependencies add 119 MB. Each post becomes
384 numbers, and a person is the average of their posts. **We sampled 20 posts per user** (seed 0,
the same posts for raw and stripped text) to fit the time budget. That is 260,679 unique texts in
18 minutes at about 240 posts/s on a shared machine. Using all ~50 posts would likely add a little.
Only per-person averages are stored (`data/raw/mbti/cache/`, 36 MB); storing per-post vectors would
have taken about 540 MB.

**Reference: bag of words.** TF-IDF over up to 30k words plus logistic regression with fixed C=1.
The vocabulary is learned inside each training fold. This is the classic method in papers on this
dataset. It is not one of our two families; it is here as a yardstick and as a leak probe.

## Results

AUC on held-out people (5-fold, out-of-fold; 0.5 = coin flip, 1.0 = perfect). N = 8,675.
95% CIs (Hanley-McNeil) are about ±0.01 for every cell; full CIs are in `text_results.csv`.

| text | features | I/E | N/S | T/F | J/P |
|---|---|---|---|---|---|
| raw | leak tokens only (21 counts) | 0.879 | 0.898 | 0.894 | 0.867 |
| raw | interpretable | 0.654 | 0.590 | 0.760 | 0.588 |
| raw | embeddings | 0.822 | 0.835 | 0.887 | 0.778 |
| raw | both | 0.827 | 0.834 | 0.891 | 0.779 |
| raw | bag of words (ref) | 0.881 | 0.898 | 0.924 | 0.850 |
| **stripped** | **interpretable** | **0.660** | **0.591** | **0.759** | **0.592** |
| **stripped** | **embeddings** | **0.721** | **0.696** | **0.826** | **0.637** |
| **stripped** | **both** | **0.730** | **0.701** | **0.834** | **0.641** |
| stripped | bag of words (ref) | 0.774 | 0.779 | 0.879 | 0.695 |
| strict | bag of words (ref) | 0.773 | 0.772 | 0.878 | 0.694 |

The bold rows are the real numbers.

How much the leak inflated each family (raw minus stripped):
- embeddings: +0.10 (I/E), +0.14 (N/S), +0.06 (T/F), +0.14 (J/P)
- bag of words: +0.11, +0.12, +0.05, +0.16
- interpretable: about 0. These counts never look at type names, so there is nothing for them to leak.

What the plain measures say (correlation with the letter on stripped text, whole sample; descriptive,
not a prediction score):
- **T vs F** has the clearest signal. Thinkers write fewer "I" words (r = -0.22), fewer joy words
  (-0.23), fewer exclamation marks (-0.20), longer words (+0.19), more swearing (+0.16), and more
  negation (+0.15).
- **I vs E**: extraverts use more "!" (-0.14 for I), more questions (-0.11), and more "you" and "we";
  introverts use slightly more sadness words (+0.09).
- **N vs S and J vs P**: no single plain measure moves more than about 0.1. Their signal, where it
  exists, is in *what* people talk about. The word model's top words: N "ideas, universe, philosophy,
  science, books" vs S "car, sports, practical, mechanic, driving"; P "lazy, weed, fun, creative,
  chaotic" vs J "plan, plans, control, purpose".

## Verdict

After the leak is removed, writing still tells you something real. T/F is clearly readable
(AUC 0.83 to 0.88), and I/E and N/S are modest (0.70 to 0.78). J/P is weak (0.64 to 0.70). The leak
was large: just counting type names scores 0.87 to 0.90, and it inflated embedding scores by 0.06 to
0.14. Simple word counts beat a small sentence-embedding model here, because forum topic and
vocabulary carry most of the signal. One caveat: these labels are self-reported by people steeped in
type theory, so some of the "signal" may be people writing to their type's stereotype. That is a
weaker claim than "this is your personality".

**Compared with prior work.** Many papers on this dataset report per-letter accuracy above 80 to 90%.
Two problems explain most of that. First, accuracy is easy to fake when classes are lopsided: always
guessing "N" is 86% "accurate", and always guessing "I" is 77%. Second, the type-word leak above.
A University of Malaya paper (eprints.um.edu.my/27576) explicitly flags leakage in this dataset and
calls earlier results overly optimistic; work on it consistently finds T/F and N/S easier than J/P,
which matches us. We report AUC, which is not fooled by lopsided classes, and our leak-free numbers
are in the modest range the careful work points to. Our T/F is on the strong end. We could not verify
exact published leak-free AUCs offline, so we do not quote them.

## Things to know

- We used `classify()` as is. It returns no CI and no out-of-fold predictions, so the CIs come from
  the Hanley-McNeil formula on the AUC and class sizes (`text.auc_ci`). A suggested change to
  `evaluate.py`: return the OOF predictions, and add a bootstrap CI and permutation p like
  `predict()` has.
- `data/processed/mbti_pc/` holds the **stripped** features only (24 lang + 384 emb). The raw-text
  embeddings live in the cache for the leak comparison and must not be used as features.
- `classify` fits on everyone in one pass, with no grouping. Fine here: one row per user.

## PANDORA (ready, waiting on data)

`text_pandora.py` expects `data/raw/pandora/author_profiles.csv` (Big Five percentiles for about 1.6k
authors) and `all_comments_since_2015.csv` (several GB, beyond this machine's disk budget today).
It does the following:
- z-scores the Big Five within PANDORA
- streams the comment file in 500k-row chunks, keeping at most 1,000 comments per labelled author
- **drops comments in personality subreddits** (r/mbti, r/INTJ, r/BigFive, r/Enneagram, ...)
- strips MBTI leaks plus Big Five self-report words ("openness", "neurotic", "percentile", "IPIP",
  "truity", ...), because the labels came from people posting their own test results
- builds the same two feature families and scores O/C/E/A/N with `evaluate.predict()`
  (Pearson r, permutation p, bootstrap CI, Holm)

Run it with `uv run python scripts/run_text.py --dataset pandora`. It was tested end to end on a
synthetic file set; with no files present it exits with code 1 and tells you how to get access.
Column names are matched case-insensitively with aliases. If the release differs, the error lists the
columns it found.

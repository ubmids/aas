# Text readers: trained, saved, run on one person

Two readers turn what a person writes into personality estimates:

| reader | output | trained on | artifact |
|---|---|---|---|
| `text-mbti` | P(first letter) for I/E, N/S, T/F, J/P | 8,675 PersonalityCafe forum users | `models/text_mbti/` (0.4 MB) |
| `text-bigfive` | Big Five z with an 80% interval | 2,467 student essays (Pennebaker & King 1999) | `models/text_bigfive/` (0.6 MB) |

Code: `src/aas/readers/text.py` (the readers), `src/aas/readers/text_train.py` (training),
`scripts/train_text_readers.py` (rebuilds both from scratch, about 5 minutes, seed 0),
`tests/test_readers_text.py`. Every number below is also in each model's `card.json`.

```python
from aas.readers.text import TextMBTIReader, TextBigFiveReader
ok, why = TextMBTIReader().can_read(person)
estimates = TextMBTIReader().read(person)       # list[Estimate]
```

Artifacts total 1 MB, far under the 50 MB budget, so they are kept in the repo (no gitignore).
The train script is still the source of truth: it rebuilds them exactly.

## Data and licences

- **MBTI PersonalityCafe** (Kaggle `datasnaek/mbti-type`), via the Hugging Face mirror
  `jingjietan/kaggle-mbti` (Apache-2.0 tag; scraped public forum text). Already on disk from round 1.
- **Essays** (Pennebaker & King 1999, J Pers Soc Psychol 77(6):1296). University of Texas students
  writing 20 minutes of stream of consciousness, 1997 to 2004, with Big Five from a questionnaire,
  **median-split into yes/no** per trait. Copy used:
  `https://raw.githubusercontent.com/SenticNet/personality-detection/master/essays.csv`
  (8.2 MB, sha256 `3e0ec809...f7bc7f`, checked on every training run), the file distributed with
  Mairesse et al. (2007) and Majumder et al. (2017). It is a **research dataset with no explicit data
  licence**; the mirror repo's MIT licence covers its code, not the essays. Fine for this research;
  check before any commercial use. Encoding is cp1252.

## How the pipeline was chosen (before any final fit)

People are split once: 80% dev, 20% test (stratified, seed 0). On dev only, 5-fold CV compares:

- bag of words: word counts (unigrams, or unigrams + bigrams) capped at 20,000 terms, sublinear TF-IDF,
  logistic regression with C in {0.1, 0.3, 1, 3}. One vectorizer shared by all traits (best mean AUC),
  C picked per trait.
- the challenger from round 1: MiniLM sentence-embedding means + logistic regression.

Then the chosen pipeline is fit on dev, calibrated on dev, and scored once on the untouched test people.
The saved model is refit on everyone; the card's numbers come from the dev-only fit, so they are
honest held-out numbers.

**MBTI**: unigrams won; C = 1 (I/E, N/S, J/P), 3 (T/F). Embeddings lost on every letter by 0.05 to 0.07.

| letter | dev 5-fold CV AUC [95% CI] | held-out test AUC [95% CI] | embeddings (dev CV) |
|---|---|---|---|
| I/E | 0.757 [0.745, 0.769] | **0.787** [0.762, 0.812] | 0.710 |
| N/S | 0.769 [0.753, 0.784] | **0.778** [0.746, 0.809] | 0.698 |
| T/F | 0.878 [0.869, 0.885] | **0.878** [0.860, 0.894] | 0.826 |
| J/P | 0.681 [0.669, 0.695] | **0.709** [0.685, 0.734] | 0.622 |

These match round 1's stripped bag-of-words numbers (0.77 / 0.78 / 0.88 / 0.70); the stricter leak strip
used here (theory jargon and Big Five words too) costs nothing measurable.

**Big Five (Essays)**: unigrams + bigrams won; C = 0.1 (O), 1 (C, E), 0.3 (A, N). Embeddings lost
on every trait, narrowly (by 0.004 to 0.03).

| trait | dev 5-fold CV AUC [95% CI] | held-out test AUC [95% CI] | implied r | embeddings (dev CV) |
|---|---|---|---|---|
| O | 0.670 [0.644, 0.694] | **0.682** [0.633, 0.730] | 0.40 | 0.666 |
| C | 0.612 [0.586, 0.637] | **0.573** [0.520, 0.627] | 0.16 | 0.583 |
| E | 0.608 [0.582, 0.633] | **0.594** [0.546, 0.645] | 0.21 | 0.605 |
| A | 0.569 [0.542, 0.595] | **0.584** [0.533, 0.638] | 0.19 | 0.554 |
| N | 0.616 [0.590, 0.640] | **0.585** [0.535, 0.632] | 0.19 | 0.589 |

Openness is the only trait with a real effect (vocabulary and topics of curious people). The others are
barely above chance, in line with published work on this dataset (accuracy 55 to 62%). The test set is
494 people, hence the wide CIs.

## Calibration

A model is **calibrated** when its probabilities mean what they say: of all the people it gives 70%,
about 70% really are that letter. Ranking people well (AUC) and stating the right confidence are separate
skills; logistic regression on 20,000 word features ranks well but is overconfident on short texts and
miscalibrated across lengths.

So after the head, each trait gets a **length-aware Platt recalibration**:
`logit P = a*z + b*z*log(words/ref) + c*log(words/ref) + d`, fit on dev people's out-of-fold scores with
their text cut to 25, 50, ... words and full length. The `b` term lets the model be bolder when it has
more text, and humbler when it has less.

Measured on held-out test people (ECE = average gap between predicted and observed rate over 10 bins;
the "noise floor" is the ECE a perfectly calibrated model would still show on this many people):

| | ECE calibrated | ECE before | noise floor | Brier | Brier of guessing the base rate |
|---|---|---|---|---|---|
| I/E | 0.028 | 0.065 | 0.021 | 0.144 | 0.178 |
| N/S | 0.021 | 0.042 | 0.015 | 0.102 | 0.119 |
| T/F | 0.027 | 0.041 | 0.022 | 0.142 | 0.248 |
| J/P | 0.024 | 0.042 | 0.024 | 0.208 | 0.239 |
| O | 0.043 | 0.101 | 0.044 | 0.226 | 0.250 |
| C | 0.064 | 0.034 | 0.040 | 0.246 | 0.250 |
| E | 0.022 | 0.023 | 0.039 | 0.243 | 0.250 |
| A | 0.018 | 0.024 | 0.035 | 0.244 | 0.249 |
| N | 0.064 | 0.020 | 0.042 | 0.246 | 0.250 |

MBTI: calibration halves the error and lands near the noise floor at every text length (ECE 0.01 to 0.04
from 25 words to full). Big Five: everything sits around the noise floor of a 494-person test set; C and N
read slightly worse after recalibration (0.064 vs a 0.04 floor), which on this sample size is within noise
but is flagged here rather than hidden. Full reliability tables are in the cards.

**Prior.** PersonalityCafe is 77% I and 86% N, which says who joins a typology forum, not who people are.
The MBTI reader removes the training base rate before reporting (`output_prior: "balanced"`), so text with
no telling words gives 0.5, not "77% introvert". Calibration above was measured at the forum's own prior,
where it can be checked.

## How much text is enough

Held-out people's writing was cut down to k words (random items until k is reached, 10 repeats) and scored
against the same people's full-text AUC.

MBTI, mean over the four letters:

| words | 25 | 50 | 100 | 200 | 300 | 400 | 600 | **800** | 1200 | full (~1,300) |
|---|---|---|---|---|---|---|---|---|---|---|
| I/E | .59 | .62 | .65 | .69 | .71 | .72 | .75 | .76 | .79 | .79 |
| N/S | .58 | .60 | .62 | .66 | .69 | .70 | .73 | .76 | .78 | .78 |
| T/F | .64 | .67 | .71 | .76 | .80 | .82 | .85 | .86 | .88 | .88 |
| J/P | .56 | .58 | .60 | .63 | .64 | .66 | .68 | .69 | .72 | .71 |
| drop vs full | .20 | .17 | .14 | .11 | .08 | .06 | .04 | **.024** | .008 | |

Big Five (full is ~620 words; essays are short so the curve is short):

| words | 25 | 50 | **100** | 150 | 200 | 300 | 400 | full |
|---|---|---|---|---|---|---|---|---|
| O | .61 | .62 | .64 | .66 | .67 | .67 | .68 | .68 |
| mean drop vs full | .044 | .040 | **.028** | .020 | .010 | .006 | .003 | |

Rules (fixed in code, `pick_min_text`): **recommended** = shortest length whose mean AUC is within 0.03 of
full text and stays so; **floor** = shortest length keeping at least half of the full-text lift above chance.

| | floor (below: usable=False) | recommended (above: full accuracy) |
|---|---|---|
| text-mbti | **100 words** | **800 words** |
| text-bigfive | **25 words** | **100 words** |

Between floor and recommended the reader still answers, says accuracy is still dropping, and reports the
AUC from this curve at the person's own length as `validity`. The Big Five floor is low only because there
is little accuracy to lose: its full-text signal is already weak.

## What one Estimate contains

- **text-mbti**: `value` = P(I), P(N), P(T), P(J). `low/high` = 80% range of that probability when the
  person's own items are bootstrap-resampled (200 resamples, centred on the point estimate), i.e. how much
  the answer depends on which texts we happened to get. `interval_level = 0.80` refers to that resampling
  range; it is not a coverage claim about the letter itself. `validity` = held-out AUC at their length, for the
  kind of writing they brought (see "Validity by kind of writing" below; off the forum it is inferred, untested).
- **text-bigfive**: the model gives P(above median). Assuming trait and model score are jointly normal with
  correlation r (r implied by the tested AUC at that length; table above), the **expected z** is
  `m = sqrt(1 - r^2) * probit(P)` and the person's true z is in `m +- 1.28 * sqrt(1 - r^2 + bootstrap var)`
  80% of the time (`interval_level = 0.80`). Plain probit(P) would overstate z by 1/sqrt(1-r^2), only 1 to
  9% here. The z is **derived for display**, not measured; the evidence string says so and gives P.
  Intervals are honestly wide (about +-1.25 z) because r is 0.16 to 0.40. `percentile = 100*Phi(m)`.
  This coverage can't be checked on Essays (only yes/no labels); `profile.evaluate()` checks it against
  quiz scores.
- `evidence`: e.g. "from 1,240 comments you wrote, 58,312 words (14 searches left out: queries are not
  prose); read as 36 chunks of about 1,609 words ... Tested on 8,675 PersonalityCafe forum users: AUC 0.79".
- Too little text: `can_read` returns False with the reason; `read` returns one `usable=False` Estimate
  per trait with the same reason.

**Long histories.** Training people had at most ~1,600 (MBTI) / ~1,000 (Essays) words (90th percentile).
TF-IDF vectors of 50,000 words look different from anything the model saw, so beyond 1.5x that the reader
cuts the text into training-sized chunks, scores each and averages. That under-uses the extra text
(slightly humble probabilities) rather than extrapolating; validity is capped at the full-text AUC.

**Leak stripping** (`readers.text.clean`, the same function for training and for new people): MBTI strict
strip (type names incl. INxJ / ENFP-T, temperaments, Fi/Ne-dom function jargon, introvert/extravert and
friends, enneagram codes, typology words, URLs) plus Big Five self-report words (openness, neurotic,
percentile, IPIP, truity ...). A Reddit user who writes "INTJ here, 95th percentile openness" gets exactly
the same reading as without it (tested).

## Why searches are left out

Google searches are 2 to 4 word keyword strings ("cheap flights lagos"). The models learned from sentences:
function words (I, we, not, maybe), tone and topic in running prose. A query has almost none of that, and
the few content words it has are about the task of the moment, not style. With no labelled search data we
cannot show it works, so `kind == "search"` is excluded by default and the evidence says how many were
dropped. `TextMBTIReader(exclude_kinds=())` opts back in, for research only. Searches are better read by a
separate interests model, tested on its own.

## Domain shift: what will degrade, and how we'll measure it

The models learned from forum posts by typology fans and from 1990s student essays. New people bring Reddit
comments, tweets, YouTube comments. Expected damage, roughly worst first:

1. **Topic and era vocabulary.** Bag of words leans on topics: Essays learned "roommate, class, boyfriend,
   Austin"; MBTI learned "philosophy, universe" vs "car, sports". A 2026 Reddit user's topics follow their
   subreddits and slang the 1999 vocabulary never saw. Expect the biggest loss on Big Five.
2. **Construct.** MBTI labels are self-typed by people steeped in the theory, so part of the "signal" is
   writing like your type's stereotype. Ordinary people do less of that.
3. **Genre and length.** Tweets are short and full of handles, hashtags, emoji. Many short items is fine (we
   pool them) but genre words differ.
4. **Leaks we can't see.** Without the subreddit, a comment in r/intj is stripped of type names but still
   talks the theory. Proposed contract change below.
5. **Priors.** Handled for MBTI (balanced output); Essays is median-split, so roughly balanced already.

Measured now, a first lower bound (each model run on the other dataset and scored against the related
trait there, McCrae & Costa 1989 links I~low E, N~O, F~A, J~C at |r| 0.4 to 0.7; this changes genre AND
construct at once):

| model on other data | I/E -> E | N/S -> O | T/F -> A | J/P -> C |
|---|---|---|---|---|
| text-mbti on Essays (n=2,467) | 0.584 | 0.589 | 0.552 | 0.556 |

| model on other data | E -> I/E | O -> N/S | A -> T/F | C -> J/P |
|---|---|---|---|---|
| text-bigfive on forum users (n=8,675) | 0.602 | 0.596 | 0.665 | 0.568 |

All above chance (CIs exclude 0.5), all well below in-domain. So the signal transfers, weakly. Real new-domain
accuracy is somewhere between this floor and the in-domain numbers.

**Measured on other Big Five datasets (2026-11-01, `text_crossdomain.md`).** Leave-one-domain-out over Essays,
FriendsPersona (sitcom lines, observer-rated), PAN 2015 (tweets, self-report) and YouTube vlogs (spoken, observer-rated).
The current Essays model, on datasets it never saw: vlogs mean AUC 0.59 (A 0.70, C 0.61, N 0.61, E 0.56, O 0.47),
tweets 0.54 (only N clears chance alone, 0.57), Friends 0.48 (chance). Pooling the open datasets, domain-stable
vocabulary, style-only features and MiniLM embeddings all did worse on held-out domains (best 0.514 vs 0.539), so
the artifact was kept; the per-domain scores are in its card under `held_out_domains`. Dataset survey and licences:
`text_datasets.md`.

## Validity by kind of writing (2026-11-01)

Until today both readers stated the accuracy they reached on their own training data (student essays,
a typology forum). Real users bring tweets and Reddit comments, where the measured accuracy is much lower
(`text_crossdomain.md`). The fuser and the study's power analysis both took the stated number at face
value. Now each reader states the accuracy **measured on the kind of writing it actually read**, from a
`validity_by_source` block in its card.

**How an item is sorted** (`readers.text.kind_of_writing`):

| kind | rule |
|---|---|
| tweets | source `x` / `twitter`, or kind `tweet`, `reply`, `retweet`, `quote` |
| essay-like long prose | kind `essay`, `journal`, `diary`, `blog`, `letter`, or any other single item of 300+ words (about the shortest 5% of the Essays) |
| other prose | everything else: Reddit comments, YouTube comments, short posts (searches are left out before this) |

When a person brings several kinds, the stated AUC per trait is the **word-weighted mix**. The off-domain
numbers were measured on whole histories; for a shorter text their lift above 0.5 is scaled down by the
same share the in-domain length curve loses at that length.

**text-bigfive: AUC stated per kind (full length)**

| trait | essay-like (in domain, Essays test) | tweets (PAN 2015, measured) [95% CI] | other prose / Reddit (stand-in) |
|---|---|---|---|
| O | 0.68 | **refused** (0.52 [0.46, 0.59]) | **refused** |
| C | 0.57 | 0.54 [0.47, 0.61], not clear of chance | 0.54 |
| E | 0.59 | 0.54 [0.47, 0.62], not clear of chance | 0.54 |
| A | 0.58 | 0.54 [0.47, 0.61], not clear of chance | 0.54 |
| N | 0.59 | **0.57** [0.51, 0.64] | 0.57 |

As r (what the fuser uses, the closed-form `auc_to_r` of the AUC): essay-like 0.40 for O and 0.16 to 0.21 for C to N;
tweets and Reddit 0.097, 0.095, 0.094, 0.158 for C, E, A, N. For E that is AUC 0.5427 on the card, r 0.095. The card also stores the Pearson r measured directly on PAN
(0.072 for E); nothing downstream uses it, so 0.095 is the one value stated for E on tweets.

- **Reddit is not measured yet.** The stand-in is, per trait, the lower of the two held-out domains made of a
  person's own words (PAN 2015 tweets, self-report; YouTube vlogs, spoken, observer-rated). It is the PAN number
  for every trait. FriendsPersona is left out of the minimum because its text is screenwriters' dialogue for a
  fictional character, 30 words a scene; including it would put every trait at chance. The evidence string says
  Reddit is not measured. PANDORA (Reddit, self-report) will replace this row.
- **Openness is refused** off the essays: its held-out-domain AUC 95% CI includes 0.5 on every held-out domain
  (Friends 0.50, PAN 0.52, vlogs 0.47). The reader returns `usable=False`, `value = NaN`, `validity = 0`
  (`validity_kind = "r"`) and the reason "not shown to work on this kind of writing", unless at least half of the
  person's words are essay-like. In a mix, a refused kind counts as AUC 0.5.
- **Shrinking and widening.** The Essays model's chance P(above median) is turned into the expected z as before,
  then scaled by `ratio = r(stated AUC) / r(in-domain AUC)`, and the interval uses `sqrt(1 - r_stated^2)` instead
  of the in-domain r. The same words labelled as tweets give a z closer to 0 and a wider interval than labelled as
  an essay. With r about 0.1 the 80% interval is about +-1.27 z: close to "anyone's" interval (+-1.28).
  For essay-like text nothing changes from before.
- C, E and A on tweets are kept (usable) even though their own CI includes 0.5, because they clear chance on the
  vlogs; the evidence says "not clearly better than chance" on tweets.

**text-mbti: untested off the forum.** No MBTI-labelled writing from outside PersonalityCafe is openly available
today: PANDORA and MBTI9k (Reddit flair) need access requests, the Twitter MBTI sets (Plank & Hovy 2015) are tweet
IDs only, and the open Hugging Face "MBTI" sets are PersonalityCafe copies, LLM-made, or of unknown provenance.
Nothing was downloaded (disk had 3.5 GB free). So the off-forum accuracy is **inferred**: the model's AUC against
the linked Big Five trait on 2,467 student essays, turned into r and divided by the published MBTI to Big Five link
(McCrae & Costa 1989). The same number is used for every kind of writing, and the evidence says it is untested.

| letter | in domain (forum test) AUC / letter r | off forum, inferred AUC / letter r | share of in-domain r kept |
|---|---|---|---|
| I/E | 0.79 / 0.62 | **0.61** / 0.25 | 41% |
| N/S | 0.78 / 0.60 | **0.62** / 0.27 | 46% |
| T/F | 0.88 / 0.79 | **0.62** / 0.26 | 33% |
| J/P | 0.71 / 0.46 | **0.62** / 0.26 | 56% |

The probability shown is pulled toward 50% to match (same latent score, re-expressed at the lower accuracy:
`shrink_p`). Spot checks of the same probe on the held-out Big Five domains (in the card, not used): tweets I/E
0.58, N/S 0.54, T/F 0.56, J/P 0.48 (n 294, wide CIs); vlogs 0.64, 0.52, 0.63, 0.61; Friends all at or below
chance. So on tweets the essays-based inference may be generous for N/S and J/P.

Reproduce the MBTI spot checks: score `models/text_mbti` raw head scores on `text_train.load_b5_domain(d)["docs"]`
for d in pan15, vlogs, friends, against the linked trait's median split (I/E and T/F sign-flipped), with
`text_crossdomain.score_block` (1,000 bootstraps and shuffles, seed 0).

How we'll measure it properly:
- **PANDORA** (Reddit + Big Five + MBTI flair): score the Essays and MBTI models on Reddit authors *before*
  retraining. That is the direct Reddit transfer number. Then retrain (below).
- **Real people** with the quiz, through `profile.evaluate()`: interval coverage (should be ~0.80), MAE vs
  guessing average, broken down by source (Reddit-only vs tweets-only).
- **Consistency across sources** for the same person: if Reddit and tweets disagree far more than the
  bootstrap ranges allow, the shift is real.
- **Vocabulary coverage**: share of a person's words in the model vocabulary vs training people's share;
  low coverage predicts poor transfer (not yet built).

## Retraining on PANDORA

```bash
uv run python scripts/train_text_readers.py --only bigfive --bigfive-data pandora
```

Uses `slices/text_pandora.py` loaders (personality subreddits dropped), ridge regression on the same TF-IDF
(alpha picked by dev CV from 0.3 to 30), scores by r with CI, a length-aware linear recalibration, and the
minimum-text curve by r. Intervals then come from held-out residual quantiles (10th/90th) at the person's
length plus the bootstrap spread, and `validity_kind = "r"`. The reader switches automatically on the card's
`target`. Tested end to end on a synthetic corpus (`test_continuous_training_path`); not yet on real PANDORA.

## Proposed contract changes (not made)

- `TextItem.context: str | None` (subreddit, reply-to, hashtag set), so readers can drop r/mbti-type
  communities as the PANDORA loader does.
- `Estimate.prob: float | None` for Big Five readers trained on binary labels, so P(above median) is a field,
  not only text in `evidence`.

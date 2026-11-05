# Big Five from text across kinds of writing (leave-one-domain-out)

## Decision rule (written 2026-11-01 before any cross-domain result was computed)

- **Unit of test**: a whole dataset ("domain") is held out. Models are trained on the other domains only;
  nothing from the held-out domain (labels, vocabulary, idf, feature scaling, hyperparameters) is used.
- **Label**: every dataset's labels are turned into the same thing, "above this dataset's own median" (yes/no)
  per trait. Essays only has yes/no; for continuous datasets the within-dataset median split is used for
  training and for AUC, and the continuous score (z within dataset) is used for Pearson r as a second metric.
- **Primary metric**: AUC on the held-out domain, averaged over the 5 traits, then averaged over held-out
  domains with each domain weighted equally (big domains don't dominate).
- **Hyperparameters**: chosen inside the training domains only (inner leave-one-domain-out among the
  training domains, or 5-fold CV when only one training domain is left), from a fixed grid.
- **Training domains are weighted equally** (sample weight 1/n per domain) so a big domain can't drown a small one.
- **Choice**: the recipe with the best mean leave-one-domain-out AUC wins; within 0.005 the simpler/smaller
  one wins (order of simplicity: bag of words < domain-stable bag of words < style features < embeddings < combos).
- **Ship rule**: the artifact is retrained with the winner only if, on the domains other than Essays (where the
  current Essays-only model is itself out of domain), the winner's mean held-out AUC beats the current
  artifact's mean AUC on the same domains. Otherwise the artifact stays as it is and this page says so.
- **Null**: for every number, the AUC (or r) the same scores reach against shuffled held-out labels
  (1,000 shuffles): its mean and 95% range are reported next to the real number.

Two small things were written after the rule but before the main run, and before any number was seen on PAN 2015 or
the vlogs: Friends' NEU column was flipped to neuroticism (label-only evidence, `text_datasets.md`), and retweets
were dropped from PAN 2015. A two-domain smoke test (Essays and Friends only) was run while the other datasets
were still downloading; it changed no code or setting.

## The four domains

| domain | people | words each (median) | label | kind of writing |
|---|---|---|---|---|
| Essays (Pennebaker & King 1999) | 2,467 | 626 | **self-report**, yes/no | students writing freely for 20 minutes, 1997-2004 |
| FriendsPersona (Jiang et al. 2020) | 711 scenes | 30 | **observer** (annotators read the scene), yes/no | a TV character's lines in one scene |
| PAN 2015, English | 294 | 916 | **self-report** BFI-10, continuous | tweets |
| YouTube vlogs (Biel & Gatica-Perez 2013) | 404 | 466 | **observer** (MTurk raters watched the video), continuous | spoken monologue, transcribed |

Sources, licences and download URLs: `text_datasets.md`. The vlogs are CC BY-NC-ND, so they are test-only for
anything we ship. Self-report and observer labels are different targets: an observer score says how a stranger
sees someone from one scene or one minute of video, which agrees only moderately with what the person says about
themselves. A model can do well on one and badly on the other without either being "wrong".

## Recipes compared

- **a** the current saved artifact (Essays only, uni+bigram TF-IDF + logistic), unchanged
- **b** the same bag of words, trained on all the other domains pooled
- **c1** pooled bag of words keeping only *domain-stable* terms: words in at least 0.5% of documents in every
  training domain and used no more than 4x as often in one domain as another (vocabulary falls from 20,000 to ~200
  when Friends is in the pool, ~5,700 when it isn't)
- **c2** pooled *style* features: 292 topic-free numbers per person (rates of 22 function-word categories and
  each of ~250 function words, punctuation, word length, vocabulary variety)
- **d** pooled MiniLM sentence-embedding means (100-word chunks, up to 40 per person)
- **d+c2** embeddings and style side by side

All pooled recipes: training domains weighted equally, logistic regression per trait, C chosen by inner
leave-one-training-domain-out.

## Results: held-out domain AUC, mean over the five traits

| recipe | Essays held out | Friends held out | PAN 2015 held out | vlogs held out | **mean** | mean without Essays |
|---|---|---|---|---|---|---|
| a, current Essays model | (in-domain: 0.604) | 0.482 | 0.544 | **0.590** | | **0.539** |
| b pooled bag of words | **0.523** | **0.500** | 0.490 | 0.496 | 0.502 | 0.495 |
| c1 domain-stable words | 0.516 | 0.494 | 0.500 | 0.492 | 0.500 | 0.495 |
| c2 style features | 0.514 | 0.489 | 0.527 | 0.526 | **0.514** | 0.514 |
| d embeddings | 0.514 | 0.473 | 0.505 | 0.487 | 0.495 | 0.488 |
| d+c2 | 0.513 | 0.492 | 0.513 | 0.515 | 0.508 | 0.507 |

Shuffled-label null (95% range of a single trait's AUC): Essays 0.48-0.52, Friends 0.46-0.54, PAN 0.43-0.57,
vlogs 0.44-0.56. Essays' "in-domain" number for **a** is its own held-out test (mean of the card's 0.68/0.57/0.59/0.58/0.59),
shown for scale only; it is not a held-out-domain number.

**Decision by the rule:** c2 (style features) has the best mean leave-one-domain-out AUC among pooled recipes (0.514).
On the non-Essays domains it reaches 0.514, below the current Essays-only model's 0.539. **The artifact is not
changed.** Its card now carries the held-out-domain scores below (`held_out_domains`).

### Per trait: AUC [95% CI]

| held out | recipe | O | C | E | A | N |
|---|---|---|---|---|---|---|
| Friends | a | .495 [.42,.55] | .465 [.43,.51] | .506 [.46,.55] | .443 [.38,.48] | .503 [.47,.54] |
| | b | .540 [.48,.59] | .499 [.44,.56] | .492 [.44,.54] | .470 [.41,.53] | .501 [.46,.54] |
| | c1 | .521 [.46,.57] | .494 [.46,.55] | .510 [.46,.54] | .460 [.41,.50] | .485 [.46,.53] |
| | c2 | .461 [.42,.50] | .539 [.49,.59] | .513 [.49,.55] | .456 [.41,.50] | .479 [.43,.53] |
| | d | .520 [.48,.55] | .482 [.44,.52] | .447 [.41,.50] | .458 [.43,.49] | .456 [.41,.52] |
| | d+c2 | .507 [.46,.54] | .541 [.48,.59] | .493 [.47,.54] | .442 [.39,.50] | .478 [.44,.53] |
| PAN 2015 | a | .522 [.46,.59] | .544 [.47,.61] | .543 [.47,.62] | .542 [.47,.61] | **.571** [.51,.64] |
| | b | .431 [.36,.49] | .491 [.42,.57] | .470 [.40,.54] | .517 [.45,.58] | .539 [.47,.61] |
| | c1 | .496 [.43,.56] | .431 [.37,.49] | .501 [.42,.58] | .518 [.46,.59] | .553 [.49,.62] |
| | c2 | .427 [.36,.49] | .541 [.47,.61] | .565 [.49,.63] | .543 [.48,.61] | .558 [.50,.62] |
| | d | .437 [.37,.50] | .477 [.41,.54] | .522 [.45,.59] | .592 [.53,.66] | .497 [.43,.56] |
| | d+c2 | .414 [.35,.48] | .519 [.45,.59] | .523 [.45,.59] | .569 [.50,.63] | .541 [.48,.61] |
| vlogs | a | .472 [.41,.53] | **.611** [.55,.66] | **.560** [.51,.62] | **.701** [.65,.75] | **.608** [.55,.67] |
| | b | .494 [.44,.55] | .462 [.41,.52] | .429 [.38,.49] | .559 [.50,.61] | .537 [.48,.59] |
| | c1 | .485 [.43,.54] | .426 [.37,.48] | .468 [.41,.52] | .579 [.52,.64] | .500 [.44,.56] |
| | c2 | .488 [.43,.55] | .605 [.55,.66] | .483 [.43,.54] | .497 [.44,.55] | .559 [.50,.62] |
| | d | .521 [.47,.58] | .519 [.46,.58] | .418 [.36,.47] | .523 [.46,.58] | .454 [.40,.51] |
| | d+c2 | .524 [.47,.58] | .582 [.52,.64] | .443 [.38,.49] | .520 [.47,.58] | .508 [.45,.56] |
| Essays | b | .467 [.44,.49] | .528 [.50,.55] | .527 [.50,.55] | .540 [.52,.56] | .552 [.53,.57] |
| | c1 | .513 [.49,.54] | .514 [.49,.54] | .499 [.48,.52] | .507 [.49,.53] | .547 [.52,.57] |
| | c2 | .473 [.45,.50] | .521 [.50,.54] | .522 [.50,.54] | .540 [.52,.56] | .514 [.49,.54] |
| | d | .533 [.51,.56] | .527 [.51,.55] | .492 [.47,.51] | .514 [.49,.54] | .504 [.48,.53] |
| | d+c2 | .501 [.48,.52] | .534 [.51,.56] | .502 [.48,.53] | .522 [.50,.54] | .505 [.48,.53] |

Friends CIs resample whole characters (80% of scenes belong to the six main characters; characters are grouped by first name), so they are honest about how few
independent people there are.

### Pearson r with the continuous scores (PAN 2015 and vlogs only)

| held out | recipe | O | C | E | A | N |
|---|---|---|---|---|---|---|
| PAN 2015 | a | +.02 | +.12 [.01,.22] | +.07 | +.04 | **+.22** [.12,.33] |
| | c2 | -.14 | +.13 | +.08 | +.15 | +.27 [.15,.37] |
| | d+c2 | -.13 | +.18 | +.08 | +.17 | +.21 |
| vlogs | a | +.02 | **+.22** [.13,.31] | +.09 [.01,.18] | **+.40** [.32,.48] | **+.27** [.18,.36] |
| | c2 | +.02 | +.20 | -.03 | +.14 | +.10 |
| | d+c2 | .00 | +.17 | -.15 | +.08 | +.05 |

Shuffled-label null for r: about -0.11 to +0.11 (PAN), -0.10 to +0.10 (vlogs). Every number, including b, c1 and d,
is in `text_crossdomain_results.csv` (with null mean, null range and permutation p).

## What the current model does on writing it never saw

- **Vlogs (observer-rated, spoken): it works.** Agreeableness AUC 0.70 (r = 0.40), neuroticism 0.61,
  conscientiousness 0.61, extraversion 0.56, all clear of the shuffled null. Openness, its best trait on Essays
  (0.68), drops to chance (0.47). What carries over looks like *tone* (warm vs hostile, calm vs anxious), which
  both a writer's self-report and a viewer's impression pick up; what Essays learned about openness was mostly
  student topics, which a vlog doesn't share.
- **Twitter (PAN 2015, self-report): weak.** Mean 0.54, only neuroticism (0.57, r = 0.22) clears the null alone.
  With 294 people the CIs are about +-0.07, so this can't separate "slightly works" from "doesn't".
- **Friends (observer, 30 words of a fictional character): nothing.** Mean 0.48, no trait above the null,
  agreeableness slightly *below* (0.44). Thirty words of scripted sitcom dialogue, judged by annotators from the
  whole scene, is not what the model reads.

## Why pooling lost (post-hoc diagnostics, not used for the decision)

`scripts/text_crossdomain.py --explore` (results in `text_crossdomain_results_explore.csv`), written after seeing
the table above: train on **one** domain, test on another (bag of words, mean AUC over traits).

| trained on \ tested on | Essays | Friends | PAN 2015 | vlogs |
|---|---|---|---|---|
| Essays | | .484 | **.545** | **.593** |
| Friends | .500 | | .503 | .481 |
| PAN 2015 | .512 | .497 | | .494 |
| vlogs | **.537** | .489 | .493 | |

Only Essays (2,467 people) teaches anything that travels, and vlogs teach a little back to Essays. Friends (30
words a scene, observer-rated fiction) and PAN (294 people) teach nothing transferable. Weighting domains equally,
as the rule said, gives those two two-thirds of the say when Essays is in the pool, which drowns its signal. With
natural weights (each person counts once, so Essays dominates) the pooled bag of words gets 0.554 on vlogs and 0.517
on PAN, closer to Essays-only (0.593 / 0.545) but still below it. Style and embedding single-domain transfers show
the same pattern. So with the data that is open today, adding domains adds noise, not robustness.

A warning about pooled *random-split* scores: `train_text_readers.py --bigfive-data pooled` trains the pooled bag of
words and tests it on random held-out people from the same mix, where it scores a mean AUC of 0.62, *higher* than
Essays alone. That number is flattering because the test people come from domains the model trained on, and PAN's
median splits are lopsided (only 33% "high" C, 29% "high" E, because of ties), so knowing "this is a tweet" already
predicts the label. Leave-one-domain-out removes that shortcut.

## Plain English: domain shift and why this is the honest test

A model that reads personality from text learns whatever separates high from low scorers *in its training data*.
Some of that is real (anxious people use more negative-emotion and self-focused words wherever they write) and
some is local to that data: Texas students in 1999 who wrote about "roommate, class, boyfriend" differ from others
partly because of who they were and where. When the kind of writing changes (tweets, a vlog, a sitcom line, a 2026
Reddit comment), the local part stops working and may point the wrong way. That change is **domain shift**.

The usual test, holding out 20% of the *same* dataset, can't see domain shift: the held-out people wrote the same
kind of text in the same place at the same time, so local patterns still pay off, and so does knowing which dataset
a text came from. **Leave-one-domain-out** holds out a *whole kind of writing*: the model is built without ever
seeing a tweet (or a vlog, or a sitcom), then scored on tweets. That is the situation every real user is in,
because nobody's own messages look like the training set. It gives lower numbers, and they are the ones to believe.

What we found: the current Essays model's signal for agreeableness, neuroticism and conscientiousness partly
survives a change to spoken vlogs (AUC 0.61-0.70), barely survives a change to tweets (0.54), and doesn't survive
30-word sitcom lines; its openness signal doesn't travel at all. None of the robustness recipes (shared vocabulary,
style-only features, sentence embeddings) or pooling the three extra open datasets beat it. The way forward is more
*self-report* data in the kinds of writing users bring: PANDORA (Reddit, ~1,600 people, continuous scores) is the
obvious one and needs only the access request; ChaLearn First Impressions needs a sign-up.

## Reproduce

```bash
cd py
uv run python scripts/text_crossdomain.py            # ~12 min first time (MiniLM embeddings cached after), writes the CSV + card section
uv run python scripts/text_crossdomain.py --explore  # post-hoc transfer matrix
uv run python scripts/train_text_readers.py --only bigfive --bigfive-data pooled --out /tmp/pooled   # the losing pooled model
```

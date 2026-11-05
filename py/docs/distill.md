# A distilled on-device text reader for the Big Five

Claude Haiku 4.5, reading a person's text with no training, beat our bag-of-words reader on kinds of writing the
bag of words never saw (`llm_rater.md`: PAN 2015 tweets and YouTube vlogs, mean AUC 0.618 against 0.558, a lead of
+0.060 [0.025, 0.093]). Haiku runs on Anthropic's servers, so using it would mean a participant's words leave their
computer, which the study promises never happens. A 1.7B model run locally copied the prompt's example score and
told us nothing. This page tests a third way: ask Haiku to rate a large pool of other people's writing once, train a
small model on this laptop to give the same ratings, and ship only the small model. The small model (the
"student") then reads a participant's text on their own machine.

Code: `scripts/distill.py`. Reader: `src/aas/readers/distilled.py`. Artifact: `models/text_distilled/`.

## Pre-registration (2026-11-05, committed before any rating of the pool was obtained)

### The pool

- 2,000 users of the PersonalityCafe forum (the MBTI dataset already on disk, `data/raw/mbti`), drawn in a seeded
  random order (seed 20261105). Anyone with fewer than 100 words after leak stripping, or whose prompt the label
  guard refuses, is skipped and the next person is taken. The dry run skipped 2 people for length and none for the
  guard.
- Their MBTI types are dropped when the data is loaded and never read.
- None of the three test sets (Essays, PAN 2015, YouTube vlogs) is in the pool, and none of their people, text or
  labels is used to fit or tune anything. FriendsPersona (30 words of sitcom dialogue per scene) is left out: Haiku
  can say little from 30 words, so it would mostly teach the student to answer 3.0.
- Each person's text is prepared exactly as for the Haiku test run: `readers.text.clean` on every post, then cut to
  about 1,500 tokens (a seeded random sample of whole posts, kept in order). Median 1,034 words sent per person.

### The teacher

- `claude-haiku-4-5-20251001`, prompt `b5-ipip50-v1`, the same request settings (no thinking, max_tokens 600), the
  same leak stripping, label guard and parser as `llm_rater.py`. Nothing in the prompt changes.
- Sent through the Message Batches API (half price). Each answer is written into the existing disk cache under the
  key a normal call would use, so nothing is paid twice and a re-run is free. A reply that fails to parse gets one
  synchronous retry, as the rater already does. A refusal or a second bad reply makes that person missing, and they
  are left out of training (counted below).
- Budget: the owner approved $10. Dry-run estimate for 2,000 people: $2.96 expected and $4.71 if every reply hit
  max_tokens (Batch API), against $5.92 / $9.42 without it. The script refuses to submit if the expected cost is
  over $8 or if spend so far plus the upper bound is over $10. Spend is logged to `data/distill_spend.jsonl`.

### The students (three, fixed now)

Each student is five ridge regressions, one per trait, from features of the cut text to Haiku's 1-5 rating. The
ridge penalty is chosen by generalised cross-validation inside the pool only (grid 10^-2 to 10^4, 25 steps, as in
`evaluate.py`).

1. `bow`: unigram and bigram counts, top 20,000 terms seen in at least 5 pool people, sublinear TF-IDF, L2 per
   person (the same features as the current reader, refitted on the pool).
2. `emb`: all-MiniLM-L6-v2 sentence embeddings (the same ONNX model used before), averaged over 100-word chunks
   (at most 40), standardised on the pool.
3. `mix`: the average of `bow` and `emb` predictions after z-scoring each (on the pool for selection, within a
   dataset for testing, from the scores alone).

Which student goes forward is decided on the pool alone, before any test-set score is computed: the one with the
highest 5-fold cross-validated agreement with Haiku (Pearson r between the student's out-of-fold prediction and
Haiku's rating, averaged over the five traits). If two are within 0.01, the simpler wins, in the order bow, emb, mix.
The pass rule applies to that one student only. The other two are reported.

### The test and the pass rule

- People: the same 150 per test set that Haiku rated (seed 20261101; Essays from the saved model's held-out test
  split), so every comparison with Haiku is paired. The student reads the same cut text Haiku read.
- Arms compared on those people: the student, Haiku (its cached ratings from the earlier run, no new calls), and
  the current bag-of-words reader on the same cut text (on Essays, the model refitted on the Essays dev split,
  exactly as in the Haiku comparison, because the saved one was trained on those people).
- Scoring: as in the Haiku comparison (`scripts/llm_crossdomain.py`, which uses `score_block` from
  `text_crossdomain.py`): AUC against each dataset's own median split, averaged over the five traits; Pearson r
  against the continuous scores on PAN 2015 and vlogs. Paired bootstrap, 2,000 resamples of people within each
  dataset, for every mean and every difference.
- Primary metric: the mean AUC over the three test sets, each weighted equally. All three are kinds of writing the
  student never saw.
- The brief gave the pass mark as "keeps at least half of Haiku's +0.060 lead over bag of words, i.e. a mean AUC
  of at least 0.588". Those numbers come from PAN 2015 and vlogs only, while the primary metric covers all three
  sets. Over all three, on the same people, Haiku scores 0.6185 and bag of words 0.5737 (bag of words is in its own
  domain on Essays), so half the lead puts the mark at 0.5961. To honour both readings, the student passes only if
  both hold:
  1. mean AUC over PAN 2015 and vlogs at least 0.588, and
  2. mean AUC over Essays, PAN 2015 and vlogs at least bow + 0.5 × (Haiku − bow) on the same people (0.5961 from the
     cached results; the script recomputes it).
- Reported alongside, not part of the rule: the 95% CI of the student's mean AUC, of student minus bag of words and
  of student minus Haiku, and the share of Haiku's lead the student keeps (with CI); Pearson r; per-trait AUCs with
  CI and a shuffled-label null; and the student and bag of words on every test person (494 Essays, 294 PAN, 404
  vlogs), which costs nothing.
- If the student fails, it is still saved with its card, the card and this page say it failed, and it is not
  wired as the default reader.

### Contamination and other caveats, stated in advance

- Haiku may have seen Essays, PAN 2015 and the vlogs during training (`llm_rater.md`). The student cannot inherit
  a memorised test label from that, because Haiku never sees a test text in this experiment and the student never
  trains on one. Haiku's own test numbers carry the caveat as before, and so does the threshold derived from them.
- PersonalityCafe posts are public too, and their MBTI types might be memorised. If Haiku leaned on a remembered
  type, its pool ratings would carry type stereotypes. That would make the teacher worse, not the test unfair.
- The pool is one kind of writing (a typology forum). The student has to carry what it learned there to essays,
  tweets and speech, so a fail could mean "the forum is the wrong pool" as much as "distillation doesn't work".
- Training on Claude outputs: Anthropic's commercial terms restrict using outputs to build competing AI models. A
  five-number personality regressor is unlikely to count, but it should be checked before any commercial use.

## Results (2026-11-05)

### Verdict: fail, narrowly

The student that went forward (`mix`) closes about half of the gap to Haiku on tweets and vlogs, which is roughly
what the rule asked for, but it lands just under the mark there (0.586 against 0.588) and well under it once Essays
is counted (0.583 against 0.596). Both conditions had to hold, and neither did. It is saved under
`models/text_distilled/` with a card that says it failed, and nothing uses it by default. The current bag-of-words
reader stays.

### What it cost

All 2,000 pool people were rated in one batch, which finished in under four minutes. There were no refusals, no
invalid replies and no errors, so nobody was dropped and no retry was paid for. Haiku read 3,519,614 input tokens
and wrote 637,058 output tokens, at Batch API prices $3.35 in total, a little above the $2.96 estimate because the
replies were longer than the assumed 250 tokens (319 on average). The log is `data/distill_spend.jsonl`. Free disk
went from 22 GiB before to 27 GiB after (other processes on the machine move it; this run added about 2 MB of cache
and 0.6 MB of model).

### Choosing the student on the pool

On the pool, before any test score was computed, each student was scored by how closely its out-of-fold
predictions matched Haiku's ratings (Pearson r, five-fold cross-validation). The mixture of the two matched best,
by more than the 0.01 tie margin, so it went forward.

| student | O | C | E | A | N | mean |
|---|---|---|---|---|---|---|
| bow | .53 | .47 | .54 | .70 | .57 | .563 |
| emb | .62 | .49 | .58 | .69 | .59 | .595 |
| mix (chosen) | .63 | .52 | .61 | .74 | .62 | .624 |

So the student reproduces Haiku only in part. It explains roughly
40% of the variance in Haiku's ratings of new forum users. On the test sets its agreement with Haiku was similar
or higher on tweets (r .59 to .80 by trait) and vlogs (.55 to .60), and lowest on Essays (.42 to .70), the kind of
writing least like a forum.

### The test, on the same 150 people per set that Haiku rated

Each number is the AUC against that dataset's own median split, averaged over the five traits, with a 95% paired
bootstrap CI. Bag of words reads the same cut text; on Essays it is in its own domain.

| arm | Essays | PAN 2015 | vlogs | PAN + vlogs [CI] | all three [CI] |
|---|---|---|---|---|---|
| Haiku 4.5 (API) | .620 | .557 | .679 | .618 [.588, .648] | .618 [.594, .643] |
| student mix (chosen) | .576 | .512 | .661 | .586 [.556, .616] | .583 [.558, .608] |
| student emb | .591 | .495 | .656 | .576 [.545, .606] | .581 [.556, .605] |
| student bow | .546 | .520 | .633 | .577 [.546, .607] | .566 [.541, .593] |
| bag-of-words reader | .605 | .508 | .607 | .558 [.529, .587] | .574 [.550, .598] |
| pass mark | | | | .588 | .596 |

Differences for the chosen student, paired:

| contrast | PAN + vlogs | all three |
|---|---|---|
| student minus bag of words | +.028 [−.002, +.059] | +.009 [−.017, +.036] |
| student minus Haiku | −.031 [−.061, −.003] | −.036 [−.060, −.012] |
| share of Haiku's lead over bag of words kept | 47% [−4%, 94%] | 20% [−68%, 67%] |

Read plainly: on tweets and vlogs the student sits halfway between bag of words and Haiku, and with 150 people per
set we can't tell it apart from either end with confidence (the CI on the share kept runs from nothing to nearly
all). It is clearly worse than Haiku. On Essays it loses to the bag-of-words reader, which was trained on essays,
and that pulls the three-set mean down.

Pearson r with the continuous scores, mean over traits: PAN 2015 Haiku .136, student .087, bag of words .057;
vlogs Haiku .416, student .366, bag of words .220.

### Per trait (same 150 people)

| set | arm | O | C | E | A | N |
|---|---|---|---|---|---|---|
| Essays | Haiku | .598 | .609 | .622 | .672 | .600 |
| | student | .554 | .583 | .520 | .650 | .572 |
| | bag of words | .645 | .577 | .552 | .717 | .535 |
| PAN 2015 | Haiku | .530 | .589 | .619 | .458 | .587 |
| | student | .426 | .466 | .573 | .480 | .613 |
| | bag of words | .480 | .510 | .470 | .495 | .586 |
| vlogs | Haiku | .638 | .730 | .642 | .716 | .667 |
| | student | .576 | .684 | .627 | .778 | .640 |
| | bag of words | .485 | .632 | .508 | .737 | .675 |

The student loses most where Haiku itself is thin. On tweets its openness and conscientiousness fall below chance
by AUC. Conscientiousness on tweets is an odd case: by r it is one of the student's better traits (+.25), and PAN's
median split is lopsided (ties put only about a third of people "high"), so AUC and r disagree there. On vlogs the
student keeps most of what Haiku has and beats it on agreeableness.

### Every test person (student and bag of words only, no extra spend)

With all 494 Essays, 294 PAN and 404 vlog people the picture is a little kinder to the student. These numbers were
not part of the rule, and Haiku has no ratings for most of these people, so there is no Haiku column.

| arm | Essays | PAN 2015 | vlogs | PAN + vlogs [CI] | all three [CI] |
|---|---|---|---|---|---|
| student mix | .590 | .540 | .660 | .600 [.580, .622] | .597 [.582, .612] |
| student emb | .583 | .529 | .656 | .593 | .589 |
| student bow | .581 | .546 | .632 | .589 | .587 |
| bag-of-words reader | .603 | .548 | .592 | .570 [.549, .589] | .581 [.566, .597] |

On the larger sample the student beats the bag-of-words reader on tweets and vlogs by +.030 [+.010, +.051], a CI
clear of zero, and over all three by +.016 [.000, +.032]. That mostly comes from vlogs (agreeableness .77,
conscientiousness .70, extraversion .65), where the observer-rated labels match the kind of judgment Haiku makes.
On tweets the two are level. These are exploratory numbers, and the 150-person comparison above is the one the rule
was written for.

### What this means

Distilling from 2,000 forum users moved most of Haiku's advantage on vlogs onto the laptop, and almost none of it
on tweets or essays. Three explanations fit, and this run can't separate them:

1. The pool is one kind of writing. A student trained only on forum posts has to carry Haiku's judgments to
   tweets and essays, and its agreement with Haiku is lowest on Essays.
2. Haiku's ratings are themselves weak on tweets (.557), so even a perfect copy would gain little there.
3. A linear model on TF-IDF and MiniLM features can match about 40% of Haiku's rating variance, so the copy loses
   part of whatever Haiku knows.

The cheapest next test is a pool with more kinds of writing (for example PANDORA's Reddit comments once access
lands, or public blog and review text), rated the same way for a few dollars, with the same frozen rule.

### Contamination, after the fact

The student never saw a test text or a test label, and Haiku never saw a test text in this experiment, so the
student's test numbers carry no memorisation risk of their own. Haiku's test numbers, and therefore the pass mark,
still carry the caveat in `llm_rater.md`. Haiku's mean pool ratings (O 3.93, C 2.80, E 2.95, A 3.32, N 3.54) look
like an introverted, open, somewhat anxious crowd, which is what the forum is known to be. That fits either real
reading or remembered MBTI stereotypes; nothing here tells them apart.

### The artifact

`models/text_distilled/` (0.6 MB plus the 90 MB MiniLM model already used elsewhere): `model.joblib` and
`card.json`. The card carries the teacher and prompt, the pool, the features, the pool fidelity, AUC with CI per
trait on every test person, the verdict, and `validity_by_source`. Essay-like prose takes the Essays numbers,
tweets the PAN numbers, and other prose (Reddit and so on, unmeasured) the lower of PAN and vlogs per trait. The
reader `TextDistilledReader` (`src/aas/readers/distilled.py`) follows the Reader protocol: it cuts the person's
text the way the teacher's input was cut, reads longer histories as up to five such samples and averages them,
turns the score into a z against the pool, shrinks it by the r implied by the stated AUC and gives an 80% interval
of ±1.28·sqrt(1 − r²). A trait whose stated AUC is at or below 0.5 for the person's writing (openness and
conscientiousness on tweets, here) comes back as not usable. Every estimate's evidence says the reader failed its
test and is not the default. Tests: `tests/test_reader_distilled.py`.

### Reproduce

```bash
cd py
uv run python scripts/distill.py --dry-run          # pool and cost, no network
uv run python scripts/distill.py --rate             # free now: every rating is cached
uv run python scripts/distill.py --evaluate         # ~15 min (MiniLM embeddings, cached after)
```

Outputs: `docs/distill_results.json` (selection, contrasts, verdict, spend), `docs/distill_results.csv` (every
arm, set and trait, with CI, shuffled-label null and r).

# Pre-registration draft (OSF style)

Version 0.1, 1 Nov 2026. To be finalised after the pilot (feature list frozen, extractor version hashed) and
posted on OSF **before** the first main-study participant is invited. Fields in [brackets] are filled at
posting. Everything else is decided now, before any data.

## 1. Study information

**Title.** Does combining a person's own digital sources predict their personality better than one source? A
two-session data-donation study with local feature extraction.

**Authors.** [owner]. Unaffiliated.

**Description.** Adults in the US and UK take the IPIP-50 Big Five questionnaire, request their own Reddit/X,
Google (Search, YouTube) and Spotify data exports, and run an in-browser extractor that sends only derived
numbers (PRIVACY.md). We test pre-trained AAS readers on these people, learn models for the sources that have
no reader, and test whether a combined model beats the best single source and whether its stated intervals
are honest.

**Existing data.** None of the study data exist at registration. The readers were trained and their held-out
numbers published in `docs/readers_text.md` and `docs/reader_music.md` before this registration; pilot data
(10 people) were used only to fix the instrument and are excluded from every analysis below.

## 2. Hypotheses

Traits: O, C, E, A, N, scored from IPIP-50 on the 597k-person norms (`quiz.score`, z). "Text reader" =
`text-bigfive` as saved at [model card hash], applied unchanged.

- **H1 (text validity).** The text reader's z correlates positively with quiz z. Directional, per trait.
  Expected: O strongest (r about 0.2 to 0.3); the others 0.05 to 0.2.
- **H2 (increment, primary).** A combined model (text reader + learned models of the other sources the person
  has) predicts quiz z better than the text reader alone on held-out people, measured as lower squared error.
  Directional. Primary test pooled over the five traits; secondary per trait.
- **H3 (coverage, primary).** (a) The text reader's own 80% intervals contain quiz z for between 75% and 85%
  of person-traits (equivalence). (b) The same for the combined model's 80% intervals.
In H2 to H4, "combined model" means the model learned inside cross-validation in section 6 (ridge per
non-text source, stacked with the text reader z). The zero-data fusion that the product ships
(`aas.fusion`, `bayes`; docs/fusion.md) is a different model and is analysed separately in S7.

- **H4 (skill).** The combined model's mean absolute error is lower than guessing z = 0 for everyone. Directional,
  pooled over traits.

Secondary (directional where stated, Holm within each family, reported whatever the result):

- S1. Each non-text source alone (Google features; Spotify features), learned in CV, correlates positively with
  quiz z, per trait.
- S2. MBTI reader letters vs OEJTS letters: AUC > 0.5 per letter (I/E, N/S, T/F, J/P).
- S3. Text validity rises with words of text (interaction of reader z with log words in predicting quiz z),
  positive.
- S4. Refitting the text model's calibration (all participants) and its weights (common-word-block subset
  only) on study people improves on the shipped reader in CV. Exploratory in direction for the refit weights.
- S5. Quiz retest reliability at 2 weeks: reported per trait with CI. No hypothesis.
- S6. Reddit-only vs X-only writers: difference in text validity. Two-sided, exploratory.
- S7. Fusion as shipped (added 1 Nov 2026, see the change log). (a) `bayes`, the zero-data fusion of
  text-bigfive and text-mbti that `FusedReader` shows people, applied unchanged: its z correlates positively
  with quiz z per trait (one-sided, Holm over 5); its 80% intervals' pooled coverage (same TOST rule as H3);
  skill; paired per-person error against the text reader as shipped, pooled over traits (one-sided).
  (b) `bayes_eb_cp` fitted inside the same CV as H2: the same outcomes, with the paired error test against
  the CV-calibrated text model; registered at n >= 250, reported as exploratory below that.
  (c) The fusion checks of docs/fusion.md, as estimates with CIs and no test: each channel's real validity
  against its stated validity (and the x0.75 discount), and the error correlation between text-bigfive and
  text-mbti per linked trait (prior 0.30).

## 3. Design plan

Observational, cross-sectional with a two-week retest on a random subset. No manipulation, no blinding
needed (the extractor computes reader outputs on the participant's machine before anyone sees quiz scores;
the analysis script is written and tested on simulated data before the data file is opened).

## 4. Sampling plan

**Recruitment.** Prolific, US and UK, 18+, fluent English; screener, Session 1, Session 2, optional Session 3,
retest subset (PROTOCOL.md section 6).

**Target.** [250 or 600, decided by the owner after the pilot] usable participants (POWER.md). If the owner
chooses the **two-stage** option: stage 1 = 250, stage 2 = 350 more (total 600); the decision to run stage 2 is
made on **budget and return rate only**, written down before stage-1 outcomes are computed. Stage-1 outcomes
are not analysed until stage 2 has finished or been cancelled. There is no interim test.

**Stopping rule.** Recruitment stops at the first of: (a) the target number of usable participants, counted from
payload and quiz quality checks only (never from outcomes); (b) the pre-set budget ceiling [amount]; (c) the
calendar deadline [date, 12 weeks after launch]. Whatever n results is analysed with the plan below. Session 1
batches of about 100 per week; recruitment may pause for technical faults, logged with dates.

**Allowed monitoring during collection.** Counts, completion rates, extractor errors, payload validity, quiz
completion time. Not allowed: any correlation, model score or coverage involving quiz answers.

## 5. Variables

**Outcome.** Quiz z per trait, IPIP-50 full form (`form == "full"`). Participants with any IPIP item missing are
scored with the `partial` form only if at least 8 of 10 items for that trait are answered; else that trait is
missing for them.

**Predictors, fixed now (feature lists frozen at the pilot's end and listed in the OSF upload):**

- Text: `text-bigfive` z and P for each trait (as computed on the participant's machine), log words.
- Google: a fixed set of **at most 10 features per trait model**: 24-hour rhythm summarised by the first
  trigonometric moment of the hour histogram, mean cosine and mean sine (2; together they carry the circular
  mean and the spread, and unlike a mean hour they can enter a linear model; the histogram is the mean of
  the search and watch histograms the person has), share of days active, mean query length, question share, channel entropy, top-10-channel share,
  repeat-watch share, and the two largest topic-share principal components fitted inside the training folds.
- Spotify: the `music` reader's 7 measured traits plus the 2 dials (9), basic-export people flagged.
- A missing source enters as an indicator plus training-fold median imputation; people without the second
  source contribute only through the text reader.

Keeping each learned model to about 10 features is a deliberate choice from POWER.md section 2: more features at
this n shrink further under cross-validation.

## 6. Analysis plan

All code is in the repo before data are opened (`scripts/study_analysis.py`, [commit hash]); it is run once on
simulated data shaped like the study (output posted with the registration) and once on the real data.

**H1.** Pearson r between text reader z and quiz z per trait, no fitting. One-sided p from Fisher z; 95% CI by
person bootstrap (2,000 resamples). Family: 5 traits, Holm, alpha 0.05. The reader is not refit, so no
permutation null is needed; `null_mean` is 0 by construction.

**H2.** For each trait, 10-fold CV repeated 10 times with fixed seeds (0 to 9), folds stratified by which sources a
person has:
- Single: quiz z regressed on the text reader z inside the training folds (linear calibration), predicted for
  the test fold.
- Combined: ridge models for each non-text source (alpha by inner 5-fold CV on a fixed grid, `evaluate.ALPHAS`),
  their inner out-of-fold predictions stacked with the text reader z by OLS on the training folds, then
  predicted for the test fold.
- Per person i and trait t: d_it = (q_it - single_it)^2 - (q_it - combined_it)^2, averaged over the 10 repeats.
- **Primary test:** D_i = sum over traits of d_it; one-sided t-test that mean D > 0, alpha 0.05, confirmed by a
  sign-flip permutation test (10,000 flips; if they disagree, the permutation result is the one reported).
- Per trait: the same test on d_it, Holm over 5 traits.
- Effect size reported: Delta r = r(combined, q) - r(single, q) per trait with person-bootstrap CI, and the
  pooled mean.
- The "best single source" is fixed in advance as text. As a sensitivity check (not a test), we also report the
  best single source chosen inside the CV loop.
- Sub-analyses by source pair (text + Google; text + Spotify) use only people with both; reported with CIs, Holm
  within each pair's 5 traits.

**H3.** (a) The text reader's stated `[low, high]` with `interval_level` 0.80, exactly as shipped. Coverage =
share of person-traits with quiz z inside. (b) Combined model: interval = test prediction + the 10th and 90th
percentiles of training-fold out-of-fold residuals for that trait (split-conformal style, computed inside CV).
Pooled coverage over traits with a 90% person-cluster bootstrap CI; **equivalence (TOST) to 0.80 +- 0.05 is
supported if the 90% CI lies inside [0.75, 0.85]**. Per trait coverage, mean width, and coverage by word-count
tercile are reported as estimates. If (a) fails, the shipped reader's intervals are recalibrated in S4 and the
failure is reported prominently, not hidden.

**H4.** Per person, M_i = sum over traits of (|q_it| - |q_it - combined_it|). One-sided t-test mean M > 0, alpha
0.05, permutation-confirmed. Skill = 1 - MAE / MAE(z = 0) per trait reported with CI. Also reported for the text
reader alone.

**Secondary.** S1 as H2's learned components alone, with `evaluate.predict`-style permutation p (at least 1,000
permutations, so Holm over 5 can be met; see `quiz.md` note) and Holm per source over 5 traits; read against
`null_mean`. S2 AUC with DeLong CI, Holm over 4 letters. S3 regression of quiz z on reader z, log words and their
product; one-sided test of the product term, pooled over traits with trait fixed effects and person-clustered
SEs. S4 nested CV comparison like H2. S5 Pearson and ICC(2,1) with CIs. S6 difference in r with bootstrap CI.
S7 as described in section 2, computed by the same script with the same folds and seeds as H2; family of 5
traits per method, Holm.

**Inference criteria.** Alpha 0.05, one-sided for directional hypotheses, two-sided otherwise. Holm within the
families named. A result is described as "supported" only if it passes; effect sizes and CIs are reported for
everything regardless. Non-significant results are reported as "not detected at this n", with the smallest
effect the study could have detected (from POWER.md) next to it.

## 7. Data exclusion

Applied in this order and reported as a flow table:

1. Pilot participants.
2. Failed 2 or more of the instructed-response items presented (3, or 2 when the optional OEJTS block was
   skipped) (also rejected on Prolific).
3. Quiz quality, same rules as the norms cleaning (`quiz.clean`): median item time under 1 s; all 50 answers
   identical; 15 or more identical answers in a row in page order.
4. Answered "no" to "should we use your answers?".
5. No Session 2 payload (dropout; its own row in the flow table). Then: writing below 100 words after cleaning,
   or no second source above its floor (PROTOCOL.md section 4), as reported by the extractor's floor check
   (the payload's counts are rounded or bucketed, so the floors cannot be re-checked from it). A source below
   its floor is treated as absent for that person.
6. Payload integrity: unknown extractor version or hash; impossible values; duplicate payload (identical feature
   vectors under two study IDs: keep the one submitted first, ties by study ID; a copy of a payload whose
   owner was excluded under an earlier rule is still a duplicate).
7. Withdrawn participants, at any time before analysis.

Outliers are not removed; the readers' z are clipped by design and the quiz z is bounded.

## 8. Missing data

A trait missing from the quiz (under 8 of 10 items) drops that person-trait only. Sources are missing by design;
handled by indicators and fold-internal imputation as above, never by imputing the outcome.

## 9. What would count against us

- H1 null on all five traits: the text reader does not transfer to ordinary people. The product cannot claim
  personality from writing until retrained (S4).
- H2 null with a narrow CI around 0: combining sources does not help at this scale; the "whole person" claim is
  not supported.
- H3(a) coverage outside 0.75 to 0.85: the shipped intervals are miscalibrated; fix before any UI shows them.

## 10. Other

The study page and extractor are released as source code with the registration. Group-level results are
published whatever they show. Individual-level data are not shared (PRIVACY.md section 5); a summary dataset with
quiz z, reader outputs and coarse aggregates may be shared on request under a data-use agreement.

## Changes before data collection

All made before any study data existed, while writing and dry-running `scripts/study_analysis.py` on
simulated studies (`scripts/simulate_study.py`). Simulated data only; no outcome from real people was seen.

**Made, 1 Nov 2026** (clarifications and fixes of internal inconsistencies):

1. Section 7 rule 2: "failed 2 of 3 attention checks" became "2 or more of the instructed-response items
   presented". PROTOCOL.md says "2 or more", Prolific's rule is "more than one", and the third check sits in
   the optional OEJTS block, so some people see only two.
2. Section 7 rule 5: the floors are decided by the extractor's floor flags (the payload's word count is
   rounded to the nearest 100 and the search and play counts are bucketed, so 100 words, 300 searches or
   2,000 plays cannot be re-checked from the payload); a source below its floor is treated as absent; people
   with no Session 2 payload get their own flow-table row.
3. Section 7 rule 6: "keep the first" duplicate means the earliest submission; a copy of an excluded
   person's payload is still a duplicate.
4. Section 5, Google: "circular mean and spread" of the 24-hour rhythm replaced by the first trigonometric
   moment (mean cosine, mean sine). A mean hour cannot enter a linear ridge model (23:00 and 01:00 are close
   in time and far apart as numbers); the two moments carry the same information. Still 2 features.
5. Section 2: "combined model" in H2 to H4 defined as the CV-learned stack; secondary family **S7** added for
   the fusion that docs/fusion.md (same date) made the shipped default (`bayes`) and the main-study model
   (`bayes_eb_cp`). The pre-registration had no analysis of either, although README.md says checking them is
   the study's job.

**Accepted 2026-11-01 (the owner delegated method decisions); all items below are part of the registration** (the script implements them;
the reasons are in the PREREG_ISSUES block of `scripts/study_analysis.py` and in its report):

- P6. H2's status. README.md (fusion update) calls "fusion beats text alone" a secondary, exploratory outcome
  at n = 250, while H2 is primary here. Default: keep H2 primary as registered. If the owner agrees with the
  README, replace "H2 (increment, primary)" with "H2 (increment; primary only if the final n is 600 or more,
  otherwise secondary and exploratory, decided by the stopping rule before outcomes are computed)".
- P7. Add to H2 and H4: "D_i and M_i are summed over the traits for which person i has a quiz z."
- P8. Add to H2: "A non-text source enters the combined model in a training fold only if at least 20 training
  people have it; otherwise that fold's combined prediction is the text calibration."
- P9. Add to section 5: "The topic principal components are fitted on the outer training fold's Google users
  (no outcome is used) and reused inside its inner folds."
- P10. Section 5, Spotify: "basic-export people flagged" means reported, not a model feature (keeps 9 features).
- P11. Add to H3: "With 10 repeats, coverage, width and absolute error are averaged over repeats for each
  person-trait before pooling; Delta r uses the repeat-averaged predictions."
- P12. H4 "also reported for the text reader alone": the shipped z, no fitting; the CV-calibrated text model is
  reported beside it.
- P13. S3: "trait intercepts, common slopes for reader z, centred log words and their product".
- P14. S5: "Session-1 completers passing exclusion rules 1 to 4 and 7, whose retest also passes rule 3 and was
  taken 12 to 20 days after Session 1; ICC CI by person bootstrap (2,000)."
- P16. S1: "the H2 source model, 10-fold x 10 repeats on people with that source; the permutation null refits
  with one repeat, 1,000 permutations."
- P17. S4: "paired per-person squared-error difference against the shipped z, as in H2; refit weights need at
  least 30 people in the common-word subset."
- P18. S7(b) registered at n >= 250 (already in S7 above); computed and labelled exploratory from n = 60.
- **P22 (text reader change, 1 Nov 2026).** text-bigfive now refuses openness (no number, `usable=False`)
  when most of a person's words are Reddit or X, which is most participants (POWER.md section 9). "Pooled over
  the five traits" can then only be met by silently dropping or imputing O. Proposed wording, for H2, H3, H4
  and S7: "Pooled tests run over the traits for which the method gives a usable estimate to at least 90% of
  usable participants (decided from availability, before any outcome is computed, and printed with the
  results). Other traits are analysed per trait on the people who have them (correlations need at least 10
  people, cross-validated models at least 30). A person missing a pooled trait contributes the sum over the
  pooled traits they have; the number of such people is reported." With the readers as they are this pools
  C, E, A, N for the text reader and all five for `bayes` (fused O comes from the MBTI N/S letter alone). S4
  compares the raw-score refit with z = 0 for a trait the shipped reader mostly refuses, because the shipped
  answer there is "no number". The script implements this now; it is the only way to run the registered
  pooled tests on today's readers.
- P21 (found by the dry run). For every model learned under 10 repeats of CV (H2 Delta r, S1, S4, S7b), r is
  the mean of each repeat's own out-of-fold r, not r of the repeat-averaged prediction. For a model that
  learned little, the averaged prediction is close to (training mean minus a multiple of the person's own
  score), so its r runs towards -1 as repeats are added (simulation: Spotify-alone r -0.58 where the
  permutation null averaged -0.15), which also breaks the comparison with the one-repeat permutation null.
  Proposed wording for S1: "r is averaged over the 10 repeats of 10-fold CV; the permutation null uses one
  repeat". The paired error tests (H2, H4, S4) average per-repeat errors and are unaffected.
  `evaluate.predict` has the same issue and should get the same fix (not changed here).
- **P20 (found by the dry run, the most important).** The OLS stack in H2 can give large weights (up to 4.6 at
  n = 60, 2.4 at n = 250 in simulation) to a source's inner out-of-fold predictions. When a source carries
  little signal those predictions are nearly constant and run backwards under cross-validation, so the stack
  amplifies noise and the combined model loses to text for reasons unrelated to the question. Constraining the
  source weights to be non-negative (Breiman 1996, "Stacked regressions", Machine Learning 24:49-64) was better
  or equal in 7 of 8 simulated studies (n = 60: mean d -0.294 -> -0.056 in the worst case; n = 250: small but
  consistent gains). Proposed wording for H2: "their inner out-of-fold predictions stacked with the text reader
  z by least squares with the source weights constrained to be non-negative (a source whose weight would be
  negative is dropped from that fold's stack)". The script has it as `--stack-nonneg`; the default is still
  the registered OLS until the owner decides.


**Decisions on the items above, 2026-11-01.** All of P6 to P22 are accepted with the wording given. For P6
the conditional wording is adopted: H2 is primary only if the final n is 600 or more, otherwise secondary and
exploratory, decided by the stopping rule before outcomes are computed. For P20 the non-negative stack is the
registered method (`scripts/study_analysis.py` default; `--stack-ols` gives the plain OLS stack as a
sensitivity analysis). P21 is also fixed in `aas.evaluate.predict` (same date).

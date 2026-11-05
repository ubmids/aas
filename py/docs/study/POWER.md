# Sample size

Version 0.1, 1 Nov 2026. Every number here comes from `scripts/study_power.py` (seed 0, 1,000
simulated studies per cell for the increment, 2,000 for the rest), saved in
`docs/study/power_results.json`. Rerun:

```
cd py && uv run python scripts/study_power.py          # about 5 minutes
cd py && uv run python scripts/study_power.py --quick  # about 1 minute, noisier
```

n below always means **usable participants with both sources in the comparison**. Because writing is
required (PROTOCOL.md section 4), text + Google comparisons use most of n; text + Spotify comparisons
use only the Spotify users, perhaps half of n.

## 1. What effect sizes to plan for

All validities are against the quiz, so the quiz's own noise is already inside them.

| source | evidence so far | planning value |
|---|---|---|
| Text, Big Five | in domain (Essays) AUC 0.57 to 0.68, implied r 0.16 to 0.40, only O clearly real; cross-domain transfer AUC 0.55 to 0.67; the field's footprint bar is r 0.3 to 0.4 with in-domain training | **r = 0.20** central, 0.30 optimistic, 0.10 pessimistic |
| Text, MBTI letters | in domain AUC 0.71 to 0.88; on Essays vs related Big Five 0.55 to 0.59 | AUC 0.60 to 0.70 vs OEJTS |
| Google search / YouTube | no labelled data; we have no reader | r 0.10 to 0.25, unknown |
| Spotify | listening traits are reliable (0.72 to 0.99) but music taste predicts personality only weakly | r about 0.10 |
| Sensors (for comparison) | E r about 0.26 to 0.30 in two small datasets, shrinking when pooled | not in this study |

Effects first seen in small samples usually shrink, so the central values are deliberately modest.

## 2. (a) Estimating one source's validity within +-0.10

Fisher-z 95% CI for a correlation; the simulation agrees to within 0.01.

| true r | n for +-0.10 | half-width at n = 60 | n = 250 | n = 600 |
|---|---|---|---|---|
| 0.10 | **378** | 0.25 | 0.12 | 0.08 |
| 0.20 | **355** | 0.24 | 0.12 | 0.08 |
| 0.30 | **320** | 0.23 | 0.11 | 0.07 |

So **about 350 people** pin one validity to +-0.10. At 250 it is +-0.12, at 60 it is +-0.25 (a validity of
0.20 would be "somewhere between -0.05 and 0.45", which says almost nothing).

This is for the pre-trained text reader, scored with no refitting. A model learned from study people (Google,
Spotify) is worse off, because it must also learn its weights from the same people:

| true R of a 10-feature in-study model | mean CV r at n = 60 | n = 250 | n = 600 |
|---|---|---|---|
| 0.10 | -0.06 | 0.01 | 0.04 |
| 0.15 | -0.02 | 0.05 | 0.10 |
| 0.25 | 0.07 | 0.17 | 0.21 |

At n = 60 a learned model scores below zero on average even when there is a real signal of 0.15 (this is the
same "noise runs backwards under CV" effect described in `evaluate.py`). At 250 a true 0.15 shows up as about
0.05. Only R of 0.25 or more is visible at 250.

## 3. (b) Detecting an increment from adding a source

Setup (`one_increment_trial`): the text reader's score S1 is given (pre-trained). The second source is 10
features whose best combination has validity r2 on its own; it is learned by ridge inside 10-fold CV, stacked
with S1 using inner 5-fold out-of-fold predictions, and the whole thing is scored on held-out people. The test
is one-sided, paired, per person: is the combined model's squared error smaller than the text reader's
(linearly calibrated, also inside CV)? `rho` is the correlation between the two sources' non-trait parts
(0 = independent; 0.3 = they share variance unrelated to personality, e.g. "how online you are").

r2 needed for a population increment: with text r1 = 0.20, the second source must have r2 = 0.16 on its own
(independent errors) for Delta r = 0.05, or r2 = 0.23 for Delta r = 0.10. With r1 = 0.30: 0.20 and 0.29.

Power, one trait (alpha 0.05 / alpha 0.01, the latter being Holm's first step for 5 traits):

| text r1 | target Delta r | rho | n = 60 | n = 250 | n = 600 | mean CV Delta r at 250 / 600 |
|---|---|---|---|---|---|---|
| 0.20 | 0.05 | 0 | 0.04 / 0.01 | 0.07 / 0.01 | 0.29 / 0.07 | 0.01 / 0.03 |
| 0.20 | 0.05 | 0.3 | 0.03 / 0.01 | 0.10 / 0.01 | 0.40 / 0.12 | 0.02 / 0.03 |
| 0.20 | **0.10** | 0 | 0.04 / 0.01 | **0.26 / 0.07** | **0.84 / 0.58** | 0.06 / 0.08 |
| 0.20 | 0.10 | 0.3 | 0.04 / 0.01 | 0.32 / 0.08 | 0.89 / 0.65 | 0.07 / 0.09 |
| 0.30 | 0.05 | 0 | 0.02 / 0.00 | 0.16 / 0.03 | 0.58 / 0.27 | 0.02 / 0.03 |
| 0.30 | 0.05 | 0.3 | 0.03 / 0.01 | 0.21 / 0.05 | 0.74 / 0.37 | 0.03 / 0.04 |
| 0.30 | 0.10 | 0 | 0.06 / 0.01 | 0.53 / 0.21 | 0.99 / 0.91 | 0.07 / 0.09 |
| 0.30 | 0.10 | 0.3 | 0.07 / 0.01 | 0.62 / 0.27 | 1.00 / 0.95 | 0.08 / 0.09 |

Omnibus over the five traits (the pre-registered primary test, PREREG H2; simulated here as Stouffer over five
independent traits **that all have the same increment**, which is optimistic):

| text r1 | Delta r | n = 60 | n = 250 | n = 600 |
|---|---|---|---|---|
| 0.20 | 0.05 | 0.00 | 0.06 | 0.59 |
| 0.20 | 0.10 | 0.00 | **0.56** | **1.00** |
| 0.30 | 0.10 | 0.02 | 0.97 | 1.00 |

Power curve for the central case (text r1 = 0.20, Delta r = 0.10, rho = 0):

| n | 100 | 150 | 250 | 400 | 600 | 900 | 1,300 |
|---|---|---|---|---|---|---|---|
| one trait, alpha .05 | 0.07 | 0.10 | 0.28 | 0.60 | 0.85 | 0.99 | 1.00 |
| one trait, alpha .01 | 0.01 | 0.03 | 0.08 | 0.27 | 0.58 | 0.88 | 1.00 |
| omnibus, 5 traits | 0.03 | 0.11 | 0.56 | 0.95 | 1.00 | 1.00 | 1.00 |

And for Delta r = 0.05: one trait at alpha .05 reaches 0.82 only at **n = 1,300**; the omnibus reaches 0.64
at 600 and 0.97 at 900.

What this says:

1. **Delta r = 0.10 needs about 600 people** for a per-trait answer, or **about 400** for the omnibus "does
   combining help on average" question.
2. **Delta r = 0.05 is out of reach** of any scenario here except as an omnibus at 600+ (power about 0.6). A
   0.05 increment is the size most published multi-source gains actually are, so the study can at best show
   "large increments are absent or present".
3. The in-sample nested F-test would suggest far fewer people (97 to 329 for 80%), but it assumes the second
   source's weights are known. They are not; learning them costs about half of the increment at n = 250
   (target 0.10, mean observed 0.055). The CV numbers above are the honest ones.
4. At n = 60 nothing about increments can be learned. The 90% range of the observed Delta r runs from about
   -0.15 to +0.40.
5. The omnibus assumes all five traits gain equally. If only one or two do (likely: Openness carries most text
   signal), omnibus power falls towards the one-trait numbers.

## 4. (c) Estimating interval coverage within +-5 points

Coverage is a proportion; for a true 80% the 95% half-width is 1.96 * sqrt(0.16 / n).

| | n for +-5 points | n = 60 | n = 250 | n = 600 |
|---|---|---|---|---|
| one trait | **246** | +-10.1 | +-5.0 | +-3.2 |
| pooled over 5 traits, misses independent across traits | | +-4.5 | +-2.3 | +-1.4 |
| pooled, moderate within-person correlation (ICC 0.2) | | +-5.5 | +-2.6 | +-1.7 |
| pooled, strong within-person correlation (ICC 0.5) | | +-6.8 | +-3.2 | +-2.1 |

So **about 250 people** check one trait's coverage to +-5 points; pooled over traits, 60 people get close to
+-5 to 7 points, which is the one primary question the minimal pilot can answer. The pooled CI in the analysis
is a person-level (cluster) bootstrap, so the within-person correlation is handled whatever it turns out to be.

## 5. Side estimates

- **Quiz retest reliability** (expected about 0.85 at 2 weeks): 95% half-width 0.11 with 30 people, 0.07 with
  60, 0.06 with 100. Retest subsets: 30 / 60 / 100 in the three scenarios.
- **Skill over guessing the average** (Q4, `skill_power`). Even a perfectly calibrated predictor with validity r
  has small MAE skill, about 1 - sqrt(1 - r^2): 0.005 at r = 0.10, 0.02 at r = 0.20, 0.046 at r = 0.30. Power of a
  one-sided paired test that skill > 0, one trait, alpha .05:

  | validity r | n = 60 | n = 250 | n = 600 |
  |---|---|---|---|
  | 0.10 | 0.09 | 0.16 | 0.26 |
  | 0.20 | 0.17 | 0.39 | 0.67 |
  | 0.30 | 0.25 | 0.68 | 0.94 |

  Skill is the honest product metric (it is what a user would feel), but it is a weak test. It is pre-registered
  pooled over traits (sum of per-person differences), which helps, and reported per trait as estimates.
- **MBTI letter AUC vs OEJTS** (balanced letters): half-width about 0.14 at n = 60, 0.07 at 250, 0.04 to 0.05 at
  600. Real letters are not balanced, which widens these slightly.

## 6. Three scenarios

| | **Minimal pilot-plus, n = 60** | **Main, n = 250** | **Ideal, n = 600** |
|---|---|---|---|
| central cost (RECRUITMENT_AND_BUDGET.md) | about $3,600 | about $14,700 | about $35,100 |
| Q1 text validity per trait | +-0.24: only tells "near zero" from "about 0.4" | +-0.12: separates 0.1 from 0.3 | +-0.08 |
| Q2 increment, Delta r 0.10 | no power (under 0.1) | omnibus about 0.56, per trait about 0.3: a coin flip | per trait 0.84 (0.58 after Holm), omnibus near 1 |
| Q2 increment, Delta r 0.05 | no | no | omnibus about 0.6, per trait no |
| Q3 coverage | pooled over traits +-4.5 to 7 points: roughly; per trait no | per trait +-5: **yes** | per trait +-3: yes |
| Q4 skill vs guess-average | no | per trait only if r is about 0.3 (power 0.68); pooled over traits better | per trait at r 0.2 to 0.3 (0.67 to 0.94) |
| Q5 Google/Spotify alone (learned in study) | no (CV r negative on average) | only R 0.25+ visible | R 0.15+ visible |
| Retest reliability | +-0.11 | +-0.07 | +-0.06 |
| Can answer | Does the pipeline work end to end? Are the text reader's intervals grossly wrong? Is text validity near 0 or clearly positive? | Text reader's real validity and calibration on ordinary people; coverage per trait; whether a large increment exists on average | Per-trait increments of 0.10; first look at 0.05 on average; learned non-text models |
| Cannot answer | Anything about increments or non-text sources | Increments per trait; anything about 0.05 increments; weak non-text sources | 0.05 increments per trait (needs about 1,300) |

## 7. Recommendation

- Run the **10-person pilot**, then decide between 250 and 600 with measured return rates.
- **n = 250 is the smallest study worth its cost**: it settles Q1 and Q3 (is the text reader any good on ordinary
  people, and are its intervals honest?), which the product depends on regardless of combination. It answers Q2
  only for large, broad increments.
- **n = 600 is what Q2 as stated needs** (Delta r = 0.10 per trait). If the owner cares most about "whole person
  beats one slice", plan 600, or plan 250 with a pre-registered extension to 600 (PREREG stopping rule allows a
  second, fixed stage; no peeking at outcomes in between).
- **n = 60 is not a study**, it is an expensive pilot. It would be useful only to show the instrument works and
  that intervals are not wildly off.

## 8. Assumptions most likely to be wrong

1. **Text r = 0.20.** If the true out-of-domain validity is 0.10, Q2 is about adding to almost nothing, and all
   increments shrink. The numbers above for r1 = 0.20 are then roughly right for the increment itself, but the
   headline "combined model" would still be weak.
2. **Second source as 10 features with reliability 0.9.** More features (e.g. 50 topic shares and histograms)
   learned from the same n will shrink further under CV. The analysis plan therefore fixes small feature sets per
   source in advance (PREREG section 5).
3. **Independent traits in the omnibus.** The pre-registered omnibus uses the per-person sum of loss differences
   over traits, which handles within-person correlation correctly; the simulation approximated it with Stouffer
   over independent traits.
4. **Spotify n.** Text + Spotify comparisons only have the Spotify users; if that is half the sample, use the
   power at n/2.

## 9. Measured validities, 2026-11-01

Sections 1 to 8 planned around a text reader with r = 0.20 on ordinary people. That number was a guess. It has
now been measured on kinds of writing the reader never saw (`docs/text_crossdomain.md`), and the reader reports
it (`docs/readers_text.md`, "Validity by kind of writing"). The old sections are kept unchanged; every number
below is in `power_results.json` under `e_measured_2026_11_01` (computed last, so the earlier numbers
reproduce exactly).

### What the readers can actually do on Reddit and X

| trait | text-bigfive r (from tweets AUC) | 95% CI of that AUC (294 people) | text-mbti r via link (inferred, untested off forum) | zero-data fused r |
|---|---|---|---|---|
| O | **refused** (r about 0) | 0.46 to 0.59 | 0.20 | 0.20 (MBTI only) |
| C | 0.097 | 0.47 to 0.61 | 0.13 | 0.14 |
| E | 0.095 | 0.47 to 0.62 | 0.19 | 0.19 |
| A | 0.094 | 0.47 to 0.61 | 0.12 | 0.13 |
| N | 0.158 | 0.51 to 0.64 | none | 0.16 |

Text r is the closed-form `auc_to_r` of the card's tweets AUC (E: AUC 0.5427, r 0.095), the same conversion the
reader and fusion use.

Reddit is not measured yet; it uses the same numbers as tweets (the lowest seen on a person's own writing). The
planning value in section 1 was text r = 0.20 central, 0.10 pessimistic. **The measured value is the pessimistic
one** for C, E and A, a little better for N, and zero for O.

### Q1 again: can the study show the text reader works at all?

Power to show text r > 0 (two-sided, alpha .05), and people needed for 80%:

| trait | r | n = 60 | n = 250 | n = 600 | n for 80% | 95% CI at n = 250 |
|---|---|---|---|---|---|---|
| O | 0 (refused) | | | | never | |
| C | 0.097 | 0.11 | **0.33** | 0.66 | 832 | -0.03 to 0.22 |
| E | 0.095 | 0.11 | **0.32** | 0.64 | 868 | -0.03 to 0.22 |
| A | 0.094 | 0.11 | **0.32** | 0.63 | 887 | -0.03 to 0.22 |
| N | 0.158 | 0.23 | **0.71** | 0.97 | 313 | 0.04 to 0.28 |

With Holm's first step (alpha .01) the n for 80% is about 1,240 to 1,320 for C, E, A and 464 for N.
Precision (section 2) does not change: n = 250 still pins r to about +-0.12. But with r about 0.10 that interval
runs from slightly below 0 to about 0.22, so the study would tell us "weak, maybe nothing", which the
cross-domain test already says.

**Fused (text + MBTI) r > 0**, the cheapest positive claim: n for 80% power 200 (O), 223 (E), 313 (N), 429 (C),
493 (A). Power at n = 250: 0.88, 0.85, 0.71, 0.57, 0.51. Before (fusion.md section 5) the same claim needed 62 to
194 people. And the O and E parts of this lean on the MBTI letters, whose off-forum accuracy is inferred: on the
tweet spot checks the N/S letter was weaker (AUC 0.54 vs O) than the inference assumes.

### Q2 again: increments over the best single source

With text r1 = 0.10 (measured) instead of 0.20, a second source needs r2 = 0.11 alone for Delta r = 0.05 and 0.18
for Delta r = 0.10. Learned inside the study (10-feature ridge, nested CV, as in section 3):

| Delta r | n = 60 | n = 250 | n = 600 | before (r1 = 0.20) at 250 / 600 |
|---|---|---|---|---|
| 0.10, one trait, alpha .05 | 0.03 | **0.11** | 0.43 | 0.26 / 0.84 |
| 0.10, one trait, alpha .01 | 0.01 | 0.02 | 0.16 | 0.07 / 0.58 |
| 0.10, omnibus over 5 traits | 0.02 | **0.11** | 0.83 | 0.56 / 1.00 |
| 0.05, one trait, alpha .05 | 0.03 | 0.04 | 0.09 | 0.07 / 0.29 |
| 0.05, omnibus | 0.00 | 0.01 | 0.06 | 0.06 / 0.59 |

Lower r1 makes the increment test harder, not easier: the second source needed for the same Delta r is weaker,
and a weak signal learned from the same people is mostly eaten by cross-validation (mean observed Delta r at 250 is
0.06 for a true 0.10).

### Q4 again: skill over guessing average

At r 0.10 the best possible skill is 0.005 (power 0.16 at n = 250, 0.25 at 600); at r 0.16 it is 0.013 (0.29 and
0.49). A user would not feel the difference from "average on every trait".

### Q3 (coverage) is still answerable, but means less

Coverage precision does not depend on validity (section 4 stands: +-5 points for one trait at n = 246). But with
r about 0.10 the reader's 80% interval is +-1.27 z, almost the +-1.28 z of "anyone". Showing that it covers 80%
mostly shows the quiz norms are right, not that the reader knows anything.

### What a better text model (PANDORA) must deliver

The smallest validity a trait's reader needs for the study to show it above zero with 80% power:

| n | alpha .05 | alpha .01 (Holm first step) |
|---|---|---|
| 60 | 0.36 | 0.42 |
| **250** | **0.18** | **0.21** |
| 600 | 0.11 | 0.14 |

So for the planned main study (n = 250) to answer Q1 per trait, a retrained text model must reach **r of about
0.20 on held-out Reddit authors** (and not collapse on tweets) on most traits. That is roughly double today's
measured 0.10, and roughly what Reddit-trained models report in domain (about r 0.2 to 0.35 in the PANDORA paper as commonly cited; not re-checked here). For Q2 to be
answerable at 250 as section 3 planned, it needs r1 about 0.20 to 0.30 **and** a second source with r2 of about
0.2 or more, which no non-text source has shown.

### Should the study wait? Yes, the main study should.

- **Run the 10-person pilot now.** It tests the instrument, Session 2 return and timings, none of which depends on
  the text model.
- **Do not run the n = 250 main study with today's text reader.** At measured validities it has about a 1 in 3
  chance of showing the text reader works on C, E or A, nothing to say about openness (refused), little power for
  fusion except E and O through an untested MBTI inference, and about 0.11 power for its headline increment
  question. About $15,000 would mostly buy a confirmation of what the leave-one-domain-out test already shows.
- **Request PANDORA now** (free, the loader exists), retrain, and score the new model on held-out Reddit authors
  and on PAN tweets before retraining anything else. Gate for the main study: **r >= 0.20 on held-out Reddit
  authors for at least three traits, with tweets not below about 0.12**. If it gets there, 250 is worth running
  as planned. If it lands at 0.12 to 0.18, the study needs about 600 for Q1 alone. If it stays near 0.10, redesign
  the study as data collection (the opt-in common-word block) to train a model on ordinary people, and say so in
  PREREG, rather than as a test of the current readers.

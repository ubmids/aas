# sensors pooled: does the extraversion signal survive putting phones and wearables together?

Run it: `cd py && uv run python scripts/run_pool_sensors.py`
Code: `src/aas/slices/sensors_pooled.py` (harmonised features + the four analyses).

Round 1 (`phone_body.md`) found extraversion at r ~ +0.26 in StudentLife (n = 46,
phone) and in LifeSnaps (n = 52, Fitbit), each with raw p ~ 0.035, neither
surviving Holm. Two small datasets that point the same way are worth more
together than apart, but only if they measure the same things. This round
builds one small feature set both datasets compute with the same code, and asks
the question four ways.

## 1. Pre-registration

Written on 2026-11-01, after reading round 1 and before any harmonised feature
was correlated with any trait or any model was scored on them. Nothing below
was changed after results were seen; anything tried afterwards is in section 5
and labelled exploratory.

**Honest caveat about what was already seen.** Round 1 printed single-feature
correlations for each dataset's own features. Three of them overlap with
hypotheses below, so for those the dataset named is *not* independent evidence
(it already informed the guess): LifeSnaps activity_centroid_h vs E was +0.31
(H3), StudentLife sleep_onset_sd vs O was +0.59 (H9), StudentLife sleep_h vs C
was +0.38 (not made a hypothesis for that reason). Also seen: LifeSnaps VO2max
vs E -0.33, which runs against H1 if fitness tracks steps.

### 1.1 Harmonised features (8)

Each dataset is first turned into the same object: one person's **hourly
activity series** (local clock time, one value per hour, NaN when the hour was
not observed). Everything except sleep_h is then computed from that series by
the same function, `rhythm_features()`.

| | StudentLife (phone) | LifeSnaps (Fitbit) |
|---|---|---|
| hourly value | minutes moving (accelerometer classifier says walking or running) per hour: share of that hour's samples labelled 1 or 2, times 60. Samples labelled 3 (unknown) dropped | steps in that hour (authors' hourly CSV) |
| hour unobserved when | no activity samples at all in that hour | steps is NaN (watch off or not synced; bpm is missing in the same hours 94% of the time) |
| clock | US/Eastern, term 2013-03-24 .. 2013-06-06 | local date/hour as exported |

Shared rules: a **valid day** has >= 18 of 24 hours observed; a person needs
>= 7 valid days or the feature is NaN. On valid days the missing hours are
filled with that person's own mean for that clock hour. Units differ (minutes
moving vs steps) but every feature is either unit-free or z-scored within its
dataset before any comparison, so only each person's position *within their own
dataset* matters.

| feature | definition (same code) | compromise |
|---|---|---|
| activity_level | log(1 + mean daily total) | phone in a bag still counts walking; phone on a desk reads "still" while the person may be moving. Steps are a cleaner measure. Rank within dataset is what's compared |
| activity_cv | SD / mean of daily totals across valid days | |
| weekend_ratio | log(mean Sat+Sun total / mean Mon-Fri total) | a term-time student's weekend is not a working adult's weekend |
| interdaily_stability | actigraphy IS (van Someren 1999): variance of the average 24 h profile / total hourly variance, 0..1 | hourly, not the classic minute data |
| relative_amplitude | (M10 - L5) / (M10 + L5) on the average 24 h profile, circular windows (most active 10 h vs least active 5 h) | |
| activity_centroid_h | activity-weighted *circular* mean clock hour, written on a 04:00..28:00 axis (later = evening type) | round 1 used a linear 0..23 mean; circular avoids the midnight split |
| rest_midpoint_sd | per noon-to-noon window with >= 18 observed hours: midpoint of the least-active 5 consecutive hours (L5); SD across nights (needs >= 7 nights) | a phone left on the desk at night looks like rest; a Fitbit taken off at night is NaN, then filled from the person's own profile. This is a *rest timing* measure built identically for both, standing in for bedtime regularity, which LifeSnaps' CSVs do not record |
| sleep_h | device-native main-night sleep, mean over >= 7 nights. StudentLife: longest phone-in-darkness episode starting 20:00..05:00 (round 1's sleep_h). LifeSnaps: minutesAsleep / 60, nights > 2 h | **not the same code**: dark-phone time vs Fitbit-scored sleep. Kept because nothing else measures sleep in both; treat it as the weakest mapping |

Implementation details fixed while writing the code, before any trait was
touched: L5 ties (a night of several zero-activity hours gives many equally
"least active" windows) are broken by taking the centre of the longest run of
tied windows, not the first; weekend_ratio needs >= 2 weekend and >= 2 weekday
valid days and uses log((weekend + 1)/(weekday + 1)). The label-free feature
build gave n = 46 StudentLife (all 8 features; sleep_h for 44) and 52 LifeSnaps
people, of whom 6 have fewer than 7 valid days and fewer than 7 scored nights
(watch off at night) and therefore no harmonised feature at all. They are
excluded, so LifeSnaps contributes 46, not 52, and the pooled n is **92**, not
98. Remaining gaps (2 weekend_ratio, 3 sleep_h) are median-imputed inside the
model on training people only.

Dropped from the shared set on purpose: anything social (conversation,
Bluetooth: phone only), heart rate and VO2max (watch only), GPS/WiFi mobility
(phone only), compliance counts (different study designs). Stachl et al. 2020
found extraversion carried mostly by communication and social behaviour, which
the shared set cannot see, so the pooled model is expected to be *weaker* on E
than either full round-1 model. That is the price of comparability.

### 1.2 Hypotheses

Direction tested two-sided; "expected sign" is the bet. Evidence base:
- Stachl et al. 2020, PNAS 117(30), n = 624 smartphone users: E was the best
  predicted domain (median r ~ 0.37), carried mostly by communication/social
  behaviour and, after that, by mobility and **day/night activity patterns**;
  C was predicted more weakly, with app use and **day-night rhythm**
  (regularity) among its informative behaviours.
- Rhodes & Smith 2006 (Br J Sports Med, meta-analysis): physical activity
  correlates positively with E and C (small effects).
- Tsaousis 2010 and later chronotype meta-analyses: morningness correlates
  positively with C (the most robust personality-chronotype link) and weakly
  with lower E (evening types slightly more extraverted).

| id | pair | expected sign | why |
|---|---|---|---|
| H1 | E x activity_level | + | extraverts more active (Rhodes & Smith 2006); mobility in Stachl 2020 |
| H2 | E x weekend_ratio | + | going out at weekends (weak prior, sociability not otherwise visible here) |
| H3 | E x activity_centroid_h | + | evening types slightly more extraverted; Stachl 2020 day/night patterns. *LifeSnaps already seen (+0.31)* |
| H4 | C x interdaily_stability | + | regular days (Stachl 2020: rhythm regularity informative for C) |
| H5 | C x rest_midpoint_sd | - | irregular rest timing = less orderly |
| H6 | C x activity_centroid_h | - | morningness and C |
| H7 | C x activity_cv | - | erratic week-to-week activity |
| H8 | C x activity_level | + | Rhodes & Smith 2006 |
| H9 | O x rest_midpoint_sd | + | replication test of round 1's one Holm survivor (StudentLife bedtime SD vs O, +0.59). *StudentLife already seen; LifeSnaps is the only independent test* |

Confirmatory family = H1..H9, Holm across the 9. Primary question = E (H1-H3
and analyses a, b for E); C is secondary; O only as H9.

### 1.3 Analyses (fixed in advance)

- **(a) Pooled model.** Each harmonised feature and each trait z-scored within
  its dataset, stacked (n ~ 98), plus a 0/1 dataset column (fixed effect).
  Scored with `evaluate.predict()` (ridge, 5-fold x 5 repeats, permutations,
  bootstrap, Holm across the 5 traits) with folds **stratified by dataset** so
  both appear in every fold. 1000 permutations. Read r against null_mean.
- **(b) Transfer.** Fit the same ridge pipeline on all of one dataset, predict
  the other, for each trait. r between prediction and truth in the test set;
  permutation p by shuffling the test labels 10,000 times (no refit needed: the
  model never saw the test set). Both directions. The ridge penalty is chosen
  by RidgeCV inside the training set only.
- **(c) Meta-analysis.** For H1..H9: Spearman rho per dataset, Fisher z with
  Bonett-Wright SE sqrt((1 + rho^2/2)/(n - 3)), fixed-effect (inverse-variance)
  and DerSimonian-Laird random-effects pooled rho, Cochran Q and I^2. p from the
  random-effects z (equal to fixed when tau^2 = 0). Holm across the 9.
- **(d) Power.** For the pooled E effect from (a) and the strongest pooled
  pre-registered E correlation from (c): n for 80% power at alpha .05
  two-sided and at alpha .01 (= .05 Holm-split over 5 traits), via Fisher z;
  and the power GLOBEM's 497 people would give.

Verdict rule (set now): "the E signal holds up" requires the pooled E model to
beat its null with Holm p < .05 **and** at least one transfer direction with
p < .05. One without the other is "suggestive". Neither is "does not hold up".

---

(Results below were added after running.)

## 2. Results

Everything here is the pre-registered analysis unless marked EXPLORATORY.
CSVs: `docs/sensors_pooled_{a_pooled,a_within,b_transfer,c_meta,d_power,x_all_pairs}.csv`.
Two runs gave identical numbers. The model scores in 2.1 and 2.5 were re-run on 2026-11-05 with the
per-repeat r of the 1 Nov `evaluate.predict` fix (old numbers in section 5); transfer, meta-analysis and
power do not use that averaging and did not change.

### 2.0 Did the harmonisation work? (no labels involved)

If the 8 features mean the same thing in both datasets, the way they relate *to
each other* should look alike. Across the 28 feature pairs, the StudentLife and
LifeSnaps inter-correlation matrices correlate at **r = 0.55**: similar, not the
same. The rhythm features agree well (interdaily_stability vs activity_cv -0.81
vs -0.60; relative_amplitude vs rest_midpoint_sd -0.49 vs -0.51). The odd one
is weekend_ratio: tied to activity_level in StudentLife (+0.67, students who
move more do it at weekends) and unrelated in LifeSnaps (-0.09). sleep_h also
behaves differently (dark-phone time vs Fitbit sleep), as expected.

### 2.1 (a) Pooled model: 8 features + dataset dummy, n = 92, folds stratified

Every test fold held 9-10 people from each dataset.

| trait | r | 95% CI | null mean | p_perm | p_holm | max-penalty share |
|---|---|---|---|---|---|---|
| O | -0.126 | -0.25 .. +0.00 | -0.155 | 0.327 | 1.00 | 0.20 |
| C | -0.065 | -0.23 .. +0.10 | -0.153 | 0.207 | 1.00 | 0.04 |
| E | -0.129 | -0.26 .. -0.01 | -0.158 | 0.337 | 1.00 | 0.16 |
| A | -0.172 | -0.26 .. -0.08 | -0.153 | 0.505 | 1.00 | 0.68 |
| N | -0.218 | -0.31 .. -0.11 | -0.155 | 0.675 | 1.00 | 0.96 |

Nothing beats its own null by a margin that counts (smallest p_perm 0.21, C). E sits 0.03 above the
shuffled-label baseline, p_perm 0.34.
CV seeds 1 and 2 give E -0.11/-0.12 and C -0.09/-0.05, so this is not a bad split.

Context (same scorer, same 8 features, one dataset at a time, n = 46 each):
StudentLife E -0.23 (null -0.19), LifeSnaps E -0.00 (null -0.19). Round 1 got
E +0.24 and +0.24 from each dataset's **full** feature set (+0.27 and +0.26 before the 1 Nov fix). So the extraversion
signal of round 1 lives in the features that could not be harmonised
(conversation, Bluetooth, GPS in StudentLife; VO2max and heart rate in
LifeSnaps), not in the shared activity/sleep/rhythm set.

### 2.2 (b) Transfer: train on one dataset, test on the other

| train -> test | trait | r | 95% CI | p_perm | ridge gave up? |
|---|---|---|---|---|---|
| StudentLife -> LifeSnaps | O | -0.244 | -0.51 .. +0.12 | 0.947 | no |
| | C | +0.114 | -0.21 .. +0.41 | 0.223 | no |
| | **E** | **+0.035** | -0.24 .. +0.33 | **0.401** | no |
| | A | +0.041 | -0.22 .. +0.30 | 0.388 | yes |
| | N | -0.208 | -0.47 .. +0.10 | 0.918 | yes |
| LifeSnaps -> StudentLife | O | -0.036 | -0.25 .. +0.21 | 0.602 | no |
| | C | +0.232 | -0.06 .. +0.48 | 0.056 | yes |
| | **E** | **+0.082** | -0.16 .. +0.31 | **0.288** | no |
| | A | +0.046 | -0.28 .. +0.36 | 0.402 | yes |
| | N | -0.142 | -0.45 .. +0.13 | 0.826 | no |

n = 46 train, 46 test each way; 10,000 test-label shuffles; null mean ~0 here
(no cross-validation bias, because the test people are a separate dataset).
E does not transfer in either direction. The LifeSnaps -> StudentLife C result
(+0.23, p = .056) comes from a model whose ridge chose the maximum penalty, so
its predictions are nearly flat; a correlation from near-constant predictions
is fragile and should not be read as a lead. Looking at the weights: trained on
LifeSnaps, the E model leans almost entirely on activity_centroid_h (+0.15);
trained on StudentLife it shrinks everything to ~0.

### 2.3 (c) Meta-analysis of the pre-registered pairs

Spearman rho per dataset, Fisher z, fixed and random effects (identical here:
tau^2 = 0 and I^2 = 0 for every pair, the two datasets never disagree more than
chance), Holm across 9.

| id | pair | bet | rho SL (n 46) | rho LS (n 44-46) | pooled rho | 95% CI | p | p_holm |
|---|---|---|---|---|---|---|---|---|
| H1 | E x activity_level | + | +0.03 | +0.11 | +0.07 | -0.15 .. +0.27 | 0.55 | 1.00 |
| H2 | E x weekend_ratio | + | +0.04 | -0.06 | -0.01 | -0.22 .. +0.20 | 0.95 | 1.00 |
| **H3** | **E x activity_centroid_h** | + | **+0.27** | **+0.32** | **+0.30** | +0.09 .. +0.48 | **0.006** | **0.053** |
| H4 | C x interdaily_stability | + | +0.26 | +0.22 | +0.24 | +0.03 .. +0.43 | 0.026 | 0.21 |
| H5 | C x rest_midpoint_sd | - | -0.24 | -0.15 | -0.20 | -0.39 .. +0.01 | 0.065 | 0.46 |
| H6 | C x activity_centroid_h | - | -0.07 | -0.02 | -0.04 | -0.25 .. +0.17 | 0.70 | 1.00 |
| H7 | C x activity_cv | - | -0.21 | -0.09 | -0.15 | -0.35 .. +0.06 | 0.17 | 1.00 |
| H8 | C x activity_level | + | +0.06 | +0.06 | +0.06 | -0.15 .. +0.27 | 0.58 | 1.00 |
| H9 | O x rest_midpoint_sd | + | +0.09 | +0.16 | +0.12 | -0.09 .. +0.32 | 0.25 | 1.00 |

- 8 of 9 pooled estimates point the predicted way (H2 is the exception), and
  all 9 have the same sign in both datasets except H2. That pattern is more
  encouraging than any single p-value, but none of the 9 survives Holm.
- **H3** is the closest: people whose activity is centred later in the day
  report more extraversion, rho ~ +0.3 in both datasets, Holm p = 0.053. Recall
  the LifeSnaps half was already seen in round 1 (+0.31 there with a linear
  centroid). The uncontaminated evidence is the StudentLife half alone: rho
  +0.27, p = .07 by itself. So: a pre-registered, independently replicated
  direction, of the size Stachl et al. reported for E, that falls just short of
  the correction bar.
- **H4** (C x routine regularity, +0.24 in both) is the cleanest C result: both
  halves were genuinely new, both positive, raw p = .026, but Holm p = .21.
- **H9** (round 1's Openness lead): the bedtime-irregularity result does not
  replicate with the harmonised rest-timing measure. LifeSnaps, the only
  independent test, gives +0.16 (p ~ .3); even StudentLife falls from +0.59
  (dark-phone bedtime SD) to +0.09 (accelerometer rest timing). Treat the round
  1 O finding as unconfirmed.

### 2.4 (d) Power

The pooled E model has no positive effect (r -0.13 vs null -0.16, p_perm 0.34),
so there is nothing to power from (a). For the effects that do exist:

| effect | r | n for 80% power, alpha .05 | alpha .01 | power at our n = 92 | power at GLOBEM n = 497 (alpha .05 / .01) |
|---|---|---|---|---|---|
| H3 pooled rho (E x activity timing) | 0.30 | 89 | 130 | 0.82 | 1.00 / 1.00 |
| H3 lower 95% CI bound [added after results] | 0.09 | 1,025 | 1,524 | 0.13 | 0.50 / 0.27 |
| round-1 E model r (full features, each dataset) | 0.26 | 114 | 168 | 0.71 | 1.00 / 1.00 |
| Stachl 2020 E model r | 0.37 | 56 | 81 | 0.96 | 1.00 / 1.00 |

Read this carefully: 0.30 is a *maximum-of-three* estimate (the best of the
three E pairs) and half of it was seen before, so the true effect is likely
smaller. Effects reported first in small samples usually shrink. If the truth
is r ~ 0.15 to 0.2, you need about 200 to 350 people for 80% power; GLOBEM's
497 covers that comfortably (power 0.92 at r = 0.15 and 0.99 at r = 0.20, alpha .05), and covers r =
0.26 to 0.30 with near certainty. Caveats: GLOBEM's personality measure is, per
its documentation, a short Big Five form (BFI-10, check on access), whose
lower reliability shrinks observed correlations by roughly 10-20%; and its
Fitbit + phone streams support every feature in the harmonised set here, so
this exact pipeline could be re-run on it.

### 2.5 EXPLORATORY (chosen after seeing the results; nothing here counts as confirmation)

- **H3 as a model.** A pooled model using only activity_centroid_h + the dataset
  dummy predicts E on unseen people at r = +0.13 vs null -0.18, p_perm = 0.026.
  It works where the 8-feature model fails because, at n = 92, seven features
  with no E signal add enough noise to drown one feature with a modest signal.
  This model was picked by looking at (c), so its p-value is optimistic.
- Pearson (rather than Spearman) for H3: StudentLife +0.24 (p .11), LifeSnaps
  +0.32 (p .03). Same story, a touch weaker in StudentLife (a few students with
  very late centroids).
- All 40 feature x trait meta-analyses (Holm over 40): the top two are H3 and
  H4; nothing survives (smallest Holm p = 0.23). No surprise pair appeared.

## 3. Verdict

**The extraversion signal does not hold up as a model when the two datasets are
used together.** By the rule set in advance, it fails both tests: the pooled
model predicts E no better than shuffled labels, and a model trained on one
dataset does not predict E in the other, in either direction.

**But one behaviour does carry a consistent, if small, E signal: when in the day
you are active.** People whose movement is centred later in the day report more
extraversion, by about the same amount (rho ~ 0.3) on a 2013 US student's phone
and on a 2021 European adult's Fitbit. It just misses the multiple-testing bar
(Holm p = 0.053), and part of it was seen before it was predicted, so it is a
lead to test on fresh data, not a finding.

**Most of round 1's E signal came from behaviours only one device can see.**
With the shared activity/sleep/rhythm features, each dataset on its own also
loses its E signal (r +0.24 and +0.24 in round 1 vs -0.23 and -0.00 now). That
fits Stachl et al. 2020, where extraversion was carried mainly by social and
communication behaviour. Phones see that (conversation, Bluetooth); wrist
devices mostly do not.

**Conscientiousness has a quieter lead**: more regular daily routines go with
higher C in both datasets (rho ~ 0.24), not significant after correction.
Round 1's Openness-and-bedtimes lead does not replicate.

**What would settle it:** GLOBEM (497 people, phone + Fitbit, needs the
PhysioNet credential). Pre-register H3 and H4 alone, a 1- or 2-feature model,
and run this same code on it; at n = 497 there is ample power for the effect
sizes seen here.

## 4. Plain-English notes on the three approaches

- **Pooling** means stacking two datasets into one bigger table so the model
  learns from 92 people instead of 46. The catch is that the datasets must
  measure the same things the same way, and their averages can differ for
  boring reasons (students vs office workers, phone vs watch). We handle that
  by ranking everyone *within their own dataset* first (z-scores), so a person
  is "more active than most people in their study", not "took 9,000 steps".
  Pooling asks: with more people, does the pattern become clear?
- **Transfer** is the hardest and most honest test: train on one dataset only,
  then predict people from a completely different study the model never saw.
  If a pattern is real and general, it should carry over. If it only works
  inside one dataset, it may be a quirk of that study (its sensors, its
  population, its season). Because the test people are wholly separate, the
  shuffled-label baseline is ~0 here, unlike cross-validation.
- **Meta-analysis** does not build a model at all. It looks at one simple
  relationship (say, activity timing vs extraversion), measures it separately
  in each dataset, then averages the two, giving more weight to the more
  precise one, and checks whether they disagree more than chance would allow
  (I^2; here they never did). It is how medicine combines several small trials
  into one more trustworthy answer.
- These can disagree, and here they do: the meta-analysis sees a consistent
  small link for one feature, while the models fail. That is normal with small
  samples. A model with eight inputs has many more ways to fit noise than a
  single correlation does, so it needs more people before it pays off.

## 5. Re-run after the averaging fix (2026-11-05)

`evaluate.predict` now averages r over the 5 CV repeats instead of correlating the averaged predictions
(PREREG P21, 1 Nov). Analyses (a), the within-dataset context and the exploratory one-feature model use it
and were re-run; (b), (c) and (d) do not and are unchanged. Before the fix the pooled model read O -0.196,
C -0.099, E -0.182 (p_perm 0.524), A -0.254, N -0.450; within StudentLife E -0.263, within LifeSnaps E +0.006;
the seed checks gave E -0.16/-0.17; the one-feature model r +0.14, p_perm 0.024. Negative scores moved towards
the null and E moved from 0.02 below its null to 0.03 above it. No verdict changes: the pooled E model still
fails its rule, and H3 (Holm p 0.053) is a Spearman meta-analysis that the fix does not touch.
The power row for the round-1 E model keeps the pre-fix r 0.26 written into the script before results; at the
re-run 0.24 it would need 135 people for 80% power at alpha .05 (198 at .01), with power 0.64 at n = 92.

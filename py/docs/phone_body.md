# phone + body: can sensors tell us who you are?

Run it: `cd py && uv run python scripts/run_phone_body.py` (about 45 s).
Code: `src/aas/slices/phone.py` (StudentLife), `src/aas/slices/body.py` (LifeSnaps).

## The data

| | StudentLife (phone) | LifeSnaps (body) |
|---|---|---|
| Who | Dartmouth students, spring 2013, ~10 weeks | Adults in GR/CY/IT/SE, May 2021 to early 2022 |
| Device | Android phone running a sensing app | Fitbit Sense smartwatch |
| Source used | Zenodo 3529253, `dataset_rds.zip` (230 MB, the R package `studentlife` mirror; official server unreachable) | Zenodo 7229547 (open v4; record 6832186 is now restricted), `rais_anonymized.zip` (615 MB) |
| Kept on disk | 87 MB: survey, EMA, other, 8 sensing files | 23 MB: daily + hourly CSVs, scored surveys, `surveys.bson` |
| Skipped | `audio.Rds` (266 MB), `wifi.Rds` (51 MB) | `fitbit.bson` (9.7 GB raw minute data) |
| Questionnaire | BFI-44, pre-term survey | IPIP 50-item Big-Five Factor Markers |
| N with labels and features | **46** | **52** (only 52 of the 71 took the IPIP) |

Both zips were deleted after extraction. Total disk: 110 MB.

### Scoring the questionnaires

- **BFI-44** (John & Srivastava key). Text answers map to 1..5, the 16 reverse-keyed
  items are scored 6 - x, a trait is the mean of its items. Check: every
  reverse-keyed item correlates negatively with its trait's forward items
  (all 16 do), Cronbach alpha E .85, A .79, C .86, N .80, O .71.
- **IPIP-50** (Goldberg key, ipip.ori.org order). Same 6 - x rule. The IPIP
  scores Emotional Stability, so we flip it to Neuroticism. Check: our sums match
  the authors' own `scored_surveys/personality.csv` for 50/50 people on all
  five traits.
- Then each trait is z-scored within its dataset (mean 0, SD 1).

## The features

Fixed before any model was scored. One set per dataset. "Per day" always
divides by the days that person's sensor actually produced data.

### StudentLife (14)

| Feature | What it is | Why it might track personality |
|---|---|---|
| log_gyration_km | median daily radius of gyration from GPS | E/O: range of movement, novelty seeking |
| places_per_day | distinct campus buildings per day (WiFi) | E/O: variety of settings |
| home_frac_day | share of 9-21h WiFi fixes in the dorm you sleep in | E-: staying in |
| conv_min_per_day | minutes of detected conversation per day | E: talking |
| conv_n_per_day | separate conversations per day | E |
| log_bt_devices_per_day | distinct Bluetooth devices nearby per day | E: people around you |
| active_frac | share of time walking/running (accelerometer) | E/C: energy |
| interdaily_stability | actigraphy IS: how alike each day's hourly activity pattern is | C: routine |
| sleep_h | length of the main night-time phone-in-darkness stretch | C/N |
| sleep_onset_h | when it starts (hours after 18:00) | C-: late nights |
| sleep_onset_sd | night-to-night spread of that start | C-/N: irregular bedtimes |
| locked_h_per_day | hours/day in 1h+ locked stretches | less = more phone use |
| night_locked_frac | share of 00:00-06:00 inside a 1h+ lock | low = phone at night |
| data_days | days the phone delivered data | C: sticking with a study app |

**Dropped on data quality before scoring** (not after looking at results):
calls per day (the call log is empty for 29 of 49 students, a logging failure),
SMS (no message times, only upload times), and two app-usage features (the
"app usage" table is a fixed 20-minute poll of the recent-tasks list; every
student's night share came out 0.25 = 6/24, i.e. it measures nothing). The
mirror only keeps lock/dark stretches of 1 h or more, so unlock counts are not
recoverable.

### LifeSnaps (15)

| Feature | What it is | Why it might track personality |
|---|---|---|
| log_steps | mean daily steps on worn days (>500 steps) | E/C: activity |
| active_min | fairly + very active minutes per day | E/C: exercise |
| vo2max | Fitbit cardio-fitness estimate | C: sustained exercise |
| steps_cv | day-to-day CV of steps | C-: irregular weeks |
| weekend_ratio | weekend / weekday steps | E: going out |
| interdaily_stability | IS on hourly steps | C: routine |
| activity_centroid_h | step-weighted mean hour | chronotype; C- for evening types |
| night_step_share | share of steps 00:00-05:00 | C-/N |
| sleep_min / sleep_sd | mean and SD of minutes asleep (nights >2 h) | C/N |
| sleep_eff | mean sleep efficiency | N- |
| resting_hr | mean resting heart rate | N, fitness |
| rmssd | mean nightly HRV (only 30 of 52 have it) | N-: low HRV and anxiety |
| log_wear_days | days the watch was worn | C: compliance |
| sema_days | days with at least one EMA answer (only HOW OFTEN, never WHAT) | C: compliance |

Self-reported EMA mood/place answers are deliberately excluded: they are
questionnaires, and mood is close to a neuroticism item.

## Results

`predict()`: ridge regression, 5-fold CV repeated 5 times, 200 label
permutations, 2000 bootstraps, Holm across the 5 traits of each dataset.
`null_mean` is what shuffled labels score: about -0.19 here, because with
n ~ 50 a model with no signal anti-predicts the held-out person. The tables
were re-run on 2026-11-05 with the per-repeat r of the 1 Nov fix (see the
correction at the end); the earlier numbers are in that note.

### StudentLife, 14 features, N = 46

| trait | r | 95% CI | null mean | p_perm | p_holm | survives |
|---|---|---|---|---|---|---|
| O | +0.316 | +0.10 .. +0.53 | -0.19 | 0.035 | 0.174 | no |
| C | +0.024 | -0.14 .. +0.19 | -0.19 | 0.149 | 0.448 | no |
| E | +0.240 | +0.02 .. +0.44 | -0.19 | 0.035 | 0.174 | no |
| A | -0.121 | -0.31 .. +0.01 | -0.21 | 0.303 | 0.607 | no |
| N | -0.292 | -0.40 .. -0.17 | -0.17 | 0.731 | 0.731 | no |

### LifeSnaps, 15 features, N = 52

| trait | r | 95% CI | null mean | p_perm | p_holm | survives |
|---|---|---|---|---|---|---|
| O | -0.192 | -0.38 .. -0.04 | -0.16 | 0.483 | 0.965 | no |
| C | +0.072 | -0.14 .. +0.28 | -0.20 | 0.080 | 0.318 | no |
| E | +0.241 | +0.01 .. +0.45 | -0.19 | 0.040 | 0.199 | no |
| A | -0.010 | -0.24 .. +0.21 | -0.18 | 0.144 | 0.433 | no |
| N | -0.220 | -0.40 .. -0.05 | -0.19 | 0.552 | 0.965 | no |

Holm across all 10 tests: the smallest adjusted p is 0.35. **No trait survives
in either dataset.**

Seed check (only to show stability, nothing was chosen from it): CV seeds 1
and 2 give StudentLife O +0.30/+0.30, E +0.22/+0.22; LifeSnaps E +0.23/+0.21.
The numbers are not a lucky split.

Why N in StudentLife is -0.29 (below most shuffles): in 92% of folds
RidgeCV chose the maximum penalty, i.e. it gave up and predicted the training
average, and "average of the others" runs backwards for the held-out person.
It means no usable signal, nothing more. Same story, milder, for LifeSnaps O
and N.

### Single-feature associations (Spearman, Holm across all tests in a dataset)

- StudentLife, 70 tests: 4 raw p < .05 (chance ~3.5). **One survives Holm:
  sleep_onset_sd vs Openness, rho = +0.59, n = 44** (students with more
  irregular bedtimes report higher openness). Next: sleep_h vs O -0.39, sleep_h
  vs C +0.38 (raw p ~.01, do not survive).
- LifeSnaps, 75 tests: 5 raw p < .05 (chance ~3.8), none survive. Top: sleep
  efficiency vs A +0.40, VO2max vs E -0.33, activity centroid vs E +0.31.
  LifeSnaps has no bedtime data in the CSVs, so it cannot test the StudentLife
  bedtime finding.

## Verdict

Nothing here clears the bar. Extraversion is the closest thing to a signal: it
came out at r ~ +0.24 in both datasets, from different devices and different
features, about 0.43 above the shuffled-label baseline, with raw p 0.035 and 0.040, which is consistent with the field's finding that extraversion is the
most visible trait in sensing data, but at n ~ 50 it does not survive multiple
testing. StudentLife Openness (r +0.32, driven by bedtime irregularity) is the
most interesting lead and the one single association that survives correction,
but it is one dataset, 44 students, and needs replication before anyone
believes it. Conscientiousness, agreeableness and neuroticism show nothing.

For comparison, Stachl et al. 2020 (PNAS, n = 624, 2 to 4 weeks of rich
Android logging) found extraversion the most predictable domain (r around
0.35) and the other domains much weaker. Our E point estimates (~0.24) sit a
little under that, which is about what you would expect with fewer, cruder
features; but our confidence intervals run from ~0 to ~0.5, so these datasets
are simply too small to confirm or refute that literature. With n ~ 50, a true
r of 0.3 is detected (p < .05, uncorrected) only about half the time.

## Correction, 2026-11-01: how r is averaged over repeats

`evaluate.predict` used to average the predictions of the 5 CV repeats and then compute r. With no signal, each repeat's "average of the others" bias lines up, so that pushed scores further below zero than they should be. It now computes r for each repeat and averages those. The tables above were re-run with the new code on 2026-11-05. Before the fix they read: StudentLife O +0.338, C +0.029, E +0.270, A -0.223, N -0.585; LifeSnaps O -0.227, C +0.134, E +0.255, A -0.012, N -0.306; smallest Holm p across all 10 was 0.30. The positive results barely move (StudentLife O .34 to .32, E .27 to .24; LifeSnaps E .26 to .24) and the strongly negative ones shrink towards the shuffled baseline (StudentLife N -.59 to -.29, A -.22 to -.12). LifeSnaps C falls from +.13 to +.07. No verdict changes; nothing survives Holm before or after.

# play: what your Steam profile says about your personality

Run: `cd py && uv run python scripts/run_play.py` (about 40 s).
Code: `src/aas/slices/play.py` (build), `scripts/run_play.py` (scoring).

## Data

- **Source.** Hao Lin and Leixiao Li, "A dataset on big five personality traits and user data on digital gaming social platforms", Mendeley Data v5 (2024-07-08), doi:10.17632/yyz69kysy3.5. Tianjin University of Technology.
- **Licence.** CC BY 4.0: we may use and change it as long as we credit the authors and say what we changed.
- **Files.** `Steam0706.csv` (67 KB) and `questionnaire.docx` (148 KB), saved in `data/raw/steam/` (216 KB in all). We skipped `web2.rar` (234 MB of profile-page screenshots). It would use most of our disk budget, and screenshots of homepages could identify people.
- **Paper.** The SSRN abstract (4687656) would not load for a script (bot wall). Everything below comes from the Mendeley description, the questionnaire, and checking the CSV directly.
- **Who.** 173 people answered a Chinese BFI-2 questionnaire (170 of them in China, 132 aged 18 to 25, 110 men and 61 women). Each person gave a link to their Steam profile, which had to be at least two years old. The authors then read eight numbers off each profile.

### What each column is

| Role | Columns | What we do with it |
|---|---|---|
| Identifiers | No., nickname, IP, homepage URL (all masked as `***`), avatar URL | dropped |
| Paradata | timestamp, seconds taken on the questionnaire | used only to exclude speeders |
| Demographics | gender, age band, country | not features (used once as a comparison, see D) |
| **Questionnaire** | BFI-2(1) to BFI-2(60) raw answers from 1 to 5; EXT, AGR, CON, NEG, OPE precomputed sums | **labels only, never features** |
| **Behaviour** | platform level, badges, games, screenshots, "evaluations" (reviews), groups, friends, banned 0/1 | features |

### Scoring the Big Five ourselves

The file ships raw answers *and* totals, so we scored the answers ourselves and checked them against the totals. We used the BFI-2 key (Soto & John 2017). The items cycle through E, A, C, N, O in that order, and 30 of the 60 are reverse-keyed (for example "tends to be quiet" counts against extraversion), so we score those as 6 minus the answer. Our sums match the shipped totals for 170 of 173 people. Without the reverse-keying only about 9% match, which confirms the key. The three that don't match:
- No. 89 has one blank answer. We take the mean of the 11 items they did answer, so their total isn't a whole number.
- Nos. 90 and 91: the shipped E, A and N totals don't follow from their own answers. They look miscopied, so we trust the answers.

Scores are item means, turned into z-scores within the kept sample.

### Exclusions (N)

| Step | People |
|---|---|
| Rows in file | 173 |
| Speeders: whole questionnaire in under 120 s, which is under 2 s per item (Nos. 72 and 163 took 84 to 85 s) | -2 → **171 with labels** |
| Profile not reachable: every behaviour blank, including level | -22 → **149 with features** |

The questionnaire has no attention-check items. No one gave the same answer to every item (no straightliners). One completion time was written as "5分" (5 minutes), which we read as 300 s.

### The blank-cell problem

Many cells say `Nan`. Of the 149: badges are blank for 17, games 82, screenshots 100, reviews 124, groups 110, friends 29. The smallest non-blank value in every column is 1, never 0. That fits how Steam works: a profile hides a section when its count is zero, so blank usually means zero. But blank games can also mean the player set their game details to private, and "0 games" is implausible for someone required to have used Steam for 2+ years. We can't tell the two apart, so we ran it both ways (sets A and B below).

## Features (14 in the main set)

Counts are heavily skewed: a few people have 700 badges and most have 5. So every count goes through `log(1 + x)`. Ratios are written as `log((a+1)/(b+1))`.

| Feature | Hypothesis, written before running |
|---|---|
| log_level | C/E: levelling up takes sustained effort, and it's a social display |
| log_badges | C: badges reward completing sets, a completionist habit |
| log_games | O: a bigger library means more variety-seeking |
| log_screenshots | O: capturing scenes suggests aesthetic interest |
| log_reviews | O/A: sharing opinions, helping others choose |
| log_groups | E: joining communities |
| log_friends | E: more friends |
| banned | low A, low C: cheating or rule-breaking (only 3 of 149, so near-useless) |
| screenshots_per_game | O: aesthetic attention per game |
| reviews_per_game | A/O: share of what you play that you bother to review |
| badges_per_level | C: how much of your level comes from badge-crafting |
| friends_per_level | E: sociability relative to how invested you are |
| groups_per_friend | O, or low E: interest communities rather than one-to-one ties |
| sections_shown | O/E: how many profile sections are non-empty |

## Results

`predict()` uses ridge regression with repeated 5-fold cross-validation (5 repeats), 200 permutations and 2000 bootstrap resamples. r is the correlation between predicted and true trait **for people the model did not train on**.

**A. Main (blank = 0, 14 features), N = 149**

| trait | r | 95% CI | p_perm | p_holm |
|---|---|---|---|---|
| O | -0.108 | -0.21 to 0.00 | 0.363 | 0.78 |
| C | 0.038 | -0.08 to 0.16 | 0.090 | 0.45 |
| E | 0.045 | -0.08 to 0.16 | 0.100 | 0.45 |
| A | -0.039 | -0.16 to 0.08 | 0.259 | 0.78 |
| N | -0.171 | -0.25 to -0.09 | 0.682 | 0.78 |

**B. Blank = unknown (filled with the training median) plus 6 was-blank flags, 20 features**

| trait | r | p_perm | p_holm |
|---|---|---|---|
| O | -0.131 | 0.512 | 1.00 |
| C | 0.086 | 0.050 | 0.25 |
| E | 0.139 | 0.050 | 0.25 |
| A | -0.044 | 0.264 | 0.79 |
| N | -0.169 | 0.687 | 1.00 |

**C. Only the 7 raw log counts**

| trait | r | p_perm | p_holm |
|---|---|---|---|
| O | -0.102 | 0.348 | 0.70 |
| C | 0.066 | 0.060 | 0.30 |
| E | 0.043 | 0.109 | 0.44 |
| A | 0.004 | 0.144 | 0.44 |
| N | -0.177 | 0.721 | 0.72 |

Across all 15 rows (3 sets x 5 traits) with Holm, the best are B/C and B/E at p_holm = 0.75. **Nothing survives.**

These tables were re-run on 2026-11-05 with the per-repeat r of the 1 Nov fix (correction at the end); the old numbers are listed there.

**D. Comparison only, not behaviour: gender and age band as the only 2 features.** O r = 0.20 (p_perm 0.015, p_holm 0.075), because women in this sample score higher on O (rho = -0.28 with being male). C, E and A come out at about -0.16 with p_perm 0.57 to 0.64, which is the artifact explained below. N r = -0.04.

### Why some r are strongly *negative*, and why that isn't a finding

When the features carry almost no signal, ridge picks the strongest shrinkage on offer (alpha = 10,000, the top of the grid). Each prediction is then about "the average of the people I trained on". Taking a person out of training pulls that average *away* from them, so held-out predictions run backwards. That pushes cross-validated r below zero. We measured it here: with **shuffled** labels the typical r is -0.12 (the middle 90% runs from -0.25 to +0.04). So judge r against this null, not against 0. That is what p_perm does. N at -0.17 is a little below the null: inside each fold the model learns a direction for N that flips on new people. That is "worse than guessing", not a hidden negative relationship. The bootstrap CI ignores this effect, so a CI that sits entirely below zero still doesn't mean anything here.

### Which behaviours relate to which traits (Spearman, 14 x 5 = 70 tests, Holm across all 70)

| feature | O | C | E | A | N |
|---|---|---|---|---|---|
| log_level | .12 | **-.20*** | **-.18*** | -.15 | .04 |
| log_badges | .14 | **-.22*** | **-.19*** | **-.17*** | .06 |
| log_games | .07 | -.13 | -.09 | -.08 | -.05 |
| log_screenshots | .02 | -.10 | **-.16*** | -.14 | -.06 |
| log_reviews | .00 | -.14 | -.10 | -.14 | -.03 |
| log_groups | .08 | -.07 | -.03 | -.01 | -.05 |
| log_friends | .10 | -.10 | -.03 | .00 | .00 |
| banned | -.07 | .02 | .06 | -.06 | -.08 |
| screenshots_per_game | -.07 | .06 | .02 | .03 | .00 |
| reviews_per_game | -.07 | .11 | .09 | .04 | .05 |
| badges_per_level | -.06 | -.10 | -.12 | -.14 | .07 |
| friends_per_level | .02 | .06 | **.18*** | .11 | .00 |
| groups_per_friend | -.07 | .09 | .04 | .01 | -.04 |
| sections_shown | .14 | -.06 | -.08 | -.04 | -.06 |

`*` = raw p < .05. **No cell survives Holm.** The smallest is badges vs C (rho = -0.22, raw p = .008, Holm p = .57). We got 7 of 70 under raw p < .05 when chance alone gives about 3.5.

Our hypotheses against the table:
- friends and groups predicting E: **no** (rho of .03 and .03).
- badges and level predicting C: the **opposite sign** (-.22 and -.20). Heavier Steam investment goes with *lower* self-rated C, E and A. Gender and age don't explain it (partial rho for badges vs C, controlling for both, is -0.23). It fits an "invests heavily in Steam, describes self less flatteringly" pattern, but it doesn't survive correction, so treat it as a lead to test on new data, not a result.
- friends_per_level vs E (+.18): right direction, but doesn't survive correction.

## Verdict

These eight Steam profile counts do not predict anyone's Big Five on unseen people. Every cross-validated r is between -0.18 and +0.14, nothing beats the shuffled-label null after correction, and the three feature sets agree. The meta-analytic level for digital footprints (Azucar et al. 2018, r ≈ 0.3 to 0.4 aggregated) is far above anything here. Those studies mostly use rich data such as likes, text and photos, with much larger samples. Eight coarse counts from 149 mostly young Chinese players is a much weaker probe. The one pattern worth chasing, that more badges and higher level go with lower self-rated C, E and A, is suggestive at best (raw p = .008, Holm p = .57).

## Changes to shared code that came out of this slice

- `evaluate.predict` now reports `null_mean`, the mean r of the permutation null (here about -0.12), so a negative r or a bootstrap CI below 0 is read against it.
- It also reports `max_penalty_share`, the share of folds where ridge chose the top of the alpha grid (1e4). Near 1 means the model gave up and predicted a constant, as for N here.

## Correction, 2026-11-01: how r is averaged over repeats

`evaluate.predict` used to average the predictions of the 5 CV repeats and then compute r. With no signal, each repeat's "average of the others" bias lines up, so that pushed scores further below zero than they should be. It now computes r for each repeat and averages those. The tables above were re-run with the new code on 2026-11-05. Before the fix, set A read O -0.162, C 0.031, E 0.051, A -0.051, N -0.351; set B O -0.221, C 0.089, E 0.149, A -0.058, N -0.350; set C O -0.150, C 0.061, E 0.052, A 0.006, N -0.348; the best Holm p over 15 rows was 0.52 (B/E). Positive results barely move and the strongly negative ones shrink towards the shuffled baseline (N about -.35 to -.17 in every set). No verdict changes; nothing survives Holm before or after.

# Fusion: one Big Five estimate from several sources

Code: `src/aas/fusion.py` (methods, `Fuser`, `FusedReader`). Simulation: `scripts/sim_fusion.py`
(`cd py && uv run python scripts/sim_fusion.py`, about 6 minutes, 20 seeds; every number
below is in `docs/fusion_sim.csv`). Tests: `tests/test_fusion.py`.

```python
from aas.fusion import FusedReader
from aas.readers.text import TextBigFiveReader, TextMBTIReader
bf, mb = TextBigFiveReader(), TextMBTIReader()
prof = build(person, [bf, mb, FusedReader([bf, mb])])     # "fused" sits beside the single readers
```

No real dataset has several sources per person together with their quiz yet. Everything here is
proved on simulation, built to be ready the day the study's data arrives.

## 1. What fusion is (plain English)

Each reader is a witness. The text reader saw how you write, the MBTI reader saw the same text from
another angle, music saw how you listen, sensors saw how you move. Each witness is only a little bit
right (correlation with your quiz score about 0.1 to 0.35). Fusion asks: if I hear all the witnesses
you have, what is my best single guess, and how sure should I be?

Three ideas do all the work.

**Weigh each witness by how often it has been right.** A source with r = 0.30 gets more say than one
with r = 0.10. The weight comes from each reader's own test on unseen people.

**Start from "average" and move only as far as the evidence pushes.** Before seeing anything, the
best guess for anyone is the population average (z = 0). Weak evidence moves you a little. That is
why the fused number for someone with only a music history barely leaves zero, and why that is the
honest answer, not a bug. (Combining sources *without* this starting point is a disaster: the plain
inverse-variance average of debiased sources, trained on 600 people, had MAE 6.7 times worse than
guessing average, skill -5.7. Unshrunk weak sources are wildly noisy.)

**Witnesses who copy each other count less.** If two witnesses saw the same thing (text-bigfive and
text-mbti both read the same posts), their mistakes move together. Hearing both is not twice the
evidence. Two sources of r = 0.30 each:

| how much their errors move together | fused r |
|---|---|
| 0.0 (independent) | 0.41 |
| 0.2 | 0.38 |
| 0.4 | 0.35 |
| 0.6 | 0.33 |
| 0.8 | 0.32 |
| 1.0 (same mistakes) | 0.30, no gain at all |

**What a conformal interval guarantees.** An 80% interval means: across people like the ones it was
calibrated on, the real quiz score lands inside it 80% of the time. Conformal prediction gets this
without trusting the model. It measures how wrong the model actually was on held-out training people,
and sizes the interval from those real misses. If the model is wrong, the intervals get wider, not
dishonest. The guarantee is on average over people (not for each person) and holds only if new
people resemble the training people (same kinds of data, same population). A posterior interval
(the Bayesian one) makes the same 80% promise *if the model's assumptions are right*; it needs no
training people, but it can be fooled when a source's stated accuracy is wrong.

## 2. The model

Every source is turned into one standardised score `u` per Big Five trait (mean 0, sd 1 in the
population) with a stated validity `v` (its correlation with the quiz on unseen people):

    quiz z ~ N(0, 1)                                  prior: quiz z is already relative to norms
    u_j    = v_j z + sqrt(1 - v_j^2) e_j              each source a noisy measurement
    corr(e_j, e_k) = R_jk                             errors that move together

The best guess given the sources a person has is `mean = v' S^-1 u`, with
`var = 1 - v' S^-1 v`, `S = v v' + D R D`, `D = diag(sqrt(1 - v^2))`. `sqrt(1 - var)` is the
accuracy the model expects for that person's set of sources.

Reader outputs to `u` (`readings_from_estimates`):

- **text-bigfive**: its value `m` is already a shrunk expected z whose spread is its r (that is how
  the reader builds it), so `u = m / r`.
- **text-mbti** (soft measurement): `P(first letter)` with letter accuracy AUC gives letter r
  `rho = sqrt 2 * sin(pi (AUC - 1/2) / 2)` (exact for bivariate normal scores against a median
  split; reproduces the text-bigfive card's AUC .682 -> r .40). Latent preference score
  `sqrt(1 - rho^2) probit(P)`, divided by `rho` to standardise. Then through the published
  MBTI x Big Five link: `u_E = -u_I`, validity `rho * 0.74`, and so on.
- **measured** (music): ignored unless a tested link to a Big Five trait is supplied
  (`measured_links`). None exists yet.
- An estimate with `usable=False` or no tested validity is never fused.

### MBTI to Big Five links (verified)

| MBTI axis | Big Five | McCrae & Costa 1989, men (women) | Reddit, PANDORA letters | used |
|---|---|---|---|---|
| E-I (first letter I) | E | -.74 (-.69) | -.75 | 0.74, sign - |
| S-N (first letter N) | O | +.72 (+.69) | +.43 | 0.72, sign + |
| T-F (first letter T) | A | -.44 (-.46) | -.55 | 0.44, sign - |
| J-P (first letter J) | C | +.49 (+.46) | +.58 | 0.49, sign + |

McCrae & Costa (1989), *Reinterpreting the Myers-Briggs Type Indicator from the perspective of the
five-factor model of personality*, J Pers 57(1):17-40, 267 men and 201 women, NEO-PI vs MBTI
continuous scores. The values -.74, .72, .44, -.49 are confirmed by a secondary source quoting the
paper (arXiv 2201.07670). That these are the men's column, and the women's values in brackets, come
from how the paper is commonly cited and were **not checked against the paper itself** (paywalled). Furnham (1996, PAID 21:303-307, n = 160) and Furnham, Moutafi & Crump (2003, SBP
31:577, n = 900) replicate the same four pairings and that Neuroticism has no MBTI partner; their
exact coefficients could not be read (paywalled). PANDORA (Gjurkovic et al. 2021, Table 3, Reddit
users' own MBTI letters vs Big Five percentiles) gives the right-hand column, read from the paper.
The task brief's "~0.7, ~0.7, ~0.4, ~0.5" matches McCrae & Costa. Two cautions: the N/S to O link
is weaker on Reddit (.43), and T/F also tracks Neuroticism there (-.34), which is not modelled.

So for one person, text-mbti's validity for E is letter r x 0.74. In-domain (AUC .787) that is
0.62 x 0.74 = 0.46. Measured out of domain (text-mbti run on the Essays, `readers_text.md`) the
I/E -> E AUC was .584, r 0.19. The truth on new people is somewhere between. That gap is why stated
validities are discounted (below).

### The methods

| | method | needs joint training data? | interval |
|---|---|---|---|
| a | `single`: best source the person has (highest stated, or training, validity) | no / yes | posterior (n = 0), CV+ conformal |
| b | `precision`: inverse-variance (GLS) of debiased sources, validities and error covariance estimated on training people, then the N(0,1) prior | yes | CV+ conformal |
| c | `stack`: ridge on source scores + missingness indicators; ridge penalty by LOO inside each outer fold (nested) | yes | CV+ conformal |
| d | `bayes`: the Gaussian model above from **stated validities only**, error correlation 0.3 within a modality and 0 across, validity uncertain (Fisher-z sd 0.10, 0.15 for MBTI) and integrated over, validities discounted x0.75 | **no** | posterior |
| d+ | `bayes_eb`: same, with validities, error correlations and score scales updated from training people, shrunk toward the stated values | yes (any n) | posterior |
| d+ | `bayes_eb_cp`: `bayes_eb` inside CV+ conformal | yes | CV+ conformal |

CV+ (Barber, Candes, Ramdas & Tibshirani 2021, Ann Stat 49(1)) rather than split conformal because at
n = 60 holding out a calibration set wastes too much. Scores are normalised by each person's
model-implied sd, so a person with three sources gets a narrower interval than one with music only,
while the coverage guarantee still holds on average. Guaranteed coverage is at least 1 - 2a (60% for
an 80% interval) in the worst case and about 1 - a in practice.

## 3. Simulation

`scripts/sim_fusion.py`. Per seed: training people (n_train = 0, 60, 250, 600) and 3,000 test people.
20 seeds. Results averaged over the five traits, then mean (sd) over seeds. Scored on test people
with at least one source; people with none get no number from any method.

- **Truth**: quiz z for all five traits, with realistic intercorrelations (N with C, A, E about -.25
  to -.30).
- **Sources and true validity** (O C E A N): text .32 .20 .25 .20 .22; MBTI-derived (letter r 0.40 x
  link) .29 .20 .30 .18 none; music .12 .05 .10 .08 .08; sensors .03 .12 .15 .00 .05.
- **Correlated errors**: text with MBTI 0.40 (same text), music with sensors 0.15, text with music
  0.10, text with sensors 0.05.
- **Missingness**: text 65% (85% of those also enough for MBTI), music 45%, sensors 30%,
  independent. Result: 13% have nothing, 43% one source, 33% two, 11% three or more.
- **Scenarios** for what the readers *state* versus the truth:
  - *honest*: stated = true.
  - *optimistic*: stated = what our readers report in their own datasets, higher than on new data
    (text x1.35, MBTI from the in-domain AUC .787, music x1.5, sensors E .26 / C .20 from round 1), and
    every reader's scale is a little off (mean shift up to 0.15, sd 1.10 to 1.25).
  - *hostile*: stated as in optimistic, but on new data text works at half the value above, MBTI
    letters at r 0.20, music not at all.

"Skill" is 1 - MAE / MAE(guess average). At these validities the best possible skill is small: a
perfect r = 0.25 predictor has skill about 1 - sqrt(1 - 0.25^2) = 0.03. Read r and coverage first.

### All people, mean over traits; mean (sd over 20 seeds)

**honest**

| n_train | method | r with truth | skill | 80% coverage | width (z) |
|---|---|---|---|---|---|
| 0 | single | 0.218 (0.007) | +0.023 (0.001) | 0.805 (0.002) | 2.53 (0.00) |
| 0 | bayes, no discount | 0.240 (0.007) | +0.030 (0.002) | 0.799 (0.002) | 2.48 (0.00) |
| 0 | **bayes** | **0.240** (0.007) | +0.029 (0.002) | 0.805 (0.002) | 2.51 (0.00) |
| 60 | single | 0.159 (0.028) | -0.023 (0.018) | 0.818 (0.025) | 2.76 (0.16) |
| 60 | precision | 0.187 (0.025) | -0.004 (0.012) | 0.813 (0.030) | 2.68 (0.17) |
| 60 | stack | 0.177 (0.035) | -0.009 (0.020) | 0.816 (0.029) | 2.71 (0.18) |
| 60 | bayes | 0.240 (0.009) | +0.029 (0.002) | 0.805 (0.004) | 2.51 (0.00) |
| 60 | bayes_eb | 0.231 (0.010) | +0.027 (0.002) | 0.802 (0.005) | 2.50 (0.02) |
| 60 | bayes_eb_cp | 0.231 (0.011) | +0.027 (0.002) | 0.818 (0.030) | 2.62 (0.18) |
| 250 | single | 0.197 (0.014) | +0.015 (0.004) | 0.798 (0.013) | 2.53 (0.08) |
| 250 | precision | 0.222 (0.013) | +0.023 (0.004) | 0.797 (0.015) | 2.50 (0.09) |
| 250 | stack | 0.216 (0.014) | +0.020 (0.004) | 0.799 (0.012) | 2.52 (0.08) |
| 250 | bayes | 0.240 (0.008) | +0.029 (0.002) | 0.804 (0.004) | 2.51 (0.00) |
| 250 | bayes_eb | 0.231 (0.010) | +0.027 (0.002) | 0.799 (0.004) | 2.49 (0.01) |
| 250 | bayes_eb_cp | 0.231 (0.010) | +0.027 (0.002) | 0.798 (0.014) | 2.49 (0.09) |
| 600 | single | 0.208 (0.011) | +0.021 (0.003) | 0.800 (0.006) | 2.51 (0.04) |
| 600 | precision | 0.233 (0.007) | +0.028 (0.002) | 0.798 (0.008) | 2.49 (0.04) |
| 600 | stack | 0.228 (0.007) | +0.025 (0.002) | 0.798 (0.007) | 2.49 (0.04) |
| 600 | bayes | 0.242 (0.006) | +0.030 (0.002) | 0.804 (0.003) | 2.51 (0.00) |
| 600 | bayes_eb | 0.236 (0.006) | +0.029 (0.002) | 0.798 (0.003) | 2.48 (0.01) |
| 600 | bayes_eb_cp | 0.236 (0.006) | +0.029 (0.002) | 0.798 (0.007) | 2.48 (0.04) |

**optimistic** (stated validities too high, readers slightly mis-scaled)

| n_train | method | r | skill | coverage | width |
|---|---|---|---|---|---|
| 0 | single | 0.212 (0.005) | +0.022 (0.002) | 0.797 (0.003) | 2.49 |
| 0 | bayes, no discount | 0.240 (0.006) | +0.016 (0.004) | **0.777** (0.004) | 2.40 |
| 0 | **bayes** | **0.240** (0.006) | +0.028 (0.003) | 0.797 (0.003) | 2.47 |
| 60 | stack | 0.174 (0.039) | -0.009 (0.017) | 0.808 (0.029) | 2.66 |
| 60 | bayes_eb | 0.229 (0.007) | +0.025 (0.004) | 0.794 (0.005) | 2.47 |
| 250 | precision | 0.217 (0.013) | +0.023 (0.003) | 0.796 (0.010) | 2.50 |
| 250 | stack | 0.213 (0.013) | +0.019 (0.003) | 0.797 (0.009) | 2.51 |
| 250 | bayes_eb_cp | 0.231 (0.008) | +0.027 (0.003) | 0.795 (0.012) | 2.48 |
| 600 | precision | 0.234 (0.010) | +0.027 (0.003) | 0.798 (0.006) | 2.49 |
| 600 | stack | 0.229 (0.010) | +0.025 (0.003) | 0.800 (0.006) | 2.50 |
| 600 | bayes | 0.240 (0.011) | +0.027 (0.004) | 0.795 (0.003) | 2.47 |
| 600 | bayes_eb_cp | 0.237 (0.010) | +0.028 (0.003) | 0.798 (0.006) | 2.49 |

**hostile** (stated validities 2x the truth or more, music worthless)

| n_train | method | r | skill | coverage | width |
|---|---|---|---|---|---|
| 0 | single | 0.105 (0.008) | -0.006 (0.002) | 0.784 (0.003) | 2.49 |
| 0 | bayes, no discount | 0.120 (0.007) | -0.031 (0.003) | **0.755** (0.004) | 2.40 |
| 0 | bayes | 0.120 (0.007) | -0.010 (0.002) | 0.779 (0.004) | 2.47 |
| 60 | stack | 0.061 (0.024) | -0.014 (0.011) | 0.806 (0.030) | 2.66 |
| 60 | bayes_eb | 0.117 (0.008) | -0.004 (0.004) | 0.787 (0.004) | 2.49 |
| 250 | stack | 0.093 (0.016) | +0.001 (0.003) | 0.802 (0.015) | 2.58 |
| 250 | bayes_eb | 0.117 (0.010) | +0.004 (0.002) | 0.795 (0.005) | 2.52 |
| 250 | bayes_eb_cp | 0.117 (0.010) | +0.004 (0.002) | 0.801 (0.014) | 2.56 |
| 600 | stack | 0.107 (0.014) | +0.004 (0.002) | 0.798 (0.008) | 2.55 |
| 600 | bayes | 0.116 (0.009) | -0.013 (0.004) | 0.777 (0.005) | 2.47 |
| 600 | bayes_eb_cp | 0.116 (0.010) | +0.005 (0.002) | 0.797 (0.009) | 2.54 |

### Gain from adding sources

By how many sources a test person has (mean over traits and seeds):

| scenario | n_train | method | 1 source r / cov | 2 sources r / cov | 3+ sources r / cov |
|---|---|---|---|---|---|
| honest | 0 | single | .140 / .803 | .230 / .807 | .246 / .805 |
| honest | 0 | bayes | .140 / .803 | .249 / .808 | .278 / .804 |
| honest | 60 | stack | .082 / .816 | .189 / .818 | .217 / .817 |
| honest | 600 | stack | .125 / .800 | .240 / .796 | .268 / .797 |
| honest | 600 | bayes_eb_cp | .139 / .800 | .248 / .797 | .270 / .798 |
| hostile | 0 | bayes | .072 / .793 | .129 / .779 | .137 / **.771** |
| hostile | 600 | bayes_eb_cp | .061 / .797 | .115 / .799 | .140 / .796 |

Best possible fusion (true parameters, population r) for nested source sets:

| sources | O | C | E | A | N | E if errors were independent |
|---|---|---|---|---|---|---|
| text | .320 | .200 | .250 | .200 | .220 | .250 |
| + MBTI letters | .358 | .235 | .324 | .224 | .220 | .374 |
| + music | .364 | .236 | .329 | .230 | .227 | .384 |
| + sensors | .364 | .255 | .347 | .231 | .229 | .405 |
| music + sensors (no text) | .121 | .124 | .168 | .081 | .089 | .179 |

Adding MBTI letters to text is the biggest step (E .25 to .32) because the MBTI link for E is strong.
Its gain is cut by about half by the shared-text error correlation (it would be .37 if independent).
Music adds almost nothing on top of text. Sensors help E and C a little.

### What the simulation says

1. **The zero-data Bayesian model is as good as anything trained on 600 people** in the honest and
   optimistic worlds (r .240 vs .233 to .236; coverage .80). Trained methods spend their data
   relearning weights the readers already state.
2. **Trained methods are worse than guessing average at n = 60** (stack, precision and single all
   have negative skill, r .16 to .19, intervals about 8% wider and noisy). Do not fit a stack on a pilot.
3. **The discount matters for intervals, not ranking.** r is identical with and without it. Without
   it, optimistic stated validities make the 80% intervals cover 77.7% (hostile: 75.5%); with the
   x0.75 discount, 79.7% (hostile 77.9%).
4. **Training data's real value is fixing calibration when the stated validities are badly wrong**
   (hostile): bayes_eb at n = 250 turns skill from -0.011 to +0.004 and coverage from .78 to .80.
   It cannot recover ranking accuracy that the sources do not have.
5. **Conformal coverage held at .80 for every trained method from n = 250 on**, in every scenario,
   including the hostile one. At n = 60 it is conservative (.81 to .82) and the width varies a lot
   between training sets (sd 0.17 z).
6. **Stacking never won.** It is within 0.01 of bayes_eb at n = 600 and is the method most hurt by
   small n. It earns its place only if sources interact non-linearly or are much stronger than now.

## 4. Decision

| data | method | why |
|---|---|---|
| **pre-study, n = 0** | `bayes` (default in `FusedReader`): stated validities x0.75, Fisher-z uncertainty, error corr 0.3 within a modality | the only option with no joint data, and in simulation no worse than trained methods at n = 600 unless stated validities are badly wrong |
| **pilot, n about 60 to 150** | keep `bayes` for what people see; fit `bayes_eb` on the pilot and report both | use the pilot to *check*, not to fit: measured coverage of the pilot (pooled across traits), each source's validity on the pilot vs stated, the text-to-MBTI error correlation. Switch to `bayes_eb` if the pilot validities fall clearly below the stated ones (the hostile pattern) |
| **main study, n >= 250** | `bayes_eb_cp` | same point estimates as bayes_eb, plus conformal intervals that stay at 80% even if the model is wrong |

`FusedReader` leaves `validity=None` on its estimates: `Estimate.validity` means *tested* accuracy and
no fusion has been tested on people with several sources. The model-implied r goes in `evidence`.

## 5. What this means for the study's sample size (for POWER.md)

Effects from this simulation (true parameters as above, which are assumptions, not measurements):
fused r for a person with all four sources is .36 O, .26 C, .35 E, .23 A, .23 N (mean about .29);
text alone .32, .20, .25, .20, .22 (mean about .24).

| question | n needed (80% power, alpha .05 two-sided unless stated) |
|---|---|
| fused r > 0 for one trait, r = .35 / .30 / .25 / .20 | 62 / 85 / 124 / 194 (alpha .01: 91 / 125 / 183 / 288) |
| fused beats text alone, E (.347 vs .25, Williams t, dependent correlations, everyone has all sources) | about 420 |
| same with text + MBTI only (.324 vs .25) | about 595 |
| same for O (.364 vs .32) / C (.255 vs .20) / an average trait (.29 vs .23) | about 860 / 1,055 / 835 |
| check 80% coverage to +-0.05 (95% CI) for one trait | 246 (+-0.08: 97; +-0.03: 683) |
| estimate one source's validity: 95% CI for r = .25 | n 60: [.00, .47]; 100: [.06, .43]; 250: [.13, .36]; 600: [.17, .32] |
| trained fusion stops being worse than guessing (simulation) | about 250 people (n_train counts everyone; 13% of them had no source) |
| trained fusion matches the zero-data model (simulation) | about 600, and only matches it |

Design implications:
- **Collect every source from every participant.** Missingness can be simulated later by masking
  sources, but cannot be undone. The "fused beats single" tests above assume everyone has all sources;
  with the realistic 1-to-2-source mix, multiply by about 2 to 3.
- **Proving fusion beats text alone is the expensive claim** (400 to 1,000). Proving fused
  predictions carry signal (r > 0) and have honest intervals is cheap (about 100 to 250).
- **The pilot's job is checking, not fitting.** 60 to 100 people with all sources and the quiz can
  confirm the stated validities are not wildly off and coverage is roughly right (pooled over five
  traits, about 300 to 500 trait-checks), which is what decides whether the zero-data model ships.

## 5b. Measured validities, 2026-11-01

Everything above (sections 3 to 5) used validities **assumed** before any cross-domain test. Those results are
kept as they were and reproduce exactly. This section reruns the same simulation with the validities
**measured** on writing the readers never saw (`text_crossdomain.md`, `readers_text.md` "Validity by kind of
writing", cards' `validity_by_source`). Same code, same seeds: `scripts/sim_fusion.py` now has three more
scenarios; all rows are in `docs/fusion_sim.csv`.

### What changed in the inputs

| source | assumed (sections 3 to 5), r for O C E A N | measured, for a Reddit or X user |
|---|---|---|
| text-bigfive | .32 .20 .25 .20 .22 | **refused** .097 .095 .094 .158 (closed-form `auc_to_r` of the card's tweets AUC .5438 .5427 .5423 .5713; Reddit uses the same numbers until PANDORA) |
| text-mbti, via the links | .29 .20 .30 .18 none (letter r 0.40) | .20 .13 .19 .12 none (letter r about 0.26, **inferred**, untested off the forum) |
| music, sensors | as before | as before (nothing new measured) |

Openness has no text-bigfive source at all now (the reader refuses it), so for O the only text signal is the
MBTI N/S letter, whose off-forum number is itself an inference.

New scenarios:

- **measured**: the readers state the measured numbers and the truth is the measured numbers (what the readers
  now do, if the measurements are right).
- **measured_old_stated**: the truth is the measured numbers (text O truly 0), but the readers state what they
  stated before today (text .40 .16 .21 .19 .19, MBTI from the forum AUC), with the slightly-off scales of the
  optimistic scenario. This is what the shipped readers did yesterday on real users.
- **measured_hostile**: readers state the measured numbers, but the truth is at the low end of their CIs (text at
  half, MBTI letters at r 0.13, music worthless).

### Results: all people with at least one source, mean over traits, mean (sd over 20 seeds)

| scenario | n_train | method | r with truth | skill | 80% coverage | width (z) |
|---|---|---|---|---|---|---|
| assumed, honest (before) | 0 | bayes | 0.240 (0.007) | +0.029 | 0.805 | 2.51 |
| **measured** | 0 | single | 0.140 (0.007) | +0.010 | 0.802 | 2.55 |
| **measured** | 0 | **bayes** | **0.150** (0.007) | +0.011 | **0.802** | 2.54 |
| measured | 0 | bayes, no discount | 0.150 (0.007) | +0.012 | 0.800 | 2.53 |
| measured | 60 | stack | 0.079 (0.029) | -0.010 | 0.812 | 2.70 |
| measured | 60 | bayes_eb | 0.138 (0.011) | +0.009 | 0.800 | 2.54 |
| measured | 250 | stack | 0.117 (0.017) | +0.002 | 0.801 | 2.57 |
| measured | 250 | bayes_eb_cp | 0.138 (0.013) | +0.009 | 0.801 | 2.55 |
| measured | 600 | precision | 0.138 (0.011) | +0.009 | 0.799 | 2.54 |
| measured | 600 | stack | 0.129 (0.010) | +0.007 | 0.800 | 2.55 |
| measured | 600 | bayes_eb_cp | 0.139 (0.009) | +0.010 | 0.799 | 2.54 |
| **measured_old_stated** | 0 | bayes | 0.137 (0.008) | **-0.003** | **0.785** | 2.49 |
| measured_old_stated | 0 | bayes, no discount | 0.137 (0.008) | -0.018 | **0.765** | 2.43 |
| measured_old_stated | 250 | bayes_eb_cp | 0.138 (0.011) | +0.007 | 0.802 | 2.57 |
| measured_old_stated | 600 | bayes_eb_cp | 0.144 (0.010) | +0.010 | 0.800 | 2.55 |
| measured_hostile | 0 | bayes | 0.071 (0.008) | +0.001 | 0.795 | 2.54 |
| measured_hostile | 0 | bayes, no discount | 0.071 (0.008) | -0.001 | 0.792 | 2.53 |
| measured_hostile | 250 | bayes_eb_cp | 0.057 (0.012) | -0.000 | 0.803 | 2.59 |
| measured_hostile | 600 | bayes_eb_cp | 0.071 (0.009) | +0.002 | 0.800 | 2.56 |

Per trait, measured, bayes at n = 0: r .18 O, .13 C, .18 E, .11 A, .15 N; coverage .80 on every trait. With the
old stated validities the per-trait coverage was .76 for O (.72 without the discount), .78 for E, and .79 to .80
for the rest: the over-claimed openness reading was the worst offender.

By number of sources (measured, bayes, n = 0): 1 source r .11, 2 sources .16, 3+ sources .18; coverage .80 in
every group. Best possible fusion with measured parameters, text + MBTI: O .20 (MBTI only), C .14, E .19, A .13,
N .16; all four sources: .22, .18, .24, .14, .17.

### What changed

1. **Fused r falls from about 0.24 to about 0.15** (mean over traits, zero-data model). Text alone for C, E, A is
   about 0.1; N is the only text trait above 0.15. Most of what fusion adds now comes from the MBTI letters, whose
   off-forum accuracy is inferred, not measured. If that inference is too generous (measured_hostile), fused r is
   about 0.07.
2. **Coverage is right again.** With the readers stating measured validities, the zero-data intervals cover
   80.2% (every trait .80). With yesterday's stated validities on the same people they covered 78.5% with the
   x0.75 discount and 76.5% without it, and only 76% for openness. Reporting validity by kind of writing fixes
   the calibration that the discount only partly fixed.
3. **The x0.75 discount is now nearly irrelevant, not harmful.** With honest stated validities it widens intervals by
   0.01 z and costs nothing else (r identical, coverage .802 vs .800). It is kept: it is cheap insurance against the
   measured_hostile case (coverage .795 with it, .792 without). It should not be raised.
4. **The n = 0 vs trained decision is unchanged, and stronger.** Bayes at n = 0 (r .150) beats every trained
   method at every n up to 600 (best trained: .139 to .144). Stack and single are worse than guessing average at
   n = 60 (skill -0.010 to -0.027). The only thing training data buys is calibration repair when stated
   validities are wrong: in measured_old_stated, bayes_eb_cp at n = 250 brings coverage from .785 back to .802 and
   skill from -0.003 to +0.007. Since the readers now state measured validities, that repair is mostly done
   without data.
5. **Intervals are now almost the population interval.** Width 2.54 z against 2.56 for "anyone" (+-1.28 z). The
   fused number moves people very little from average. That is honest, and it means a user will see a z close to 0
   with a wide range for most traits.

Decision table (section 4) stands: `bayes` with no joint data; pilot to check; `bayes_eb_cp` at n >= 250. What
changes is the expectation: the pilot's main check (stated vs real validity) now compares measured cross-domain
numbers, and the study's power to see anything at all is lower (POWER.md "Measured validities, 2026-11-01").

Effects for POWER.md, measured: fused text + MBTI r .20 O, .14 C, .19 E, .13 A, .16 N (mean about 0.16). n for
fused r > 0 at 80% power: 200 O, 429 C, 223 E, 493 A, 313 N. Text alone, C, E, A: about 830 to 890 each; N 313.

## 5c. v1 defaults, 2026-11-05

The measured scenario above still had hypothetical music and sensor sources. v1 fuses only what its default
readers give: `text-bigfive` and `text-mbti`. Scenario `v1` in `scripts/sim_fusion.py` drops music and sensors and
reads the text validities from the cards; `v1_distilled` is a sensitivity row in which the distilled reader, which
failed its pre-registered rule (`distill.md`), replaces `text-bigfive` at its card's tweets AUCs (it refuses O and C
there). Rows are in `docs/fusion_sim_v1.csv`; the derivation and caveats are in `MATH.md` section 5.6.

| scenario | O | C | E | A | N | mean r | coverage | width (z) |
|---|---|---|---|---|---|---|---|---|
| v1, bayes, n = 0 | .196 | .146 | .172 | .121 | .148 | .157 | .802 | 2.54 |
| v1_distilled (sensitivity) | .193 | .124 | .219 | .131 | .251 | .184 | .802 | 2.54 |

Everyone scored here has text, so the mean is slightly above the .150 of section 5b, which also scored people with
only music or sensors. Coverage is .80 on every trait in both rows.

## 6. Limits

- The simulation's validities, error correlations and missingness rates are assumptions informed by
  our readers' tests, not measurements on joint data. The study replaces them. Since 2026-11-01 the text
  validities in the measured scenarios come from held-out-domain tests (tweets; Reddit not yet measured), and
  the MBTI ones are inferred from a cross-construct test (section 5b).
- The quiz score is the target. Quiz unreliability is not modelled separately (the readers' stated
  validities are already against questionnaires).
- Traits are fused one at a time with an independent N(0,1) prior. A joint prior using the Big Five
  intercorrelations (N with C, A, E about -.25) would let, say, an E reading inform N a little. Small
  gain, not done.
- MBTI secondary links (T/F with N, -.34 on Reddit) are ignored.
- Music has no tested link to the Big Five; `measured_links` is empty by default, so the real
  `FusedReader` fuses text-bigfive and text-mbti only until one is measured.
- A person's text validity varies with their word count; the zero-data model uses each person's own
  stated validity, the trained methods use the source as a whole.

## 7. Changes outside this module

Two of the three changes proposed here are in place:

- `profile.py` derives `MBTI_TO_BIGFIVE` from `fusion.MBTI_LINKS`, so the link strengths live in one place.
- `profile.dump()` prints "several readers, shown separately (the fused reader, if run, is the combination)"
  when a trait has more than one estimate, and the `profile` docstring says the `fused` reader is the only
  combination.

Still proposed, not made: an optional `validity_tested: bool` on `person.Estimate` (or `validity_kind =
"implied"`), so the fused reader could report its model-implied accuracy in a field instead of only in
`evidence`.

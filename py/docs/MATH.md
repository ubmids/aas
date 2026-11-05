# AAS engine v1: the maths

Version 1, 5 Nov 2026. This document states what the Python engine computes, as the code does it, and puts each measured number next to the method that produced it. Paths are relative to `py/`, and every formula names the file and function it comes from. Numbers are copied from the docs and model cards named beside them. Where a number has not been measured, the section says so.

The listening work (taste maps, skip model, the game, selves, eras, new loves, doors, stick model, measured traits, the 304-listener ListenBrainz round) has its own document from the TypeScript engine: [`../../docs/MATH.md`](../../docs/MATH.md). It is summarised in section 3.8 and not repeated.

Throughout, $z$ is a Big Five trait score standardised against the IPIP norms (mean 0, sd 1 in the norm population). $r$ is Pearson's correlation between a prediction and the questionnaire score on people the model never saw. AUC is the area under the ROC curve for a yes/no label (0.5 is a coin flip, 1 is perfect). $\Phi$ is the standard normal distribution function, $\Phi^{-1}$ its inverse (the probit), $\sigma(x) = 1/(1+e^{-x})$ the logistic function, and $\operatorname{logit} p = \log\frac{p}{1-p}$.

## 1. The question and the ruler

The question is how well a person's own digital traces (what they write, what they listen to, how their phone or watch moves) predict their Big Five personality, and how sure each prediction should be. Every answer is checked against one ruler: the person's own answers to the IPIP-50 questionnaire, scored against a large norm sample. If the ruler is noisy, every validity measured with it is capped, so the ruler's own accuracy comes first.

### 1.1 Scoring the IPIP-50

The 50 items are Goldberg's Big-Five Factor Markers, each answered 1 to 5. An item is either forward keyed (agreeing means more of the trait) or reverse keyed. A reverse-keyed answer $x$ is replaced by $6-x$, and a trait score is the mean of its ten keyed answers (`src/aas/quiz.py:keyed`, `scales`):

$$s_t = \frac{1}{|I_t|}\sum_{i \in I_t} \tilde x_i, \qquad \tilde x_i = \begin{cases} x_i & \text{forward} \\ 6 - x_i & \text{reverse} \end{cases}$$

The dataset's EST scale measures emotional stability, so neuroticism N is EST with every item's direction flipped (`quiz.py:_build_key`). The key was checked by correlating each raw item with the keyed total of the other nine on its scale: all 21 reverse items come out negative raw, and all 50 positive after keying (`docs/quiz.md`).

### 1.2 Norms, z and percentiles

The norm sample is the Open-Source Psychometrics IPIP-FFM dump (1,015,341 rows, 2016 to 2018), cleaned in a fixed order: missing or skipped answers, duplicate IP addresses, missing timing, median time per item under 1 s, the same answer to all 50 items, and 15 or more identical answers in a row in page order (`quiz.py:clean`). 597,365 people remain (58.8%). Means and sds are population values ($\text{ddof}=0$) on the 1 to 5 scale (`quiz.py:make_norms`):

| | O | C | E | A | N |
|---|---:|---:|---:|---:|---:|
| mean | 3.94 | 3.34 | 2.91 | 3.76 | 3.08 |
| sd | 0.62 | 0.74 | 0.91 | 0.74 | 0.86 |

One person's score becomes a z in one of three ways (`quiz.py:score`). With all ten items (form `full`) or exactly the short-form items (form `short`), $z = (s_t - \mu_t)/\sigma_t$ with that form's mean and sd, and the percentile is read from an exact table of every possible raw sum $S$, using the mid-rank rule

$$\text{pct}(S) = 100 \cdot \frac{\#\{S' < S\} + \tfrac12\#\{S' = S\}}{n}$$

(`quiz.py:_percentile_table`). With any other subset $J$ of items (form `partial`), the mean and sd come from the stored item means $\mu_i$ and keyed item covariance $\Sigma$:

$$\mu_J = \frac{1}{|J|}\sum_{i\in J}\mu_i, \qquad \sigma_J = \frac{\sqrt{\mathbf 1^\top \Sigma_{JJ}\mathbf 1}}{|J|}, \qquad \text{pct} = 100\,\Phi(z).$$

Because the norms describe people who describe themselves as open and agreeable, answering 3 to everything lands at z $-1.53$ on O and $-1.04$ on A (`docs/quiz.md`). Neutral answers are below average on those traits.

### 1.3 Reliability

Internal consistency is Cronbach's alpha over the $k$ keyed items of a scale (`quiz.py:cronbach_alpha`):

$$\alpha = \frac{k}{k-1}\left(1 - \frac{\sum_i \operatorname{Var}(\tilde x_i)}{\operatorname{Var}\big(\sum_i \tilde x_i\big)}\right).$$

Full scales: O 0.80, C 0.82, E 0.90, A 0.84, N 0.87. A five-factor varimax solution on 50,000 cleaned people puts all 50 items on their own trait's factor, with own-factor loadings 0.37 to 0.80 (`docs/quiz.md`). Two-week retest reliability is a study outcome (PREREG S5) and has not been measured.

### 1.4 The short form and the ceiling it sets

The in-app quiz uses 4 items per trait. Items were ranked on a random training half (299k people) by corrected item-total correlation, each item against the sum of the other items of its scale (`quiz.py:corrected_item_total`), and the top 4 kept, with at least one forward and one reverse item per trait so that agreeing with everything does not score high on everything (`quiz.py:choose_short_form`). On the held-out half (298k) the 4-item score correlates with the 10-item score at O 0.91, C 0.92, E 0.95, A 0.92, N 0.94. These are part-whole correlations. Against only the six items left out, the 4-item score correlates 0.72, 0.72, 0.83, 0.74, 0.81.

This sets a ceiling. A model that predicted the full-scale score perfectly would still correlate only about 0.91 to 0.95 with the 4-item quiz, so validities checked against the short quiz read about 5 to 10% lower. Validation work therefore uses the full 50 items; 20 items serve the in-app "how close were we" check.

## 2. The referee

One function scores every claim that a set of features predicts a trait: `src/aas/evaluate.py:predict`. Its rule is that a number only counts if it was measured on people the model never saw, and that it is read against what pure noise scores under the same procedure.

### 2.1 Cross-validated r

The model is a pipeline of median imputation, standardisation and ridge regression (`evaluate.py:ridge`). All three steps are fitted inside each training fold, so test people never shape them. Ridge solves

$$\hat\beta_\alpha = \arg\min_\beta \|y - X\beta\|^2 + \alpha\|\beta\|^2,$$

with $\alpha$ chosen by scikit-learn's efficient leave-one-out over 25 values from $10^{-2}$ to $10^{4}$ (`evaluate.py:ALPHAS`).

For each trait, people are split into $K = 5$ folds, and this is repeated $R = 5$ times with different shuffles (seeds $s, s+1, \dots$). Repeat $\rho$ gives an out-of-fold prediction $\hat y^{(\rho)}_i$ for every person (`evaluate.py:_oof`). The reported score is the mean of the per-repeat correlations:

$$\bar r = \frac{1}{R}\sum_{\rho=1}^{R}\operatorname{corr}\big(\hat y^{(\rho)}, y\big).$$

Traits with fewer than 15 labelled people are skipped. Until 1 Nov 2026 the code averaged the predictions over repeats and then took one correlation. With no signal, each repeat's bias lines up, so averaging first pushes $r$ towards $-1$ as repeats are added. The per-repeat average (PREREG P21) removes that. On the slices already run, positive results barely moved and strongly negative ones moved towards the shuffled baseline; no verdict changed (`docs/phone_body.md`, correction note).

### 2.2 Why noise scores below zero: null_mean

When features carry nothing, ridge picks the strongest penalty and predicts roughly the mean of the training people. Removing person $i$ from the training set moves that mean away from $y_i$, so held-out predictions run backwards and cross-validated $r$ for pure noise is negative. At $n \approx 50$ it is about $-0.19$; at $n \approx 150$ about $-0.1$ to $-0.12$ (`docs/phone_body.md`, `docs/play.md`). So $\bar r$ is read against `null_mean`, the mean score of shuffled labels (section 2.3), and a negative $r$ is never a finding by itself.

A second diagnostic, `max_penalty_share`, is the share of folds in which ridge chose the largest $\alpha$ on the grid. Near 1 means the model gave up and predicted a constant.

### 2.3 Permutation p

Labels are shuffled $B$ times (default 200). For each shuffle $y^{(b)}$ the whole procedure is refitted with one repeat, giving a null score $r^{(b)}$ (`evaluate.py:predict`). Then

$$p_{\text{perm}} = \frac{1 + \#\{b : r^{(b)} \ge \bar r\}}{1 + B}, \qquad \texttt{null\_mean} = \frac1B\sum_b r^{(b)}.$$

The test is one-sided and compares like with like: the null goes through the same cross-validation bias as the real score. The smallest possible p is $1/(1+B)$, so after Holm over $m$ traits nothing can survive when $m/(1+B) \ge 0.05$. The function warns in that case and asks for $B \ge 20m$.

### 2.4 Bootstrap confidence interval

People are resampled with replacement $2{,}000$ times (index sets $I_b$). The out-of-fold predictions are held fixed, and each resample's statistic is the same per-repeat average:

$$r^*_b = \frac1R\sum_\rho \operatorname{corr}\big(\hat y^{(\rho)}_{I_b}, y_{I_b}\big), \qquad \text{CI} = \big[q_{2.5\%}(r^*), q_{97.5\%}(r^*)\big].$$

The interval shows sampling spread around $\bar r$. It does not include the cross-validation bias of section 2.2, so an interval that sits entirely below zero still says nothing about a negative relationship (`docs/play.md`).

### 2.5 Holm

Traits tested together are one family. With $p_{(1)} \le \dots \le p_{(m)}$, Holm's step-down adjustment is (`evaluate.py:holm`)

$$\tilde p_{(i)} = \min\Big\{1,\ \max_{j \le i}\,(m-j+1)\,p_{(j)}\Big\},$$

and a trait is marked `survives` when $\tilde p < 0.05$. This keeps the chance of any false claim in the family below 0.05.

### 2.6 Yes/no labels: cross-validated AUC

For binary labels such as one MBTI letter, `evaluate.py:classify` fits a pipeline of imputation, scaling and `LogisticRegressionCV` (10 values of $C$) in stratified 5-fold CV, one pass, and reports the out-of-fold AUC with a 1,000-resample bootstrap interval. It has no permutation test; for the forum MBTI slice the intervals came from the Hanley-McNeil formula instead (`docs/text.md`). AUC is used for every yes/no target because it is unaffected by lopsided classes: always guessing N is 86% accurate on the MBTI forum and has AUC 0.5.

The text readers and cross-domain scripts use their own scorers with the same ideas: a person bootstrap for the CI and a label-shuffle null (`readers/text_train.py:boot_ci`, `scripts/text_crossdomain.py:score_block`, which resamples whole characters for FriendsPersona and uses 1,000 shuffles).

### 2.7 Converting between AUC and r

Many readers are tested with AUC against a median split, while fusion and the power analysis work in $r$. If a score and a normal trait are bivariate normal with correlation $r$, the AUC for "trait above its median" is a three-variable orthant probability, which gives (`src/aas/fusion.py:r_to_auc`, `auc_to_r`; the same formula in `readers/text.py:auc_to_r`)

$$\text{AUC} = \tfrac12 + \tfrac{2}{\pi}\arcsin\!\frac{r}{\sqrt2}, \qquad r = \sqrt2\,\sin\!\Big(\frac{\pi}{2}\big(\text{AUC} - \tfrac12\big)\Big).$$

For example AUC 0.682 corresponds to $r = 0.40$, and the tweets AUC 0.5427 for E corresponds to $r = 0.095$. The model cards also store an `r_implied` per text length computed from a simulated version of the same relation (200,000 draws, `text_train.py:r_auc_table`). It sits 0.0004 to 0.0009 above the closed form, so the readers, fusion and the power analysis all use the closed form and the card column is kept only as a record.

### 2.8 Wins and losses across people

The TypeScript bench added a third condition to "survives": the model must beat its baseline for more people than it loses to, by an exact two-sided sign test with Holm across the report's comparisons, and the bootstrap interval of the mean difference must sit above zero ([`../../docs/MATH.md`](../../docs/MATH.md) section 16). `evaluate.predict` has no per-person comparison, because each slice there has one model and a shuffled-label null.

The Python engine expresses the same idea in two places. `src/aas/profile.py:evaluate` reports, per reader and trait, the mean absolute error against guessing the average for everyone,

$$\text{skill} = 1 - \frac{\frac1n\sum_i |\hat z_i - z_i|}{\frac1n\sum_i |z_i|},$$

which is positive only when the reader is closer to people's quiz scores than $z = 0$ is. And the study's analysis tests per-person differences directly: for H2, $d_{it} = (q_{it} - \text{single}_{it})^2 - (q_{it} - \text{combined}_{it})^2$ summed over traits; for H4, $M_i = \sum_t \big(|q_{it}| - |q_{it} - \text{combined}_{it}|\big)$. Each is tested with a one-sided t-test confirmed by 10,000 random sign flips, and the permutation p is reported when the two disagree at 0.05 (`scripts/study_analysis.py:paired_test`).

## 3. The readers

A reader turns a `Person` (their own texts, plays and watches, `src/aas/person.py`) into `Estimate`s. Each Estimate carries a value, an optional interval with the level it claims, the reader's tested validity on unseen people, and the evidence in plain words. Too little data gives `usable=False` with the reason.

### 3.1 Text: what is read

Only the person's own prose is read. Items of kind `search` are dropped, because two-to-four-word queries look nothing like the sentences the models learned from (`readers/text.py:own_prose`, `EXCLUDED_KINDS`).

Every item, in training and for a new person, goes through one leak stripper, `readers/text.py:clean`. It removes URLs, the 16 MBTI types in any case with plurals and wildcards (INxJ, ENFP-T), temperaments, cognitive-function jargon, words naming a letter or the system, and in strict mode leftover theory jargon (`slices/text.py:strip_leaks`), then Big Five self-report words such as openness, neurotic, percentile, IPIP and truity (`slices/text_pandora.py:BIG5_LEAK`). The leak is large: on the MBTI forum, 21 leak counts alone reach AUC 0.87 to 0.90, and 90% of users name their own type at least once (`docs/text.md`).

Resampling units are the items themselves when a person has at least 20, otherwise 20-word pieces of their text (`readers/text.py:units`).

Each item is also sorted into a kind of writing (`readers/text.py:kind_of_writing`): tweets (source X or Twitter, or kind tweet, reply, retweet, quote), essay-like long prose (kind essay, journal, diary, blog, letter, or any single item of 300 words or more), and other prose (Reddit comments and the rest). Words per kind feed the stated validity in sections 3.6 and 3.7.

### 3.2 Text: bag of words and the head

Features are word counts over a fixed vocabulary (up to 20,000 terms seen in at least `min_df` people; unigrams for MBTI, unigrams plus bigrams for Essays, both chosen by 5-fold CV on the 80% dev split, `readers/text_train.py:select_pipeline`). For one person's count vector $c$,

$$x_w = \big(1 + \log c_w\big)\,\mathrm{idf}_w \ \ (c_w > 0), \qquad \mathrm{idf}_w = \log\frac{1+N}{1+\mathrm{df}_w} + 1, \qquad x \leftarrow \frac{x}{\|x\|_2},$$

which is sublinear TF-IDF with L2 normalisation (`readers/text.py:tfidf`, `text_train.py:fit_vocab`). Each trait has a linear head, $z = \beta^\top x + b$: logistic regression with $C \in \{0.1, 0.3, 1, 3\}$ picked per trait for yes/no labels, ridge for continuous labels (`text_train.py:fit_head`). MiniLM sentence embeddings were scored on the same folds as a challenger and lost on every MBTI letter and every Essays trait (`docs/readers_text.md`).

### 3.3 Length-aware calibration

Logistic regression on thousands of word features ranks people well but is overconfident on short texts. Each head is therefore recalibrated with a length term (`readers/text.py:Head.recal`, fitted by `text_train.py:fit_recal`):

$$s = a\,z + b\,z\,\ell + c\,\ell + d, \qquad \ell = \log\frac{\max(w, 1)}{w_{\text{ref}}},$$

where $w$ is the person's word count and $w_{\text{ref}}$ the median words per training person (1,304 for MBTI, 626 for Essays, from the model cards). For yes/no labels $P = \sigma(s)$ and $(a,b,c,d)$ come from an essentially unpenalised logistic regression ($C = 10^4$); for continuous labels, from least squares. The fitting data are the dev people's out-of-fold head scores with their text cut to every length on the grid (MBTI 25 to 1,200 words, Essays 25 to 500) and at full length (`text_train.py:fit_model`). The $b$ term lets the model be bolder with more text and humbler with less. On held-out people this halves MBTI calibration error (I/E ECE 0.065 to 0.028) and leaves Big Five near the noise floor of a 494-person test set (`docs/readers_text.md`).

### 3.4 Minimum text

Held-out people are cut down to $k$ words (random items until $k$ is reached, 10 repeats) and scored against the same people's full-text AUC (`text_train.py:evaluate_test`). With $\bar A_k$ the mean over traits, two lengths are read off (`text_train.py:pick_min_text`):

$$k_{\text{rec}} = \min\{k : \bar A_{\text{full}} - \bar A_{k'} \le 0.03 \ \forall k' \ge k\}, \qquad k_{\text{floor}} = \min\Big\{k : \frac{\bar A_{k'} - 0.5}{\bar A_{\text{full}} - 0.5} \ge 0.5 \ \forall k' \ge k\Big\},$$

with the floor capped at $k_{\text{rec}}$. Below the floor the reader says nothing (`readers/text.py:_TextReader.can_read`). Text MBTI: floor 100 words, recommended 800. Text Big Five: floor 25, recommended 100; its floor is low because its full-text signal is already weak. Between floor and recommended the reader answers, says accuracy is still dropping, and states the curve's AUC at the person's length.

### 3.5 Long histories

Training people had at most about 1,600 (MBTI) or 1,000 (Essays) words at the 90th percentile. When a person has more than 1.5 times that cap, their units are shuffled and packed into chunks of at least the cap (a last chunk is kept if it has at least half), each chunk is scored at its own length, and the score is the mean over chunks; the bootstrap resamples chunk scores (`readers/text.py:_TextReader._scores`). Otherwise the whole text is scored once, and the bootstrap draws multinomial weights over units, $W \sim \text{Mult}(n_u, \mathbf 1/n_u)$, and rescores the weighted counts $W C$ 200 times.

### 3.6 Text MBTI

Training data: 8,675 PersonalityCafe forum users (last ~50 posts each), self-reported type, one yes/no head per letter (`text_train.py:load_mbti_corpus`). The forum is 77% I and 86% N, which describes who joins a typology forum. The reader removes that base rate $\pi$ so that text with no telling words gives 0.5 (`readers/text.py:TextMBTIReader._estimate`):

$$P = \sigma\big(s - \operatorname{logit}\pi\big).$$

The 80% range is the 10th and 90th percentiles of $\sigma(s^*_b - \operatorname{logit}\pi)$ over bootstrap scores re-centred on the point estimate, $s^*_b \leftarrow s^*_b - \operatorname{median}(s^*) + s$, because resampling duplicates items and sublinear TF shifts the cloud. This range describes how much the answer depends on which texts happened to be collected. It makes no claim about how often the letter itself is right.

Validity on the forum (held-out test, `docs/readers_text.md`): I/E 0.787, N/S 0.778, T/F 0.878, J/P 0.709. No MBTI-labelled writing from outside the forum is openly available, so the off-forum accuracy is inferred. The letter model is run on the 2,467 student essays against the linked Big Five trait, that AUC is turned into $r$ (section 2.7), and divided by the published MBTI to Big Five link $c$ (McCrae and Costa 1989: 0.74 E, 0.72 O, 0.44 A, 0.49 C):

$$\rho_{\text{letter}} = \frac{r(\text{AUC}_{\text{essays, linked trait}})}{c}.$$

For I/E: AUC 0.584 gives $r = 0.19$, divided by 0.74 gives letter $r$ 0.2514, AUC 0.6138. The inferred AUCs are 0.61 (I/E), 0.62 (N/S), 0.62 (T/F), 0.62 (J/P), keeping 33% to 56% of the in-domain letter $r$. The same number is used for every kind of writing, scaled for short texts as in section 3.7.

The forum-calibrated probability is too bold at this lower accuracy, so the reader keeps the person's latent score and re-expresses it at the stated accuracy (`readers/text.py:shrink_p`). With $\rho_{\text{in}}$ and $\rho$ the letter correlations implied by the in-domain and stated AUCs,

$$u = \frac{\sqrt{1-\rho_{\text{in}}^2}\;\Phi^{-1}(P)}{\rho_{\text{in}}}, \qquad P' = \Phi\!\left(\frac{\rho\,u}{\sqrt{1-\rho^2}}\right) \quad (\rho < \rho_{\text{in}}).$$

This applies to the point and to both ends of the range. The Estimate reports `validity` = the stated AUC, `validity_kind = "auc"`, `interval_level = 0.80`.

### 3.7 Text Big Five (Essays)

Training data: 2,467 University of Texas student essays (Pennebaker and King 1999), Big Five from a questionnaire released only as a median split, so each head predicts "above the median" (`text_train.py:load_essays_corpus`). Held-out test AUC (494 people): O 0.682, C 0.573, E 0.594, A 0.584, N 0.585 (`docs/readers_text.md`).

The head gives a probability, and the reader turns it into an expected z. Suppose the trait $z$ and the model's standardised latent score $S$ are bivariate normal with correlation $r$. Then $E[z \mid S] = rS$, $\operatorname{Var}(z \mid S) = 1-r^2$, and $P(z > 0 \mid S) = \Phi\big(rS/\sqrt{1-r^2}\big)$, so the expected z follows from the calibrated probability:

$$m = E[z \mid S] = \sqrt{1-r^2}\;\Phi^{-1}(P).$$

Plain $\Phi^{-1}(P)$ would overstate $z$ by $1/\sqrt{1-r^2}$, 1 to 9% at these validities.

The stated accuracy depends on the kind of writing (`readers/text.py:_TextReader.validity_mix`). In-domain accuracy at the person's length is $A_{\text{in}}(w)$, read off the card's length curve by linear interpolation in $\log w$. Off-domain AUCs were measured on whole histories, so their lift above 0.5 is scaled by the share of full-text lift the in-domain curve keeps at that length:

$$\lambda(w) = \operatorname{clip}\Big(\frac{A_{\text{in}}(w) - 0.5}{A_{\text{full}} - 0.5},\, 0,\, 1\Big), \qquad A_k(w) = \begin{cases} A_{\text{in}}(w) & k \text{ in domain} \\ 0.5 + (A_k^{\text{meas}} - 0.5)\,\lambda(w) & k \text{ shown off domain} \\ 0.5 & k \text{ not shown}\end{cases}$$

The stated AUC is the word-weighted mix $A = \sum_k f_k A_k(w)$ over the person's kinds of writing, with word shares $f_k$. For text Big Five, essay-like prose is in domain; tweets use the PAN 2015 measurement (C 0.54, E 0.54, A 0.54, N 0.57); other prose (Reddit) has not been measured and uses the lower of the two held-out domains made of a person's own words, which is the PAN number for every trait (model card `validity_by_source`). Openness is "not shown" off domain, because its 95% interval includes 0.5 on every held-out domain (Friends 0.50, PAN 0.52, vlogs 0.47). If less than half of a person's words are on kinds where a trait has been shown to work, the reader refuses it: `usable=False`, `validity = 0`, `validity_kind = "r"` (`readers/text.py:TextBigFiveReader._refusal`). In practice this refuses openness for anyone writing mostly on Reddit or X.

The expected z and the interval then follow the stated accuracy (`readers/text.py:TextBigFiveReader._estimate`). With $r_{\text{in}} = r(A_{\text{in}})$, the closed-form conversion of section 2.7, and $P_{\text{in}} = \sigma(s - \operatorname{logit}\pi)$,

$$\kappa = \operatorname{clip}\Big(\frac{r(A)}{r(A_{\text{in}})}, 0, 1\Big), \qquad r = \kappa\,r_{\text{in}}, \qquad m = \kappa\sqrt{1-r_{\text{in}}^2}\;\Phi^{-1}(P_{\text{in}}),$$

$$\text{sd} = \sqrt{(1-r^2) + \operatorname{Var}_b(m^*_b)}, \qquad [\text{low}, \text{high}] = m \pm 1.2816\,\text{sd},$$

where $m^*_b$ repeats the computation of $m$ on each bootstrap score and 1.2816 is the standard normal 90th percentile, so the interval claims 80%. The percentile shown is $100\,\Phi(m)$ and the chance of being above the median is $\Phi\big(m/\sqrt{1-r^2}\big)$. For essay-like text $\kappa = 1$ and nothing changes. Whenever the clip is inactive $r = r(A)$, the same $r$ that fusion recovers from the stated AUC in section 5.2. The same words labelled as tweets give a z closer to 0 and a wider interval. With $r \approx 0.1$ the interval is about $\pm 1.27$ z, close to the $\pm 1.28$ that fits anyone.

A retrain on PANDORA (Reddit, continuous self-report) is wired behind one flag and switches the reader to the continuous path: $m = s$, with the interval from the 10th and 90th percentiles of held-out residuals at that length widened by the bootstrap spread, $[\,m + q_{10}(w) - 1.2816\,\text{sd}^*,\ m + q_{90}(w) + 1.2816\,\text{sd}^*\,]$, and `validity_kind = "r"`. It has been tested on a synthetic corpus only; the PANDORA files have not been requested.

### 3.8 Music

The music reader measures how someone listens and states no personality claim, because music taste predicts the Big Five only weakly. It is a port of the traits from the TypeScript engine that held up on 304 ListenBrainz listeners (`readers/music.py`; definitions and the bench in [`../../docs/MATH.md`](../../docs/MATH.md) section 15). A full play is one of 30 s or more; a play with no recorded length counts as full and as 3 minutes on the clock. Times are start times in UTC.

With $a_i$ the artist of play $i$, $t_0$ the first play, $\text{first}(a)$ the time artist $a$ was first heard (any play, skips included) and $P$ the full plays starting at least 12 weeks after $t_0$:

- New share (`music.py:new_share`), at least 300 plays in $P$: $\dfrac{\#\{i \in P : t_i - \text{first}(a_i) < 28\,\text{d}\}}{\#P}$.
- Old friends (`music.py:old_friends`): cut the history into 8-week stretches $S_j$ of full plays; for $j \ge 2$, $F_j = \text{Top}_{50}(S_{j-2}) \cup \text{Top}_{50}(S_{j-3})$ (only $S_{j-2}$ when $j = 2$); average $\#\{i \in S_j : a_i \in F_j\}/\#S_j$ over stretches with more than 100 plays (at least two).
- Season turnover (`music.py:season_turnover`): over 13-week seasons $B_j$, the mean of $|\text{Top}_{10}(B_j) \setminus \text{Top}_{10}(B_{j+1})|/10$ over consecutive pairs with full top tens (at least two).
- Clock spread and regularity (`music.py:clock_shape`). Each play is an angle $\theta_i = 2\pi\,(\text{minute of day})/1440$ with weight $w_i = \min(\text{ms}_i, 10\,\text{min})$ in minutes. Spread is $1 - R$ with $R = \big|\sum_i w_i e^{\mathrm i\theta_i}\big| / \sum_i w_i$ (at least 500 plays). For regularity, days start 12 hours from the person's centre; each day with at least 5 plays gives a unit vector $v_d = \sum_{i\in d} e^{\mathrm i\theta_i} / |\cdot|$, and regularity is $\big|\frac{1}{D}\sum_d v_d\big|$ over at least 30 such days. A fixed time-zone offset rotates every angle equally and changes neither length.
- Finish share: the share of plays whose `reason_end` is trackdone (at least 300). Skip seconds: the median seconds played before a skip ended by fwdbtn or unknown (at least 100).

Each trait's percentile is read from a 1st to 99th percentile table of the 304 listeners, linearly between knots, half the first knot below it and halfway to 100 above the last (`music.py:percentile_of`). The two dials come from a five-trait factor check on 301 listeners (eigenvalues 2.568, 1.861, then 0.273; parallel analysis keeps 2) and are weighted sums of normal scores (`music.py:dial_scores`, weights in `src/aas/data/music_dials.json`):

$$D = \sum_k w_k\,\Phi^{-1}\!\big(\operatorname{clip}(p_k/100,\ c,\ 1-c)\big), \qquad c = 1/302,$$

$$D_{\text{explore}} = 0.360\,z_{\text{new}} - 0.375\,z_{\text{old}} + 0.365\,z_{\text{turnover}}, \qquad D_{\text{regular}} = -0.528\,z_{\text{spread}} + 0.527\,z_{\text{regularity}}.$$

Music Estimates have kind `measured`, no interval (`interval_level = None`) and no validity; they carry `reliability`, the split-half retest $r$: new share 0.754, old friends 0.811, season turnover 0.802, clock spread 0.720, clock regularity 0.725 (304 listeners), finish share 0.86 and skip seconds 0.99 (13 Spotify people). The Python port matches the TypeScript values for 304 of 304 people on four traits and 303 of 304 on clock regularity (`docs/reader_music.md`).

The TypeScript document covers the rest of the listening work. Its headline results, as stated there: the skip model reaches AUC 0.848 against 0.617 for the artist's skip rate, better for 13 of 13 people; whether a just-heard artist sticks reaches median AUC 0.773 after the first session on 13 Spotify people and 0.792 on 304 ListenBrainz listeners; the forecast of new loves does not beat popularity on 13 people and does on 304 rolling windows, with recall still small (4.7% in the top 50); eras do not improve the forecast. None of this links music to the Big Five.

### 3.9 The LLM rater (benchmark only)

`readers/llm_rater.py` asks a large language model to rate a person's Big Five as their IPIP-50 score would come out (1.0 to 5.0 per trait) from up to about 1,500 tokens of their leak-stripped text, with prompt `b5-ipip50-v1`. Nothing is trained on our data. It refuses to build a prompt whose text still contains a type or a Big Five word (`llm_rater.py:assert_no_labels`). Because the product promises that a participant's text never leaves their machine, an API rater can only be a research benchmark on public datasets.

The pre-registered test (`docs/llm_rater.md`, `scripts/llm_crossdomain.py:live`) rates the same 150 people per dataset (seed 20261101) and compares, on the same cut text, the LLM score with the bag-of-words score and with their sum of within-dataset z-scores. The metric is AUC against each dataset's median split, mean over traits, averaged over PAN 2015 and the vlogs with each dataset weighted equally. With $\Delta$ the difference in that metric and a 95% CI from 2,000 paired bootstraps of people within each dataset:

- REPLACE if $\Delta(\text{llm} - \text{bow}) \ge 0.03$ and its CI excludes 0;
- AUGMENT otherwise, if $\Delta(\text{combined} - \text{bow}) \ge 0.03$ and its CI excludes 0;
- KEEP otherwise.

Result for Claude Haiku 4.5 (`docs/llm_crossdomain_results_anthropic_claude-haiku-4-5-20251001.json`): held-out mean AUC 0.618 against 0.558 for bag of words, $\Delta = +0.060$ [0.025, 0.093], so REPLACE; the combined arm is $+0.050$ [0.031, 0.069]. No rating was missing. Section 6 gives the per-dataset numbers. A local run with `qwen3:1.7b` stopped on the disk rule after 17 people and gave no verdict.

### 3.10 Distilled on-device text reader

The LLM rater of section 3.9 reads writing it was never trained on better than the bag of words, but it runs on Anthropic's servers, and the product promises that a participant's text stays on their machine. Distillation tries to move that skill onto the laptop: the teacher (Claude Haiku 4.5, prompt `b5-ipip50-v1`, unchanged) rates a pool of other people's public writing once, a small model (the student) learns to give the same ratings, and only the student would ship (`docs/distill.md`, `scripts/distill.py`, `readers/distilled.py`, `models/text_distilled/`).

The pool is 2,000 PersonalityCafe forum users in a seeded order (seed 20261105); their MBTI types are dropped on load and never read. Each person's posts go through the leak stripper of section 3.1 and are cut to about 1,500 tokens exactly as for the rater (median 1,034 words). None of the three test sets (Essays, PAN 2015, vlogs) contributes a person, a text or a label to the pool, to fitting or to tuning. The teacher rated all 2,000 through the Message Batches API with no refusals or invalid replies, for $3.35 (`data/distill_spend.jsonl`).

Each student is five ridge regressions, one per trait, from features of the cut text to the teacher's rating $h_{it} \in [1, 5]$, minimising $\sum_i (h_{it} - \beta_t^\top x_i - b_t)^2 + \alpha_t \|\beta_t\|^2$ with $\alpha_t$ chosen by generalised cross-validation on the pool (grid $10^{-2}$ to $10^4$). Three students were fixed before any rating was obtained: `bow` (the TF-IDF features of section 3.2, refitted on the pool), `emb` (all-MiniLM-L6-v2 embeddings averaged over 100-word chunks) and `mix` (the mean of the two students' z-scored predictions). The one that goes forward was chosen on the pool alone, by 5-fold cross-validated Pearson $r$ with the teacher averaged over traits: `bow` 0.563, `emb` 0.595, `mix` 0.624, so `mix`. It explains about 40% of the variance in the teacher's ratings of new forum users. On disk it is 0.6 MB plus the 90 MB MiniLM model already used elsewhere.

The pre-registered test (committed before any pool rating) scored the student on the same 150 people per test set that the teacher rated in section 3.9, on the same cut text, with AUC against each set's median split averaged over traits and 2,000 paired bootstraps within each set. The student passed only if both held: (1) mean AUC over PAN 2015 and vlogs at least 0.588, half of Haiku's lead over bag of words there; (2) mean AUC over all three sets at least bag of words plus half of Haiku's lead on the same people, 0.596.

It failed both, narrowly on the first. Over PAN 2015 and vlogs the student scored 0.586 [0.556, 0.616] against the mark of 0.588 (Haiku 0.618, bag of words 0.558). Over all three sets it scored 0.583 [0.558, 0.608] against 0.596 (Haiku 0.618, bag of words 0.574), because on Essays it lost to a bag of words trained on essays (0.576 against 0.605). On the 150-person samples the student's lead over bag of words was $+0.028$ [$-0.002$, $+0.059$] on PAN and vlogs, and it was clearly behind Haiku ($-0.031$ [$-0.061$, $-0.003$]). Outside the rule, on every test person (494 Essays, 294 PAN, 404 vlogs; no Haiku column, since Haiku rated only the 150), the student beat bag of words on PAN and vlogs by $+0.030$ [$+0.010$, $+0.051$], almost all of it on vlogs, with tweets level (`docs/distill_results.json`).

What this shows is that 2,000 forum users rated by Haiku moved much of Haiku's advantage on vlogs onto the laptop and almost none of it on tweets or essays. It does not separate three explanations: a pool of one kind of writing, a teacher that is itself weak on tweets (0.557), and a linear student that copies only about 40% of the teacher's variance. It also says nothing about Reddit, which no reader has been measured on. The student's numbers are fixed facts of this one test. Re-tuning it against these test sets would be a second look at them, so the next step is a new pool under the same frozen rule.

The reader exists and is not a default. `TextDistilledReader` (name `text-distilled`) reads own prose from 50 words up, cuts it as the teacher's input was cut (longer histories as up to five such samples, averaged), and turns the student score into $z_s = (s - \bar s_{\text{pool}})/\text{sd}_{\text{pool}}$. With $A$ the AUC its card states for the person's word-weighted mix of writing (essay-like from Essays, tweets from PAN 2015, other prose the lower of PAN and vlogs per trait, all on every test person) and $r = r(A)$ by section 2.7,

$$\text{value} = r\,z_s, \qquad [\text{low}, \text{high}] = r\,z_s \pm 1.2816\sqrt{1-r^2},$$

with `interval_level = 0.80`, `validity` $= A$ and `validity_kind = "auc"`. A trait whose stated $A \le 0.5$ comes back `usable=False`; on tweets that is openness (0.473) and conscientiousness (0.476). Every estimate's evidence says the reader failed its test. Fusion would read it like `text-bigfive` (section 5.2) and put it in the same modality as `text-mbti`; section 5.6 shows what that would change.

Contamination. The student never saw a test text or label, and Haiku saw no test text in this experiment, so the student's numbers carry no memorisation risk of their own. Haiku's numbers, and so the pass marks, carry the caveat of section 7. Haiku might also remember PersonalityCafe users' types and rate by stereotype, which would make the teacher worse and the test no less fair.

Terms. The student is trained on Claude outputs. Anthropic's commercial terms restrict using outputs to build competing AI models. A regressor that outputs five personality numbers is unlikely to count as one, but this has to be checked before any commercial use of the student or of anything trained the same way (card `terms_note`).

## 4. Estimates with intervals

### 4.1 What each reader claims

`Estimate.interval_level` is how often `[low, high]` claims to contain the truth, and `Estimate.validity` is the reader's tested accuracy on unseen people (`person.py:Estimate`). Each reader sets them as follows:

| reader | value | interval and `interval_level` | `validity` |
|---|---|---|---|
| `text-bigfive` | expected z, section 3.7 | $m \pm 1.2816\,\text{sd}$, 0.80, a coverage claim about the quiz z | stated AUC for this person's mix of writing; 0 with `validity_kind = "r"` when refused |
| `text-mbti` | $P(\text{first letter})$ | bootstrap 10% to 90% range, 0.80, a claim about resampling the person's texts | stated (inferred off-forum) AUC |
| `text-distilled` (not a default) | $r\,z_s$, section 3.10 | $r\,z_s \pm 1.2816\sqrt{1-r^2}$, 0.80 | stated AUC for this person's mix of writing; `usable=False` at or below 0.5 |
| `music` | the measured trait or dial | none, `None` | none; `reliability` holds the split-half $r$ |
| `fused` | posterior mean z, section 5 | posterior or conformal interval, 0.80 | `None`: fusion has not been tested on people with several sources; the model-implied $r$ goes in the evidence text |
| any reader, too little data | NaN, `usable=False` | `None` | none |

### 4.2 How coverage is checked

For one person, `profile.py:compare` puts every Big Five estimate next to the quiz z and records whether the quiz z falls inside the interval. Across many people, `profile.py:evaluate` reports per reader and trait:

$$\text{coverage} = \frac1n\sum_i \mathbf 1\{\text{low}_i \le z_i \le \text{high}_i\}, \qquad \text{mean width} = \frac1n\sum_i (\text{high}_i - \text{low}_i),$$

together with MAE, skill (section 2.8), $r$ and the share on the same side of average. The stated level is taken from the estimates' `interval_level`, and the function raises an error if one reader and trait state more than one level (`profile.py:_stated`). Coverage well below the stated level means overconfident intervals, well above means wider than needed. A simulated overconfident reader with $r = 0.70$ shows coverage 0.25 against a stated 0.80 and no skill (`docs/profile.md`), which is why coverage and skill are read before $r$.

The study turns this into a registered test (PREREG H3): pooled coverage over person-traits with a 90% person-cluster bootstrap interval, and equivalence to 80% is supported if that interval lies within [0.75, 0.85] (`scripts/study_analysis.py:coverage_stats`). For a true 80%, one trait's coverage is known to $\pm 5$ points at $n = 246$, since the 95% half-width is $1.96\sqrt{0.16/n}$ (`docs/study/POWER.md` section 4). MBTI estimates get only a loose same-side-of-average check against their linked Big Five trait, which `evaluate` ignores.

No reader's coverage has been measured on real people yet. The Essays labels are yes/no, so the text Big Five interval could not be checked on its own data.

## 5. Fusion

### 5.1 The measurement model

Each source that speaks about a trait is treated as a noisy measurement of the person's quiz z. Every source is put on one scale, a standardised score $u_j$ with mean 0 and sd 1 in the population and correlation $v_j$ (its validity) with the quiz (`src/aas/fusion.py`, module docstring):

$$z \sim N(0, 1), \qquad u_j = v_j\,z + \sqrt{1-v_j^2}\;e_j, \qquad \operatorname{corr}(e_j, e_k) = R_{jk}.$$

The prior $N(0,1)$ holds because quiz z is already relative to the norms. Correlated errors capture two sources that read the same thing, such as text Big Five and text MBTI on the same posts.

### 5.2 From Estimates to readings

`fusion.py:readings_from_estimates` converts each usable Estimate with a tested validity:

- Big Five: the value $m$ is a regression-shrunk expected z whose spread is about its validity $r$, so $u = m/r$, with $r$ the stated validity (an AUC is converted by section 2.7). A validity below 0.02 is skipped.
- MBTI letter, as a soft measurement: with $\rho$ the letter $r$ implied by the stated AUC (capped at 0.95), the latent preference score is $\sqrt{1-\rho^2}\,\Phi^{-1}(P)$ with spread $\rho$, so $u_L = \sqrt{1-\rho^2}\,\Phi^{-1}(P)/\rho$. The linked trait gets $u = \text{sign}\cdot u_L$ and validity $\rho\,c$ with the link $c$ from `fusion.py:MBTI_LINKS` (I/E to E, $-$, 0.74; N/S to O, $+$, 0.72; T/F to A, $-$, 0.44; J/P to C, $+$, 0.49). Since `shrink_p` (section 3.6) already re-expressed $P$ at the stated $\rho$, this recovers the same latent $u$.
- Measured traits are ignored unless a tested link to a Big Five trait is supplied; none exists, so music is not fused.

### 5.3 The zero-data posterior

For a person with sources $j = 1..k$, the posterior of $z$ is Gaussian (`fusion.py:gaussian_posterior`):

$$S = v v^\top + D R D, \qquad D = \operatorname{diag}\big(\sqrt{1-v_j^2}\big), \qquad E[z \mid u] = v^\top S^{-1} u, \qquad \operatorname{Var}(z \mid u) = 1 - v^\top S^{-1} v.$$

$\sqrt{1 - \operatorname{Var}}$ is the accuracy the model expects for that set of sources, reported as the implied $r$. For two sources of equal validity $v$ and error correlation $\rho$ this gives $r^2_{\text{fused}} = 2v^2/\big(1 + v^2 + \rho(1-v^2)\big)$: two sources of 0.30 reach 0.41 when independent, 0.35 at $\rho = 0.4$, and 0.30 when their errors are identical (`docs/fusion.md` section 1).

The default method `bayes` (`fusion.py:Bayes` with `eb=False`) uses only the stated validities, with three adjustments:

1. Stated validities come from each reader's own data, and new people's data is out of that domain, so each is discounted: $v \leftarrow \tanh\big(\operatorname{artanh}(0.75\,v)\big)$, that is $0.75\,v$ clipped to $\pm 0.95$ (`DISCOUNT = 0.75`, `Bayes.params`).
2. Validities are uncertain on new people. Each is drawn 200 times in Fisher z, $\operatorname{artanh} v + \tau\,\varepsilon$ with $\tau = 0.10$ for Big Five readers and 0.15 for MBTI (a chain of two correlations). The posterior is the mixture over draws: the mean of the means, and the mean of the variances plus the variance of the means.
3. Errors are correlated within a modality: $R_{jk} = 0.30$ for two sources of the same modality (same reader-name prefix, so `text-bigfive` and `text-mbti` share one) and 0 across modalities (`fusion.py:prior_error_corr`).

The interval is $\mu \pm 1.2816\,\text{sd}$ (`fusion.py:_Gauss`), and `FusedReader` reports the percentile $100\,\Phi(\mu)$ and leaves `validity = None`. The implied $r$ in its evidence is computed at the discounted validities with no draws, $\sqrt{v^\top S^{-1} v}$, so it describes the set of sources. Until 5 Nov it was taken from the person's own mixture variance, which includes the spread of means over validity draws and so shrank as that person's reading grew (a strong MBTI letter printed "r about 0.00").

### 5.4 Methods that need joint training data

These exist for the day the study supplies people with several sources and a quiz (`fusion.py:make_method`):

- `bayes_eb`: the same model with each validity, score location and scale, and each error correlation updated from training people and shrunk towards the stated values. A validity is combined in Fisher z with prior precision $1/\tau^2$ and data precision $n-3$; an error correlation with prior weight 20 against $n-3$; the matrix is then projected to a valid correlation matrix (`fusion.py:Bayes.fit`, `_psd_corr`).
- `precision`: plug-in validities and error covariance, inverse-variance weighting of the debiased scores $u_j/a_j$, then the $N(0,1)$ prior, $\mu \leftarrow \mu/(1+\sigma^2)$ (`fusion.py:Precision`).
- `stack`: ridge on source scores plus missingness indicators (`fusion.py:Stack`).
- CV+ conformal intervals around any of these (Barber, Candes, Ramdas and Tibshirani 2021; `fusion.py:CVPlus`). With 10-fold out-of-fold residuals $R_i = |y_i - \mu_{k(i)}(x_i)|/\text{sd}_{k(i)}(x_i)$ and $a = 0.2$, the interval for a new person is the $\lfloor a(n+1)\rfloor$-th smallest of $\mu_{k(i)}(x) - \text{sd}_{k(i)}(x)R_i$ and the $\lceil(1-a)(n+1)\rceil$-th smallest of $\mu_{k(i)}(x) + \text{sd}_{k(i)}(x)R_i$. Coverage is guaranteed to be at least $1-2a$ for exchangeable people and is about $1-a$ in practice. `bayes_eb_cp` is `bayes_eb` inside CV+.

### 5.5 What the simulation showed

`scripts/sim_fusion.py` simulates 3,000 test people per seed over 20 seeds, with realistic trait intercorrelations, missing sources (13% none, 43% one, 33% two, 11% three or more) and correlated errors, and scores every method with 0, 60, 250 or 600 training people (`docs/fusion.md`, every row in `docs/fusion_sim.csv`).

With the validities assumed before any cross-domain test (text 0.32, 0.20, 0.25, 0.20, 0.22 for O C E A N), the zero-data `bayes` reached $r = 0.240$ with 80% coverage 0.805, and no trained method beat it even at $n = 600$ (best 0.233 to 0.236). Trained methods were worse than guessing the average at $n = 60$. The discount changed coverage and left ranking alone: when stated validities were too high, coverage was 0.777 without it and 0.797 with it.

Rerun with validities measured on writing the readers never saw (section 5b of `docs/fusion.md`; text Big Five refuses O and has $r$ 0.097, 0.095, 0.094, 0.158 for C, E, A, N on tweets, the closed-form conversion of the card's AUCs (E: 0.5427 to 0.095); MBTI via the links 0.20, 0.13, 0.19, 0.12 for O, C, E, A, inferred):

| scenario | method, n_train | $r$ with truth | skill | 80% coverage | width (z) |
|---|---|---|---|---|---|
| measured | `bayes`, 0 | 0.150 | +0.011 | 0.802 | 2.54 |
| measured | `bayes_eb_cp`, 600 | 0.139 | +0.010 | 0.799 | 2.54 |
| readers state the old in-domain numbers, truth measured | `bayes`, 0 | 0.137 | $-0.003$ | 0.785 | 2.49 |
| same, no discount | `bayes`, 0 | 0.137 | $-0.018$ | 0.765 | 2.43 |
| truth at the low end of the CIs | `bayes`, 0 | 0.071 | +0.001 | 0.795 | 2.54 |

Fused $r$ falls from about 0.24 to about 0.15, and most of what fusion adds comes from the MBTI letters, whose off-forum accuracy is inferred. Coverage is right again once readers state the accuracy measured on the kind of writing they read. Intervals are close to the population interval (2.54 z against 2.56), so a user sees a z near 0 with a wide range on most traits. The decision stands: `bayes` with no joint data, the pilot to check stated against real validity, `bayes_eb_cp` from $n \ge 250$.

### 5.6 v1 re-measured combined r and coverage

The simulation of section 5.5 was rerun with only the sources the v1 default readers can fuse (`scripts/sim_fusion.py`, scenario `v1`, rows in `docs/fusion_sim_v1.csv`). `FusedReader` combines `text-bigfive` and `text-mbti`. Music has no tested Big Five link, and v1 has no sensor reader, so neither is a source. The measured scenario of section 5.5 still carried hypothetical music and sensor sources, and it scored people who had only those. Here everyone scored has text: 65% of simulated people, of whom 85% also have enough for the MBTI reader. Validities are read from the model cards at run time: `text-bigfive` refuses openness and has $r$ 0.097, 0.095, 0.094, 0.158 for C, E, A, N (the closed-form conversion of the tweets AUCs); the MBTI letters give 0.197, 0.125, 0.186, 0.115 for O, C, E, A through the links. Stated validities equal the truth, the method is `bayes` with no joint data, and everything else (error correlation 0.40 between the two text sources, 20 seeds of 3,000 test people) is as in section 5.5. The rerun of the old measured scenario still gives 0.150 and 0.802.

| quantity | O | C | E | A | N | mean | source |
|---|---|---|---|---|---|---|---|
| fused $r$, zero-data `bayes` | 0.196 | 0.146 | 0.172 | 0.121 | 0.148 | 0.157 | `fusion_sim_v1.csv`, scenario `v1` |
| 80% coverage | 0.803 | 0.800 | 0.803 | 0.802 | 0.801 | 0.802 | same |
| mean interval width (z) | 2.54 | 2.55 | 2.54 | 2.55 | 2.55 | 2.54 | same |
| skill | +0.017 | +0.009 | +0.014 | +0.007 | +0.011 | +0.012 | same |
| sources | MBTI N/S only | text + J/P | text + I/E | text + T/F | text only | | |

Across seeds the per-trait $r$ has an sd of 0.015 to 0.025, and the mean over traits 0.007. The mean of 0.157 is a little above the 0.150 of section 5.5 because people with only music or sensors, whose fused $r$ was low, are no longer scored. Overall, v1 fusion is about as accurate as each trait's best single source (`single` gives 0.153). The second source adds a little where both exist, for C (0.139 to 0.146) and A (0.109 to 0.121), and nothing measurable for E (0.173 to 0.172). Openness rests entirely on the inferred MBTI N/S letter, and neuroticism entirely on text Big Five. Coverage is 0.80 on every trait, and the intervals are 2.54 z wide against 2.56 for the interval that fits anyone, so the fused number moves a person only a little from 0.

A sensitivity row follows. The distilled reader of section 3.10 failed its pre-registered rule and is not the default, so this row describes a change that has not been made. Scenario `v1_distilled` replaces `text-bigfive` with `text-distilled` for non-forum text at the AUCs its card states for tweets (measured on all 294 PAN 2015 people): $r$ 0.198, 0.102, 0.251 for E, A, N. That reader refuses O and C on tweets (AUC 0.473 and 0.476), so those traits fall back to the MBTI letters alone.

| quantity, `v1_distilled` | O | C | E | A | N | mean |
|---|---|---|---|---|---|---|
| fused $r$ | 0.193 | 0.124 | 0.219 | 0.131 | 0.251 | 0.184 |
| 80% coverage | 0.805 | 0.802 | 0.802 | 0.797 | 0.806 | 0.802 |
| mean interval width (z) | 2.54 | 2.55 | 2.52 | 2.55 | 2.52 | 2.54 |
| skill | +0.018 | +0.007 | +0.024 | +0.008 | +0.030 | +0.017 |

The swap would raise E (0.172 to 0.219) and N (0.148 to 0.251), lower C (0.146 to 0.124, where it gives up the text source), and leave O and A within seed noise. The mean would go from 0.157 to 0.184 and coverage would stay at 0.80. The N gain rests on one tweets AUC (0.614 [0.551, 0.677]) and the E gain on another whose interval reaches 0.512.

Both rows are still simulations. They assume the stated validities are right, the error correlation between the two text sources is 0.40, and each person's validity is the whole-history tweets number (the real readers scale it down for short texts and mix it over kinds of writing). Nothing here has been checked on people who have a quiz and several sources.

## 6. Results, v1

Every slice that has been run, with its number as the source doc gives it. "Survives" means Holm-adjusted $p < 0.05$ within the family named in that doc. Every `predict` number uses the per-repeat average of section 2.1; the slices first run before that 1 Nov change (quiz timing, phone, body, Steam, sensors pooled) were re-run with it on 5 Nov, and each source doc lists its old numbers.

| slice | data, n | method | result | source |
|---|---|---|---|---|
| The ruler: reliability | IPIP norms, 597,365 | Cronbach's alpha | O 0.80, C 0.82, E 0.90, A 0.84, N 0.87 | `quiz.md` |
| The ruler: 4-item short form | held-out half, 298k | r with 10-item score | 0.91, 0.92, 0.95, 0.92, 0.94 (part-whole); against the other 6 items 0.72, 0.72, 0.83, 0.74, 0.81 | `quiz.md` |
| Quiz response timing | IPIP, random 20,000 | `predict`, 100 perms | O 0.126, C 0.094, E 0.077, A 0.082, N 0.121; every trait survives; weak (r 0.13 is 1.7% of variance) | `quiz.md` |
| MBTI forum, leak only | PersonalityCafe, 8,675 | 21 leak counts | AUC 0.879, 0.898, 0.894, 0.867 (I/E, N/S, T/F, J/P): what reading the answer off the page gives | `text.md` |
| MBTI forum, leak stripped | 8,675 | bag of words, strict strip | AUC 0.773, 0.772, 0.878, 0.694 | `text.md` |
| `text-mbti` reader, in domain | 1,735 held-out test people | saved model | AUC 0.787, 0.778, 0.878, 0.709 | `readers_text.md` |
| `text-mbti` reader, off forum | inferred via Essays | section 3.6 | AUC 0.61, 0.62, 0.62, 0.62, untested with MBTI labels; tweet spot checks 0.58, 0.54, 0.56, 0.48 | `readers_text.md` |
| `text-bigfive` reader, in domain | Essays, 494 held-out | saved model | AUC O 0.682, C 0.573, E 0.594, A 0.584, N 0.585 | `readers_text.md` |
| `text-bigfive` on vlogs | 404 vloggers, observer-rated | held-out domain | mean AUC 0.590: O 0.472, C 0.611, E 0.560, A 0.701, N 0.608; r A +0.40, N +0.27, C +0.22 | `text_crossdomain.md` |
| `text-bigfive` on tweets | PAN 2015, 294, self-report | held-out domain | mean AUC 0.544; only N clears chance alone (0.571, r +0.22) | `text_crossdomain.md` |
| `text-bigfive` on sitcom lines | FriendsPersona, 711 scenes | held-out domain | mean AUC 0.482, no trait above the null | `text_crossdomain.md` |
| Pooled text recipes | 4 domains, leave one out | 5 recipes | best pooled (style features) 0.514 against 0.539 for Essays only on non-Essays domains; artifact unchanged | `text_crossdomain.md` |
| LLM rater, Claude Haiku 4.5 | 150 each of Essays, PAN, vlogs | zero-shot, `b5-ipip50-v1` | held-out mean AUC 0.618 vs bag of words 0.558, $\Delta$ +0.060 [0.025, 0.093]: REPLACE; combined +0.050 [0.031, 0.069] | `llm_rater.md`, results JSON |
| LLM rater, local `qwen3:1.7b` | 17 people | same | stopped on the disk rule; no verdict; 15 of 16 valid replies were 3.0 on every trait | `llm_rater.md` |
| Distilled text reader (`mix` student) | pool of 2,000 forum users rated by Haiku 4.5; test on the same 150 per set | ridge on TF-IDF and MiniLM, pre-registered | PAN + vlogs 0.586 vs mark 0.588, three sets 0.583 vs 0.596: FAIL; on every test person +0.030 [+0.010, +0.051] over bag of words on PAN + vlogs (exploratory) | `distill.md`, `distill_results.json` |
| Phone (StudentLife) | 46 students | `predict`, 14 features | O +0.32, C +0.02, E +0.24, A $-0.12$, N $-0.29$ (null about $-0.19$); nothing survives Holm (best 0.17 within the dataset, 0.35 over both) | `phone_body.md` |
| Body (LifeSnaps, Fitbit) | 52 adults | `predict`, 15 features | O $-0.19$, C +0.07, E +0.24, A $-0.01$, N $-0.22$ (null about $-0.19$); nothing survives (best Holm 0.20) | `phone_body.md` |
| Steam profiles | 149 players | `predict`, 14 features | r from $-0.18$ to +0.14 across three feature sets, N about $-0.17$ in each (null about $-0.12$); nothing survives (best Holm p 0.75 over 15 rows) | `play.md` |
| Sensors pooled | StudentLife + LifeSnaps, 92 | 8 harmonised features, pre-registered | pooled E $-0.129$ vs null $-0.158$ ($p$ 0.337); transfer E +0.035 and +0.082; E fails by its rule | `sensors_pooled.md` |
| Sensors pooled, single pairs | 92 | meta-analysis, 9 pairs | activity timing vs E $\rho$ +0.30 [0.09, 0.48], Holm $p$ 0.053; routine regularity vs C +0.24, Holm 0.21; bedtime vs O does not replicate | `sensors_pooled.md` |
| Music traits | 304 ListenBrainz listeners | split-half retest | 0.720 to 0.811; no Big Five link tested | `reader_music.md` |
| Fusion | simulation, measured validities | zero-data `bayes` | $r$ 0.150, coverage 0.802 (simulation) | `fusion.md` |
| Fusion, v1 defaults | simulation, card validities, text sources only | zero-data `bayes` | $r$ O 0.196, C 0.146, E 0.172, A 0.121, N 0.148 (mean 0.157); coverage 0.80 on every trait; width 2.54 z | `fusion_sim_v1.csv` |
| Fusion, sensitivity: distilled for non-forum text | same | same | $r$ 0.193, 0.124, 0.219, 0.131, 0.251 (mean 0.184); coverage 0.80; not a verdict | `fusion_sim_v1.csv` |
| Feature record linkage | 3,000 forum users, half-split | stylometric matching | right account first for 36% with the opt-in word rates, 17% without (chance 0.03%) | `features.md` |

Per dataset for the Haiku run, computed here as the mean over the five traits of the per-trait AUCs in the results CSV: PAN 2015, LLM 0.557 against bag of words 0.508; vlogs 0.679 against 0.607; Essays (in domain for bag of words, outside the rule) 0.620 against 0.605. Per trait on PAN the LLM's agreeableness AUC is 0.458, below chance, and openness is 0.530.

Across slices, text carries the only clear signal, and it shrinks sharply when the kind of writing changes; openness, the strongest trait in domain, does not travel. Phone, wearable and Steam data show nothing that survives correction at $n \approx 50$ to 150. The zero-shot LLM is the one method that beat the trained text model on writing it never saw. A small student distilled from it on forum writing kept part of that lead on vlogs and little on tweets or essays, and failed its rule.

## 7. What is not claimed

- No reader has been tested on ordinary people against their own questionnaire. Every validity here is from a public research dataset. The study that would test this (PREREG H1 to H4) has not been run, and POWER.md section 9 recommends it wait for a better text model.
- Text Big Five on tweets is near chance: AUC 0.54 for C, E and A, each 95% interval including 0.5, and 0.57 for N. Reddit has not been measured at all; it borrows the tweet numbers. Openness is refused off essays.
- Text MBTI off the forum is an inference from a cross-construct test on student essays, with no MBTI-labelled writing from elsewhere. Its tweet spot checks (I/E 0.58, N/S 0.54, T/F 0.56, J/P 0.48, 294 people, wide intervals) all sit below the inferred 0.61 to 0.62.
- PANDORA (about 1,600 Reddit authors with continuous Big Five) is pending: the loader and the retraining path exist and the access request has not been made. The gate for the main study is $r \ge 0.20$ on held-out Reddit authors for at least three traits, with tweets not below about 0.12.
- The Haiku result may be contaminated. Essays, PAN 2015 and the vlogs were public long before the model's training cutoff, and in the Essays and vlog files labels sit on the same row as the text. None of the 450 reasons names a dataset or a score, which is suggestive and proves nothing. A clean check on unseen text, a paraphrase test and recall probes are designed and not run (`llm_rater.md`). The result is also from an API model, which the product cannot use.
- Sample sizes are small where the non-text sources are concerned: 46, 52, 92 and 149 people. At $n \approx 50$ a true $r$ of 0.3 is detected only about half the time, and nothing failing to survive there rules out a modest real effect.
- The samples are narrow. The IPIP norms are mostly US, UK, Canadian and Australian adults who took a free online test; Essays are University of Texas students from 1997 to 2004; StudentLife is Dartmouth students in 2013; LifeSnaps adults in four European countries; Steam players mostly young and in China; the music references are ListenBrainz enthusiasts. Validities and norms may not hold for other populations.
- Self-report and observer labels are different targets. The vlogs and FriendsPersona are rated by strangers, Essays and PAN by the people themselves, and agreement between the two is moderate.
- The distilled on-device reader failed its pre-registered rule and is not a default. Its sensitivity row in section 5.6 shows only what the swap would change in simulation. It was trained on Claude Haiku outputs, and Anthropic's commercial terms on using outputs to build competing models have to be checked before any commercial use.
- Music has no tested link to the Big Five and is described, never fused into a personality estimate.
- Fusion has never seen a person with several sources and a quiz. Its numbers are simulations whose error correlations and missingness are assumptions.
- The quiz is the target, with its own noise. The 4-item form caps any validity at about 0.91 to 0.95 of its full-scale value, and retest reliability is unmeasured.
- The derived feature record is linkable to public accounts (section 6, last row), so it is treated as personal data.

## References

- Barber, R. F., Candes, E. J., Ramdas, A. and Tibshirani, R. J. (2021). Predictive inference with the jackknife+. Annals of Statistics, 49(1).
- Biel, J.-I. and Gatica-Perez, D. (2013). The YouTube lens: crowdsourced personality impressions and audiovisual analysis of vlogs. IEEE Transactions on Multimedia, 15(1).
- Breiman, L. (1996). Stacked regressions. Machine Learning, 24.
- Gjurkovic, M. et al. (2021). PANDORA talks: personality and demographics on Reddit.
- Goldberg, L. R. (1992). The development of markers for the Big-Five factor structure. Psychological Assessment, 4(1).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2).
- McCrae, R. R. and Costa, P. T. (1989). Reinterpreting the Myers-Briggs Type Indicator from the perspective of the five-factor model of personality. Journal of Personality, 57(1).
- Pennebaker, J. W. and King, L. A. (1999). Linguistic styles: language use as an individual difference. Journal of Personality and Social Psychology, 77(6).
- Rangel, F. et al. (2015). Overview of the 3rd Author Profiling Task at PAN 2015.
- Stachl, C. et al. (2020). Predicting personality from patterns of behavior collected with smartphones. PNAS, 117(30).

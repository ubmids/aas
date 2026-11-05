# AAS: the models and the maths

This document defines every model in the AAS engine as the code implements it (`engine/src/model/`), states the objective each one optimises, gives the reason for each choice, and puts the measured result next to it. Results come from the bench report (`npm run report`, written to `bench/out/REPORT.md`) on 13 real people. Numbers marked "pooled" count every item once across people; everything else is the median over people with a 95% bootstrap interval over people.

## 1. Data and notation

A person's export is a time-ordered list of plays $p_1, \dots, p_N$. Each play has a start time $t$ (Spotify records the end time; the start is end minus milliseconds played), the milliseconds played $m$, the track, the artist $a$, how it started (`reason_start`: clickrow, playbtn, trackdone, fwdbtn, backbtn, appload, remote), how it ended (`reason_end`), the shuffle flag, the platform string and the connection country. Podcast and audiobook plays are kept only as times, for the silence and rhythm reading in eras.

A play is a skip when $m < 30\,\text{s}$, Spotify's own threshold for a stream. Every other play is a proper play. A session is a maximal run of plays in which each play starts at most 30 minutes after the previous one ended.

Local time. Each connection country maps to its main IANA time zone, and the offset at time $t$ is read through `Intl.DateTimeFormat`, so local hours follow daylight saving time. Plays with an unknown country use the person's most common country. Countries with several zones (US, Canada, Mexico, Brazil, Australia, Russia, Indonesia) read as their main zone, which is wrong by up to several hours for people elsewhere in those countries.

## 2. The bench protocol

Each person is split by time. The engine sees everything before a cutoff $T$ and is scored on the 8 weeks after it. Anything with a free parameter is tuned on a validation slice made of the last 8 weeks before $T$, with the model fitted on everything before that slice. The held-out weeks are never used for tuning.

Four things learn across people (the taste map for new loves, the popularity counts, the new-loves rankers, and the model of whether a new artist sticks). Each uses leave one person out: the person being scored never contributes to the map, the counts or the ranker they are scored with.

The bench has 13 people with Spotify extended histories published on GitHub. Identifying fields are stripped, and the data is never redistributed. Thirteen people is small. The report therefore attaches a bootstrap interval over people to every headline number and runs a paired sign test against each baseline, adjusted for multiple comparisons (section 16). With 13 people the sign test needs at least 10 of 13 people to agree before it can reach $p < 0.05$, so a real but small effect can fail to show.

A second bench (section 17) has 343 public ListenBrainz listening histories, 304 of them long enough for the same time split. It has no play lengths, skips, start reasons, shuffle, device or country, so only the models that do without those can be tested on it. There, every learned thing is fitted on four of five folds of people and scores the fifth, which keeps the rule that a person never shapes what scores them.

## 3. The taste map from co-listening (v0)

The map places each artist as a unit vector in $\mathbb{R}^{48}$ so that artists played near each other inside sessions sit close together. It follows Levy and Goldberg (2014), which shows that word2vec-style embeddings approximate a factorisation of the shifted PMI matrix. Here the factorisation is done directly.

Tokens. For each session, take the proper plays in order and collapse consecutive plays of the same artist into one token. Sessions with fewer than two tokens are dropped. The vocabulary is every artist with at least 3 proper plays that appears in at least 2 sessions.

Co-occurrence. For tokens $i < j$ in the same session with $j - i \le 5$, add $1/(j - i)$ to $C_{a_i a_j}$ and to $C_{a_j a_i}$. Let $r_a = \sum_b C_{ab}$ be the row sums.

PPMI with context smoothing. With $c_b = r_b^{0.75}$ and $Z = \sum_b c_b$,

$$A_{ab} = \max\left(0,\ \log \frac{C_{ab}\, Z}{r_a\, c_b}\right) = \max\left(0,\ \log \frac{P(b \mid a)}{P_{0.75}(b)}\right).$$

The 0.75 power raises the weight of rare contexts so that a few very popular artists do not dominate every row (the same smoothing word2vec uses for negative sampling). The matrix is sparse and almost symmetric; it is not exactly symmetric because the smoothing applies only to the context side.

Factorisation. The top 48 eigenpairs $(\lambda_k, u_k)$ come from a randomised subspace iteration (Halko, Martinsson and Tropp 2011): a Gaussian test matrix with 64 columns, three power iterations with re-orthonormalisation, then an exact eigendecomposition of the projected $64 \times 64$ matrix, symmetrised. Artist $a$ is placed at

$$x_a = \frac{U_{a,\cdot}\, \Lambda^{1/2}}{\lVert U_{a,\cdot}\, \Lambda^{1/2} \rVert},$$

with negative eigenvalues clipped to zero. Similarity between artists is the cosine $x_a^\top x_b$. Unit rows mean that the centroid of a set of artists (a weighted mean, renormalised) is a direction on the sphere, and nearness to it is again a cosine.

Why this map. It is built from how people actually sequence music inside a sitting, which captures mood as well as genre. It needs no external data. Read by eye, the neighbours of rock, hip hop, Bollywood, J-pop and lo-fi artists are their genre and mood mates, even from 13 people. Its weakness is coverage: it can only place artists someone in the bench played.

## 4. The ListenBrainz map and the blend

ListenBrainz publishes similar-artist lists computed from co-listening in its users' sessions. For a set of seed artists (those with at least 3 proper plays in the allowed people's training weeks), the build takes each seed's top 100 list, and then the list of every artist named on a seed's list (two hops). Each list entry $(a, b, s)$ adds $s / s_{\max}(a)$ to both $C_{ab}$ and $C_{ba}$, where $s_{\max}(a)$ is the top score on $a$'s list. An artist becomes a point if it has its own list or appears on at least two lists. The same PPMI and eigenvector steps give a 64-dimensional map. For new loves, each person gets a map built only from other people's seeds; this map has about 69,000 artists.

The blend places an artist at $\big[\sqrt{w}\, x^{(0)}_a;\ \sqrt{1 - w}\, x^{(\text{LB})}_a\big]$, renormalised, with $w = 0.5$. For artists on both maps, the blended cosine is the $w$-weighted mix of the two cosines. An artist on only one map keeps that half.

Result. Of the artists people went on to love, the share placed on the map rose from 0.31 to 0.65 on the held-out weeks and from 0.39 to 0.65 on the rolling windows. Coverage was the goal of this map, and it was met. The ListenBrainz map alone did worse than v0 in the game, and the blend did slightly better than v0 (section 7).

## 5. Finish or skip

The model predicts, at the start of each play, the probability that it will be a skip. Features are computed causally: the history is walked in order, and each play sees only the plays before it plus how it started, which is known when it starts.

Let $g$ be the person's skip rate so far. The track's skip rate is shrunk towards the artist's, and the artist's towards $g$, with pseudo-counts:

$$\hat s_{\text{artist}} = \frac{k_a + 5g}{n_a + 5}, \qquad \hat s_{\text{track}} = \frac{k_t + 3\,\hat s_{\text{artist}}}{n_t + 3},$$

where $k$ counts skips and $n$ counts plays so far. The 21 features are $\hat s_{\text{track}}$, $\hat s_{\text{artist}}$, $\log(1 + n_t)$, $\log(1 + n_a)$, the log days since the track and since the artist were last played, an exponentially weighted recent skip rate ($\rho \leftarrow 0.97\rho + 0.03\,[\text{skip}]$), whether the previous play in the session was a skip, the session's skip rate so far, the log position in the session, whether the previous play was the same artist, the shuffle flag, the local hour as $\sin$ and $\cos$ of $2\pi h/24$, and one-hot `reason_start` over seven values.

The model is an L2-regularised logistic regression on standardised features (intercept unpenalised, $\lambda = 1$), fitted by Newton's method (iteratively reweighted least squares). It minimises

$$-\sum_i \big[y_i \log \sigma(w^\top z_i) + (1 - y_i)\log(1 - \sigma(w^\top z_i))\big] + \tfrac{\lambda}{2}\lVert w_{1:} \rVert^2,$$

on the last 60,000 training plays at most. The baseline is the artist's shrunk skip rate on its own.

Result. AUC 0.848 [0.741, 0.861] against 0.617 [0.584, 0.649], better for 13 of 13 people (Holm-adjusted $p = 0.009$ across the 35 comparisons of section 16). It is also well calibrated on held-out plays: pooled expected calibration error 0.010 against 0.020 for the baseline, and Brier score 0.109 against 0.176, lower for every person. The largest single signal is `reason_start`: a play the person chose is rarely skipped, a play that arrived by itself often is. Removing it costs more AUC than any other feature, and it is known before the play begins, so it does not leak the outcome.

## 6. The next eight weeks' artists

Each artist gets a decayed play count,

$$S_a(T) = \sum_{\text{proper plays of } a} \exp\!\Big(-\ln 2\, \frac{T - t}{h}\Big),$$

with the half-life $h$ chosen per person from $\{3, 7, 14, 30, 60, 120, 365, 3650\}$ days by NDCG@10 on the validation slice. The forecast is the top 10 artists by $S_a(T)$. Quality is NDCG@10 against the held-out play counts, with gain $\log_2(1 + \text{count})$ and discount $1/\log_2(i + 2)$. The baselines are all-time counts and the last 4 weeks' counts.

Result. NDCG@10 0.813 [0.684, 0.919]. Against all-time counts (0.801) the decay wins for 7 people and loses for 4; against the last 4 weeks (0.711) it wins for 9 and loses for 3. Neither difference survives the statistics. People mostly keep playing what they already play, and every reasonable recency weighting finds that.

## 7. The game

The game tests whether the engine can tell a person's real music from plausible music. One round takes a track the person really played (properly) in the held-out weeks and four decoy tracks that the person never played, drawn from tracks with the same popularity in the bench (played by the same number of other people). The engine scores the five artists and wins the round if the real one scores highest. Up to 300 rounds per person, with a fixed seed so every scorer sees the same rounds. Chance is 0.2.

The old scorer is the decayed artist count $S_a(T)$. The map scorer adds nearness to recent favourites:

$$\text{score}(a) = \log\big(1 + S_a(T)\big) + w \cdot \max_{f \in F} x_a^\top x_f,$$

where $F$ is the 30 most played artists of the last 8 weeks that are on the map. An artist not on the map gets the person's average nearness, estimated over every seventh map artist, so that being off the map is neither a reward nor a penalty. The weight $w$ is chosen from $\{0, 0.5, 1, 2, 4, 8\}$ on a validation game inside the training period.

Result. The old scorer gets 0.687 [0.635, 0.820]. It beats chance for 12 of 13 people, which is significant before the multiple-test adjustment ($p = 0.003$) and just misses after it ($0.072$). The map adds a little on average: 0.733 with v0 and 0.742 with the blend, with the blend ahead of the old scorer for 7 of 13 people and behind for 4. Neither gain survives the statistics. The rounds where the real track is by an artist the person had never heard before are the harder test, and there every scorer is below chance (v0 0.179, blend 0.179, chance 0.200). The engine recognises a person's taste through artists it has already seen them play; it cannot yet say which unheard artist is theirs.

## 8. When you listen

The listening clock is a distribution over the 168 hours of the week (local time):

$$\pi_h \propto \alpha + \sum_{\text{plays in hour } h} \exp\!\Big(-\ln 2\, \frac{T - t}{h_c}\Big),$$

with $\alpha = 1$ (a Dirichlet prior) and the half-life $h_c \in \{30, 90, 365, \infty\}$ days chosen by validation log-likelihood. Quality is bits per play saved over a uniform clock on the held-out plays, $\frac{1}{n}\sum \log_2(168\, \pi_{h(t)})$.

Result. 0.123 [-0.065, 0.259] bits per play. The clock beats a uniform week for 8 people and loses for 5. Neither the tuned half-life nor local time beats its alternative. People's weekly rhythm in the next two months is only loosely like their rhythm before. The app shows it as a description of past listening.

On the 304 ListenBrainz listeners (section 17), with UTC hours because that data has no time zone, the clock saves 0.456 [0.405, 0.501] bits per play over a uniform week and beats it for 268 of 304 people. A person's zone is fixed for most of their history, so a personal clock read in UTC predicts as well as one in local time apart from daylight saving shifts. That bench is a different population (people who log every play to ListenBrainz, many of them for years), so the size of the gain need not carry over to the Spotify listeners; that the clock carries real information about the next 8 weeks does.

## 9. New artist or familiar

A logistic regression predicts whether a play is an artist the person has never played before, from causal features: an exponentially weighted recent rate of new artists ($\rho \leftarrow 0.98\rho + 0.02\,[\text{new}]$), whether the previous play was new, the log minutes since the previous play, the shuffle flag, and one-hot `reason_start`. Same fitting as section 5.

Result. AUC 0.757 [0.650, 0.890] against 0.5, better for 12 of 12 people with enough data (Holm-adjusted $p = 0.017$).

## 10. Selves

A self is a recurring way of listening. The music and the way of listening define it; the surroundings (time, place, device) are kept out of the definition so they can be used to test it.

Session vector. For each session with at least 3 plays: the play-time-weighted mean of its artists' map vectors, $\bar x = \sum w_i x_{a_i} / \sum w_i$ with $w_i = \min(m_i, 5\,\text{min}) + 1\,\text{ms}$ (plays off the map are ignored); the spread $1 - \lVert \bar x \rVert$, which is 0 when every artist points the same way; the share of plays the person picked (clickrow, playbtn, backbtn among plays with a known start); the skip share; the repeat share; and the log number of plays. The 48-dimensional mean is reduced to 8 dimensions by PCA, and the resulting 13 numbers are standardised.

Mixture. A Gaussian mixture with full covariances is fitted by expectation maximisation, started from k-means++ centres, with $10^{-2}$ added to each covariance diagonal and a stopping tolerance of $10^{-4}$ per session in log-likelihood. The number of components $k$ is chosen from $2$ up to $\min(7, \lfloor n/40 \rfloor)$ by a penalised BIC,

$$\text{BIC}_\pi(k) = -2 \log L_k + \pi \cdot p_k \log n, \qquad p_k = k\Big(d + \frac{d(d + 1)}{2}\Big) + k - 1,$$

where $d = 13$ and $\pi$ multiplies the usual penalty. A component whose sessions fall in fewer than 3 distinct weeks is removed and the weights renormalised, repeatedly, because a self has to recur.

Choosing the penalty. Plain BIC ($\pi = 1$) chose the cap of 7 for almost everyone. The penalty is now chosen per person: for each $\pi \in \{1, 2, 4, 8, 16\}$, fit on the sessions before the validation slice, keep fits with 3 to 6 surviving selves, and pick the one whose mixture gives the validation sessions the highest mean log-likelihood. Then refit on all training sessions with that $\pi$. The median person now has 5 selves.

Test. A multinomial logistic regression (L2 $10^{-2}$, gradient descent with momentum) predicts a held-out session's self from its surroundings alone: 8 three-hour bins of local start time, 7 weekdays, 6 device classes, the shuffle share, and the person's 3 most common countries, all one-hot except shuffle. It is fitted on the last 8,000 training sessions, labelled by the mixture. Accuracy is compared with always guessing the most common self. Accuracy is not comparable across different numbers of selves (fewer selves make guessing easier), so the fair measure is the information the surroundings carry about the self: the log-loss of the self frequencies minus the log-loss of the model, in nats per session.

Result. Accuracy 0.482 [0.341, 0.573] against 0.256 for the majority self; log-loss 1.450 against 1.529. The mean improvement is clearly above zero (bootstrap interval for the mean log-loss gain [0.020, 0.142]), but the per-person sign test (9 of 13) does not survive the adjustment. The information gain was 0.036 nats with the old 7 selves and 0.036 with 5, so fewer selves cost nothing.

## 11. Eras

An era is a season-sized stretch in which listening holds steady. Each calendar week becomes a row: the week's taste centroid reduced to 4 PCA dimensions, the share of each self, the log number of plays, the share of plays in each 6-hour quarter of the day, the podcast share, and a silent flag. A week with no plays carries the previous week's taste, self mix and hours forward and sets the flag, so silence is part of the record. Rows are standardised.

Change points. Segment boundaries $0 = \tau_0 < \tau_1 < \dots < \tau_m = n$ minimise

$$\sum_{j=1}^{m} \sum_{i = \tau_{j-1}}^{\tau_j - 1} \lVert y_i - \bar y_{j} \rVert^2 + \beta\, m, \qquad \beta = \pi\, d \log n,$$

subject to every segment lasting at least 10 weeks. This is the Gaussian mean-shift cost with unit variance, solved exactly by dynamic programming over prefix sums ($O(n^2)$ in weeks). The penalty scale $\pi \in \{0.5, 1, 2, 4\}$ is chosen on the validation slice together with the forecast below; ties keep $\pi = 1$, which gives about one to three eras a year.

Era-aware forecast. Artists are scored with plays in the current era at weight 1 and earlier plays at weight $\gamma$, both decayed: $\sum w(t) \exp(-\ln 2\,(T - t)/h)$ with $w = 1$ inside the era and $\gamma \in \{0, 0.05, 0.2, 0.5, 1\}$ before it.

Result. The era-aware forecast ties plain decay for 9 of 13 people (NDCG@10 0.813 against 0.813), because validation usually picks $\gamma = 1$, which is plain decay. Using the current era alone loses to decay for 8 people. Eras read well as a story (the boundaries line up with silences, volume jumps and turnover of favourites), and they are shown that way. They do not improve the forecast.

## 12. New loves

A new love at cutoff $T$ is an artist never played before $T$ that the person then plays properly at least 10 times on at least 3 different days within the next 8 weeks. Ranking quality is recall@$K$: the share of a window's new loves among the top $K$ unheard artists. Every new love in the window counts, including ones the ranker could not see. Windows are the held-out 8 weeks and up to 12 rolling 8-week windows inside each person's training period (each needs 26 weeks of history before it). There are 52 held-out and 407 rolling new loves. Rankers learn from other people's rolling windows only.

The first ranker scored every unheard artist on the map with a logistic regression over nearness to recent, all-time and drift-extrapolated taste, nearness to the nearest favourite, fit to the person's typical jump size, and popularity. It never clearly beat ranking by popularity, on any map.

The two-stage ranker. Stage one proposes candidates. Let $F$ be the 30 most played artists in the 8 weeks before $T$, each with weight

$$\omega_f = \frac{n_f}{\sum_{g} n_g} \cdot 2^{-(T - t^{\text{last}}_f)/14\,\text{days}}.$$

Every unheard artist on some $f$'s ListenBrainz list is a candidate, and so are the 300 unheard map artists nearest to any favourite. Stage two scores each candidate $c$ with a logistic regression over: the log number of favourites that list $c$; the weighted reciprocal-rank sum $\sum_f \omega_f / \text{rank}_f(c)$ (a weighted form of reciprocal rank fusion, Cormack, Clarke and Buettcher 2009); the best reciprocal rank; the log best rank; the weighted score sum $\sum_f \omega_f\, s_f(c)/s_f^{\max}$; how many lists in the map name $c$; how many other people played $c$ and how often; nearness to the nearest favourite on the leave-one-out ListenBrainz map; a flag for map-only candidates; the person's own share of earlier new loves that a favourite had listed, $(l + 5\bar l)/(n + 5)$, shrunk to everyone else's share; and that share times the reciprocal-rank sum. Training uses every candidate that became a love and 200 sampled candidates that did not, per window.

Result. Stage one covers 22% of rolling-window loves. Pooled rolling recall@50 is 0.054 for the ranker, 0.042 for the list evidence alone and 0.032 for popularity. Per person, the ranker beats popularity for 7 of 10 people with rolling loves and loses for 2, which is not significant ($p = 0.18$). Against the lists alone it wins 5 and loses 2. On the held-out weeks it ties popularity at 0.038. The ranker uses list evidence a little better than the raw lists do, but with 13 people it does not beat popularity in a way that holds up, and the low coverage of stage one caps it. Most new loves are not on any recent favourite's list at the moment the window opens. This forecast is therefore not shown to users. New loves are shown looking back instead (section 13).

## 13. Doors: how new loves came in

For every new love in a person's whole history (first play at least 12 weeks in, and at least 8 weeks before the end), the engine reads facts at its first play: how it started (picked: clickrow or playbtn; came on: trackdone or fwdbtn), the artist just before it in the same session, whether that artist was a recent favourite (top 30 by proper plays in the previous 8 weeks), whether that artist's ListenBrainz list names it and at what rank, whether any recent favourite's list names it and the best rank, whether the first play was a skip, and the days from the first play to the 10th proper play. Each love is then filed under one door, checked in this order: picked; came on after an artist that lists it; next to a favourite (a favourite lists it); came on by itself; out of nowhere.

Controls. A fact only explains a love if it is clearly more common for loves than for comparable artists that were not loved. For each love, the controls are up to 5 artists the same person tried and dropped (at most 2 proper plays ever) whose first play was nearest in time, within 60 days. For the list fact there is a second control: 5 never-played ListenBrainz artists named on about as many lists ($\text{round}(2\log_2 c)$ equal), read at the love's first play.

Test. For a fact $D$, the love rate is the share of loves with $D$, and the control rate is the mean over loves of the share of their controls with $D$. The ratio of pooled rates gets a bootstrap interval over people. A door counts as evidence when the ratio is at least 1.5, its 95% interval is above 1, and loves have the higher rate for most people with at least 3 loves.

Result (1,821 loves, 17,374 tried-and-dropped artists). Two doors pass. "Came on after an artist that lists it" is 2.40 times as common for loves [1.11, 3.55], and "next to a favourite" 2.47 times [1.56, 3.04]. Against never-played artists, a recent favourite lists 42% of loves and 8% of controls, higher for 12 of 12 people. Picking an artist yourself is not more common for loves across the bench (ratio 1.12), but for some people it clearly is, so each person's summary compares their loves with their own dropped artists first. A love's first play is a skip 0.39 times as often as a dropped artist's. The median love reaches 10 proper plays 15 days after it is first heard. Reading the paired sign tests strictly, the two doors that pass the ratio rule are consistent for 8 and 9 of 11 people, which does not survive a multiple-test adjustment; the never-played comparison does.

## 14. Will a new artist stick

Section 12 asks which of tens of thousands of unheard artists a person will come to love, and the answer does not beat popularity. This section asks a narrower question at a later moment: the person has just heard an artist for the first time, so will it become a new love? The question is the one a person can check against their own memory, and it is answerable.

Rows. Every artist a person plays for the first time is a row, if the first play is at least 12 weeks into the history and its 8-week outcome window ends inside the data. The label is $y = 1$ when the artist becomes a new love (at least 10 proper plays on at least 3 days within 8 weeks of the first play, as in section 12) and $y = 0$ otherwise. There are 26,401 rows in the training periods, of which 1,771 stuck (6.7%).

Horizons. A prediction is made at one of two moments: the end of the first session that contains the first play, or the end of the first 7 days. A feature may read only plays that start before that moment, and the recent favourites are taken at the start of the session that holds the first play. At 7 days, rows that already met the definition are left out, because they no longer need a guess (514 rows; 25,887 remain, 1,257 stuck).

Features after the first session:

| group | features |
|---|---|
| first play | skip indicator; $\log(1 + \text{seconds played})$, seconds capped at 600 |
| how it started | picked (clickrow, playbtn); came on (trackdone, fwdbtn); shuffle |
| ListenBrainz lists | the artist just before it in the session lists it; $1/\log_2(1 + \text{rank})$ on that list; a recent favourite lists it; $1/\log_2(1 + \text{best rank})$ over favourites; the artist is on ListenBrainz at all |
| favourites in the session | share of the session's other proper plays that were recent favourites |
| taste map | largest cosine to the 10 top favourites on the leave-one-out co-listening map (section 3), 0 when off the map; on the map |
| popularity | $\log(1 + \text{other people who played it properly})$; $\log(1 + \text{lists naming it})$ |
| the first session | $\log(1 + \text{plays of the artist})$; $\log(1 + \text{its proper plays})$; $\log(1 + \text{its distinct tracks})$; a track repeated; came back to it after another artist; $\log(1 + \text{session length})$; the session's skip rate |
| time of day | $\sin$ and $\cos$ of $2\pi h/24$ for the local hour $h$ |
| own base rate | $\operatorname{logit}\big((k_t + 20\pi)/(n_t + 20)\big)$, where $n_t$ counts the person's earlier new artists whose 8-week window ended before this first play, $k_t$ how many of them stuck, and $\pi$ is other people's rate |

After 7 days, six more: $\log(1 + \text{proper plays in the 7 days})$, the number of days with a proper play, $\log(1 + \text{distinct tracks})$, whether it was played in a later session, its skip share and its picked share.

Model. A logistic regression on standardised features with an L2 penalty of 1, fitted by Newton's method as in section 5. It is trained on other people's training-period rows (outcome window ended before their cutoff) and scored on the person's own rows, in two sets: every row in their training period, and the rows first heard in the 8 weeks before their cutoff, whose outcome falls in the held-out weeks.

Measures. AUC per person; average precision, the mean over artists that stuck of the precision at each one's rank, which is the better reading when only 4 to 7% of rows are positive; the Brier score and expected calibration error (section 16). Per-person medians are the primary reading. A pooled AUC over all rows also ranks people against each other: a person whose new artists stick more often scores higher overall, so pooled numbers are reported but not tested.

Checks for leakage. A random 1% of rows (614) were read a second time from the history cut at the horizon, and none differed. With labels permuted across all training rows, the per-person AUC is 0.529 [0.463, 0.583] after the session and 0.450 [0.358, 0.511] after 7 days, as it should be. With labels permuted only inside each training person, so that who sticks more overall survives, it is 0.582 [0.506, 0.613] and 0.643 [0.568, 0.689]: that much of the score can come from differences between people alone.

Result after the first session, training period, median AUC per person:

| model | AUC [95%] | average precision [95%] |
|---|---|---|
| this model | 0.773 [0.721, 0.812] | 0.287 [0.158, 0.407] |
| own base rate | 0.484 [0.437, 0.510] | 0.051 [0.035, 0.145] |
| first-play skip | 0.593 [0.553, 0.667] | 0.037 [0.023, 0.095] |
| popularity | 0.628 [0.577, 0.662] | 0.091 [0.067, 0.194] |

On the held-out weeks the model reaches 0.831 [0.766, 0.918], but only 6 people have at least 3 new artists that stuck and 3 that did not in those 8 weeks (50 stuck in all). Calibration is close: pooled ECE 0.007 in the training period and 0.020 held out. Pooled Brier is 0.054 against 0.061 for the base rate alone.

After 7 days, the model reaches 0.899 [0.868, 0.922] in the training period and 0.934 [0.725, 0.980] held out (6 people). The plain count of the first week's proper plays and days already reaches 0.880 [0.840, 0.904], so at 7 days the rest of the model adds about 0.02.

Statistics. After the first session the model beats the person's base rate for 12 of 12 people (Holm-adjusted $p = 0.017$). At 7 days it beats popularity for 12 of 12 ($p = 0.017$). Both survive. Against the first-play skip and against popularity after the first session it wins for 11 of 12 people ($p = 0.006$ before adjustment, 0.18 after), which does not survive 35 comparisons on 12 people. The gain at 7 days over the first week's counts (9 wins, 2 losses) is weak.

What carries it. Dropping one group at a time from the session model, the first session costs 0.051 [0.034, 0.068] of median AUC, and on its own it reaches 0.736. The ListenBrainz lists cost 0.009 [0.001, 0.016] and how it started 0.006 [0.001, 0.025]. Dropping the first play, the map, popularity, the time of day or the favourite share changes nothing measurable, because the first session already holds most of what they say. The own base rate leaves per-person AUC unchanged (it moves a person's whole level, not the order of their artists) but lifts pooled AUC from 0.747 to 0.816 and helps calibration. After 7 days the first week does the work: alone it reaches 0.884, and dropping it costs 0.149.

Boosted trees. As a check that the linear model was not leaving accuracy unused, gradient-boosted trees (Friedman 2001; 150 trees of depth 3 on features cut into 32 quantile bins, logistic loss, learning rate 0.1) were fitted the same way. They reached 0.764 after the session and 0.895 after 7 days, no better, so the logistic regression stays.

Reading it. Which artist someone will love next cannot be forecast from where they have been (section 12). Whether an artist they have just met will last is mostly decided in the first session: how much of it they play through, whether they repeat a track, whether they come back to it. Where it came from (section 13) adds a little on top. Examples from the held-out weeks, after the first session:

- An artist that came on after another artist the person played, 4 plays that session, all past 30 seconds, and a recent favourite lists it: the model gave it 66%, and it reached the definition within 2 days.
- An artist the person picked and played 10 times in the session: 95%, and it did not stick.
- An artist that came on by itself, 2 plays that session: 2%, and it stuck by day 13.

Without skips. The second bench (section 17) has no skips, play lengths, start reasons, shuffle or time zone, so the model there drops ten columns: the first play's skip and seconds, picked, came on, shuffle, the session's skip rate, the hour's sine and cosine, and at 7 days the week's skip share and picked share. On the Spotify bench the same reduced model reaches 0.762 [0.714, 0.781] after the first session against 0.773 for the full one, so those ten columns add 0.014 [0.001, 0.048] of median AUC (better for 10 of 12 people, $p = 0.039$, not after adjustment). At 7 days they add nothing (0.898 against 0.899). On the 304 ListenBrainz listeners the reduced model reaches 0.792 [0.784, 0.800] after the first session against 0.466 for the base rate and 0.618 for popularity (better than popularity for 283 of 303 people), and 0.811 [0.783, 0.836] on the held-out weeks for the 112 people with enough rows. Every comparison survives there. So the result does not depend on Spotify's skip data: what the person does in the first session carries it.

## 15. Measured traits

Traits replace inferred personality. Each is a number read from the history with its receipt, and a percentile against the other people.

| trait | definition |
|---|---|
| new share | share of proper plays going to artists first heard in the previous 28 days (after a 12-week warm-up) |
| spread | $1 - \lVert \text{mean of the unit map vectors of all proper plays} \rVert$ |
| old friends | in each 8-week stretch, the share of proper plays going to artists in the top 50 of the stretches 2 and 3 before (4 to 6 months earlier), averaged |
| finished | share of plays whose `reason_end` is trackdone |
| skip seconds | median seconds played before a skip ended by fwdbtn or unknown |
| clock hour | circular mean of local hour weighted by minutes played, read on a 4 am to 4 am day |
| night share | share of minutes between midnight and 5 am |
| clock spread | $1 - R$, where $R$ is the length of the minutes-weighted mean of unit vectors at each proper play's time of day on the 24 hour dial |
| clock regularity | length of the mean of the daily mean directions, over days with 5 or more proper plays (at least 30 such days); each day runs from 12 hours away from the person's overall centre of listening |
| turnover | mean share of the top 10 artists replaced between consecutive 13-week seasons |

Stability. A trait describes a person only if it is stable over time. Each person's training period is cut at its midpoint, the trait is measured on each half (the second half may use the first only to know which artists were already familiar), and the two halves are correlated across people with Pearson's $r$ and Spearman's $\rho$. A trait is shown when both are at least 0.5, shown with a "read it as now" flag when one is, and dropped when neither is.

Result. Stable: new share ($r = 0.72$), finished (0.86), skip seconds (0.99), clock hour (0.53, $\rho = 0.70$) and turnover (0.88). Flagged: spread and old friends. Dropped: night share. The bootstrap intervals are wide with 10 to 12 people per trait. Only skip seconds has both intervals above 0.5; new share, finished and turnover are clearly above zero. Old friends needed three definitions before one held. The first, "top 20 still played 6 months on", had $r = -0.01$. Across people the traits are not independent: old friends runs against new share and turnover ($\rho$ about $-0.8$), which suggests two underlying axes, exploring against staying, and when you listen.

Clock spread and clock regularity were added in round 3 because they do not depend on the time zone: a fixed offset turns every angle on the dial by the same amount, which leaves both lengths unchanged (the regularity day boundary turns with them), so they can be read from UTC times. On the Spotify bench they have $r = 0.61$ and $0.81$.

On the 304 ListenBrainz listeners (section 17) every trait that data can measure is stable with both intervals above 0.5: new share $r = 0.754$ [0.680, 0.819], spread 0.876 [0.841, 0.908], old friends 0.811 [0.748, 0.863], turnover 0.802 [0.755, 0.847], clock spread 0.720 [0.644, 0.783] and clock regularity 0.725 [0.652, 0.787], with Spearman intervals also above 0.5. Spread and old friends, flagged on 13 people, are therefore shown as stable. Finished, skip seconds, clock hour and night share need skips, play lengths or a time zone and keep their Spotify status.

How many dials. The six measurable traits were put through a factor check on the 301 ListenBrainz listeners who have all of them. Each trait is replaced by normal scores, the correlation matrix is decomposed, and Horn's parallel analysis (Horn 1965) keeps components whose eigenvalue beats the 95th percentile of eigenvalues from the same data with every column shuffled (1,000 shuffles). Two components are kept (eigenvalues 3.10 and 1.86 against 1.27 and 1.15 under shuffling; the third is 0.48 against 1.07), and every one of 200 bootstrap resamples also keeps two. Together they hold 83% of the variance. After varimax rotation (Kaiser 1958), with 95% intervals from 500 bootstrap resamples:

| trait | explore against stay | timing |
|---|---|---|
| new share | $-0.88$ [$-0.90$, $-0.85$] | $-0.08$ [$-0.17$, 0.01] |
| spread | $-0.79$ [$-0.83$, $-0.75$] | 0.02 [$-0.09$, 0.13] |
| old friends | 0.94 [0.92, 0.95] | 0.02 [$-0.06$, 0.09] |
| turnover | $-0.90$ [$-0.92$, $-0.87$] | 0.04 [$-0.03$, 0.11] |
| clock spread | 0.03 [$-0.04$, 0.10] | $-0.96$ [$-0.97$, $-0.95$] |
| clock regularity | 0.05 [$-0.02$, 0.12] | 0.96 [0.95, 0.97] |

The first dial holds: new share, spread and turnover load on one side and old friends on the other, and none of them loads on the second. The second dial is about time, but only in the form this data allows: how fixed a person's listening time is, from day to day and across the day. Clock hour, the time of day itself, cannot be read without a time zone, so whether it belongs to the same dial is untested here. The second dial also rests on two measures that correlate at $-0.86$, so it is close to one quantity measured twice. What the check shows is that how regular someone's listening time is has nothing to do with how much they explore ($|r| \le 0.12$ for every pair across the two groups).

## 16. Statistics used in the report

Bootstrap. People are resampled with replacement 4,000 times and the statistic recomputed; the interval is the 2.5% and 97.5% percentiles (Efron and Tibshirani 1993). People are the unit of resampling because plays within one person are not independent.

Sign test. For a model and a baseline, each person contributes the sign of their difference. Ties are dropped, and the number of positive signs is compared with a Binomial($n$, 1/2), two-sided and exact.

Multiple comparisons. The Spotify report makes 35 comparisons. Holm's step-down method (Holm 1979) adjusts the sign-test p-values so the chance of any false claim stays below 0.05. A result survives when its adjusted $p$ is below 0.05, the model wins more people than it loses, and the bootstrap interval for the mean difference is above zero. A comparison whose sign test is significant in the other direction, with the interval below zero, is reported as worse than the baseline. Round 3 is its own family: its 21 paired comparisons and 5 door comparisons are adjusted together.

Factor structure (section 15). Normal scores replace each value by $\Phi^{-1}(r/(n+1))$, where $r$ is its rank, which keeps the order and removes the pull of outliers. Parallel analysis compares each eigenvalue of the correlation matrix with the 95th percentile of the same eigenvalue when every column has been shuffled independently, which keeps each trait's spread and breaks every link between traits; components are kept in order while they beat it. Loadings are eigenvector times the square root of the eigenvalue, varimax-rotated when two or more are kept. Bootstrap intervals resample people, refit, match each refitted component to the original by the largest absolute congruence $\frac{a \cdot b}{\lVert a\rVert \lVert b\rVert}$, and flip its sign to agree.

Calibration. Predicted skip probabilities are put in 10 equal-width bins. The expected calibration error is $\sum_b \frac{n_b}{n} \lvert \bar p_b - \bar y_b \rvert$, and the Brier score is $\frac{1}{n}\sum (p_i - y_i)^2$.

What survives with 13 people: the skip model and the new-artist model, by every test; whether a new artist sticks, against the person's base rate after the first session and against popularity after 7 days; the game beats chance before adjustment; the skip model's calibration is better than its baseline's for every person. Selves from context, the new-loves ranker, the next-artists forecast, the clock and eras do not survive, and the report says so line by line.

## 17. Round 3: 343 ListenBrainz listeners

Data. 343 public ListenBrainz histories (bench/lbusers), found by following ListenBrainz's similar-users links out from ten long-standing public accounts and kept when they had 5,000 to 2 million listens, a year or more between first and last, and a listen in the last 120 days. 304 have at least 1,000 listens before their last 8 weeks and 200 in them, and get the same split as the Spotify bench. Median history 4.1 years, median 59,000 listens. Artists are keyed by MusicBrainz id where a listen carries exactly one (about 81% of listens). ListenBrainz records a listen once half the track or 4 minutes has played, so every listen counts as a proper play.

Folds. People are put in 5 folds by a hash of their pseudonym. The co-listening map for a person (section 3, vocabulary: artists played by at least 5 people in the other folds, about 33,000) is built from the other four folds. The stick model and the ranker are fitted on the other four folds. Popularity is counted over everyone's training weeks minus the person's own plays.

ListenBrainz lists. The Spotify bench's cache of similar-artist lists was extended to every person's 200 most played training artists (10,045 more lists, 38,946 in all). ListenBrainz computes these lists from its own users' sessions, and these people are its users, so their own listening is a small part of what the lists were built from. Two checks bound it. The stick model loses only 0.002 of median AUC without the list features. And if a person's own sessions were driving the list door, the people with the most listening would show it most strongly: a recent favourite lists 39% of the heaviest third's new loves against 7% of matched never-played artists (ratio 5.6 [4.9, 6.3]), and 51% against 11% for the lightest third (4.5 [4.0, 5.1]). The heavier third does show a somewhat larger ratio, which is what a leak would look like, but the lightest third, whose sessions weigh least in the lists, still shows a 4.5 times difference for 101 of 101 people. The door is real; part of its size may not be.

Results, with Holm across the round's 26 sign tests:

| question | model | baseline | people (wins/losses) | verdict | round 2, 13 Spotify people |
|---|---|---|---|---|---|
| next 8 weeks' artists, decay vs all-time (NDCG@10) | 0.706 [0.677, 0.736] | 0.604 | 218/72 | survives | no difference (7/4) |
| next 8 weeks' artists, decay vs last 4 weeks | 0.706 | 0.617 | 212/87 | survives | no difference |
| clock vs uniform (bits per play, UTC) | 0.456 [0.405, 0.501] | 0 | 268/36 | survives | no difference (8/5) |
| eras, era-aware vs decay | 0.706 | 0.706 | 67/76 (161 ties) | no difference | no difference |
| eras, current era only vs decay | 0.688 | 0.706 | 107/192 | worse than decay | no difference |
| new loves, two-stage ranker vs popular (rolling recall@50) | pooled 0.047 | pooled 0.024 | 132/24 (137 ties) | survives | no difference (7/2) |
| new loves, same, held-out weeks | pooled 0.036 | pooled 0.015 | 22/6 (156 ties) | weak | no difference |
| stick after the first session vs base rate (AUC) | 0.792 [0.784, 0.800] | 0.466 | 301/2 | survives | survives (12/0) |
| stick after the first session vs popularity | 0.792 | 0.618 | 283/20 | survives | before adjustment only (11/1) |
| stick after the first session, held-out weeks vs popularity | 0.811 [0.783, 0.838] | 0.607 | 104/8 | survives | before adjustment only (6/0) |
| stick after 7 days vs first-week counts | 0.884 [0.879, 0.890] | 0.860 | 257/46 | survives | weak (9/2) |
| stick after 7 days, held-out weeks vs first-week counts | 0.901 | 0.878 | 41/30 | weak | no difference (5/1) |

Doors without start reasons. "A recent favourite lists it" is 1.87 times as common for new loves as for tried-and-dropped artists [1.74, 2.01], higher for 255 of 298 people, and in its top 25 2.45 times [2.21, 2.73]; against never-played artists named on as many lists it is 5.3 times [4.9, 5.6], higher for 302 of 302 people. The round 2 result (12 of 12) holds. "Came on straight after an artist that lists it", read without knowing how the play started, is 1.42 times as common [1.29, 1.54], consistent across people (180/103) but under the 1.5 bar. The median new love here reaches 10 listens 7 days after it is first heard (15 on the Spotify bench).

New loves, ranked. Stage one covers only 18% of the loves in rolling windows, so the ranker's recall stays small in absolute terms: 4.7% of loves in its top 50, against 3.1% for the plain lists and 2.4% for popularity. With 293 people that difference is consistent enough to survive, where 13 could not show it. Most people have no love in any top 50 (137 ties), so the forecast is still not something to put in front of a person; what it shows is that list evidence used well is better than popularity.

What flipped with more people. Three results that were "no difference" with 13 people survive with 304: recent listening (decay) predicts the next 8 weeks' artists better than all-time listening, the clock beats a uniform week, and the two-stage ranker beats popularity on rolling windows. Two that were significant only before adjustment, or weak, survive: the stick model against popularity after the first session, and against the first week's play counts after 7 days. One went the other way: using only the current era is worse than plain decay. Eras still do not improve the forecast. The traits flagged with 13 people are stable with 304, and the explore against stay dial holds.

What could not be tested. Finish or skip, the selves, the game against matched decoys, local clock hour and night share, finished and skip seconds, and the doors that depend on how a play started. Those rest on the 13 Spotify people only.

## References

- Cormack, G. V., Clarke, C. L. A. and Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR.
- Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman and Hall.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5).
- Halko, N., Martinsson, P. G. and Tropp, J. A. (2011). Finding structure with randomness. SIAM Review, 53(2).
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. Psychometrika, 30(2).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2).
- Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis. Psychometrika, 23(3).
- Levy, O. and Goldberg, Y. (2014). Neural word embedding as implicit matrix factorization. NeurIPS.
- Roenneberg, T., Wirz-Justice, A. and Merrow, M. (2003). Life between clocks. Journal of Biological Rhythms, 18(1).
- Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics, 6(2).
- Ward, M. K., Goodman, J. K. and Irwin, J. R. (2014). The same old song. Marketing Letters, 25(1).

# The prediction game: pre-registration

Written 2026-11-05, before any of the games below was run. The code that runs them (bench/game.ts) is written against this document and commits after it. Anything decided after seeing a result is marked as such in the results section.

## Why a game

The engine's models are scored with AUC, NDCG and recall. Those are fine for comparing models, but a person cannot check them. A game turns a model into statements one person can check against their own listening: "of your next 10 tracks, these will be skipped", "of these three new artists, this one stays", "these five will be your top artists next week". The question this document asks is whether such statements beat a strong simple rule for most individuals, measured person by person. An average that is carried by a few people does not count.

Every game is played on held-out time. Each person's history is split where the existing benches split it: training is everything before their last 8 weeks, and the game is played inside those 8 weeks. Anything tuned is tuned on the 8 weeks before the cutoff (the validation slice) with a model fitted on what came before them. Any weight learned across people is learned without the person being scored: leave one person out on the 13 Spotify people, and the five hash folds of bench/lb3 on the 304 ListenBrainz listeners (the person's own fold is never in the fit), exactly as the existing benches do.

## The data

There are 13 Spotify people, the ones bench/load.ts keeps from bench/data/spotify_full. These histories carry skips, start reasons and shuffle. There are also 304 ListenBrainz listeners, the people bench/lb3 kept (a cached person file that is not marked skipped). These have no skips, so they play games 2 and 3 only.

## Game 1: "I'll call your next 10 skips" (Spotify, 13 people)

The person is about to hear 10 tracks in one listening session. Before the first one starts, the model marks each of the 10 as skip or play. A skip is a play under 30 seconds, as everywhere in the engine. The score is hits out of 10: the number of tracks marked correctly.

### Rounds

Sessions are runs of plays with gaps of at most 30 minutes. Every session is cut into consecutive blocks of 10 plays from its first play; the last block of a session may be shorter. A round is a full block of 10 whose plays all fall in the held-out weeks. A person plays every such round.

### What the model may know

All 10 calls are made at the start of the block. The model may use everything that happened before the block and the 10 tracks themselves (which track, which artist, whether a track repeats the previous one's artist, its position in the block). It may not use anything that is only known once a track in the block has started or ended. So the counters for track and artist skip rates, days since last heard, the recent skip rate and the session's skip rate so far are frozen at the block start. The skip status of the play just before the block is known. Start reason, shuffle and hour are those of the block's first play, used for all 10. This is stricter than the existing skip model, which updates after every play; the live model reads how each track was started, which reveals whether the one before was skipped.

### The model

The frozen features go into a per-person logistic regression (engine/src/model/logreg.ts, L2 1), fitted on the frozen rows of the person's last 60,000 training plays. The training rows are built with the same block tiling, so the model learns on exactly the kind of row it is scored on.

The model marks a track as a skip when its probability is at least a threshold. The threshold is chosen per person on the validation slice from 0.05, 0.10, ..., 0.95. The model is refitted on plays before the validation slice, then the threshold that gives the most hits over the full validation rounds is kept, with ties going to the one nearest 0.5. The model is then refitted on all training plays and played with that threshold.

### The baseline

The strong baseline is the better, for that person and on the held-out rounds themselves, of two rules:

1. Own skip rate. Mark all 10 as skips if the person skipped at least half their training plays, otherwise mark none.
2. Most-skipped artists. Mark a track as a skip when the artist's smoothed skip rate, frozen at the block start, is at least a threshold. The threshold is tuned on the validation slice exactly like the model's.

Picking the better rule on held-out data hands the baseline an advantage the model does not get. That is deliberate.

### Score per person

The score per person is mean hits per round for the model and for the baseline. The model wins for a person when its mean is strictly higher. A person needs at least 10 rounds to be eligible. For each person, a 95% interval for the difference in mean hits comes from a bootstrap over sessions, since rounds in one session are not independent.

As a secondary measure, the model is told how many of the 10 were skipped (k) and marks its top k. The artist rate does the same. This is reported, not tested.

## Game 2: "Of these 3 new artists, this one stays" (304 ListenBrainz and 13 Spotify)

After the first session with each of three artists the person had never heard, the model names the one that will stick. "Sticks" is the engine's definition: 10 or more proper plays on 3 or more days within 8 weeks of the first play.

### Rounds

The new artists are the held-out rows of the stick bench: artists first heard in the 8 weeks before the cutoff, whose 8-week outcome falls in the held-out weeks. A round is one artist that stuck plus two that did not, all from the same person. Every such triple is played, and accuracy is computed exactly over all of them. A tie between the stuck artist and a non-stuck one is broken at random, which counts 1/2 for one tie and 1/3 for a three-way tie. To be eligible a person needs at least 2 artists that stuck and 2 that did not.

### The model

The model is the stick model at the session horizon, unchanged.

- Spotify: the "full" logistic regression of bench/stick.ts, fitted leave one person out on everyone else's training-period rows.
- ListenBrainz: the "full" model of bench/lb3/models.ts, the columns a history without skips can fill. It is fitted on the four other folds with the same seeded 800-row subsample per person.

### The baselines

- Chance: 1/3.
- Popularity (the strong one): pick the artist the most other people played (logPeople, from people the person is not among), with ties broken by how many ListenBrainz lists name it (logListed). These are the same columns the stick bench calls popularity.

### Score per person

The score per person is accuracy over the person's triples. The model wins against popularity when its accuracy is strictly higher. It is above chance when its accuracy exceeds 1/3. Spotify and ListenBrainz are tested separately.

## Game 3: "Your next week's top 5 artists" (13 Spotify and 304 ListenBrainz)

At the start of a week, name the five artists the person will play most that week.

### Rounds

There are 8 weekly rounds, one for each held-out week, starting at the cutoff. Each round uses every play before its start, including earlier held-out weeks: this is the game played live, week after week. The truth is the artists ranked by proper plays (30 seconds or more; every ListenBrainz play counts as proper, as in the other benches) in that week. Ties at fifth place all count as top 5. A week is played only if the person played at least 5 different artists in it. A person needs at least 4 played weeks.

### Score

The score is precision at 5, the share of the five named artists that are in the week's top 5, averaged over the person's weeks.

### The model

The model is the decayed artist score of engine/src/model/taste.ts (artistScores), proper plays weighted by a half-life, top 5. The half-life is tuned per person on the validation slice. Its 8 weekly rounds before the cutoff are played with each half-life in 3, 7, 14, 30, 60, 120, 365 and 3650 days, and the best mean precision at 5 is kept, with ties going to the shorter half-life.

### The baseline

The baseline is the person's top 5 by proper plays over the 4 weeks before the round. Ties in either ranking are broken by the most recent play, then by name.

### Score per person

The model wins for a person when its mean precision at 5 is strictly higher than the baseline's.

## What gets reported for each game and data set

For each game and data set the results give:

- the headline: the mean of the per-person scores for model and baseline, and the mean per-person difference with a 95% percentile bootstrap interval over people (4,000 resamples);
- the per-person distribution: the minimum, quartiles and maximum of the difference, and a list of every person's model and baseline score (people as P01..., never by name);
- the share of individuals where the model wins, with ties counted as not winning, and its 95% Wilson interval; the counts of wins, losses and ties;
- an exact two-sided sign test on wins against losses (ties dropped).

## Multiple testing

There are five primary tests, one per row:

1. Game 1, Spotify: the model against the better of the two baselines.
2. Game 2, Spotify: the model against popularity.
3. Game 2, ListenBrainz: the model against popularity.
4. Game 3, Spotify: the model against the last 4 weeks.
5. Game 3, ListenBrainz: the model against the last 4 weeks.

Holm adjusts the five sign-test p-values. Game 2 against chance and the top-k version of game 1 are secondary, reported with unadjusted p-values only.

## The product-grade bar

A game and data set is product grade when all three of these hold:

1. the model wins for at least 70% of eligible people (ties count as not winning);
2. the lower end of the 95% Wilson interval for that share is above 50%;
3. the Holm-adjusted sign-test p is below 0.05, with more wins than losses.

With 13 people the first two together need 11 wins of 13 (11 of 13 has a Wilson lower bound of 0.58; 10 of 13 sits at 0.50). With 12 eligible people they need 10. A small consistent effect can therefore fail the bar on Spotify and still be real. The ListenBrainz rows are the ones with the power to show it.

A game that misses the bar is reported as missing it. No threshold, eligibility rule or baseline is changed after the first run. If a bug is found after the run, the fix and both sets of numbers are reported.

## Results

Run 2026-11-05 with `npm run stick` (which now also writes the held-out scores the game reads) and then `npm run game`. The full per-person tables are in the local bench/out/game.md; the aggregates below are copied from it. Nothing in the pre-registered part was changed after the run. One comparison was added after seeing the results and is marked exploratory: game 2 against the artist the person played most in its first session.

### The short version

One game clears the bar on both data sets. Picking which of three new artists will stick beats popularity for 9 of 9 eligible Spotify people and 132 of 149 eligible ListenBrainz listeners. Calling 10 skips beats the better simple rule for 9 of 12 Spotify people, which is a clear average gain that falls short of the bar. Naming next week's top 5 barely moves past "your last 4 weeks", for 5 of 12 Spotify people and 165 of 294 ListenBrainz listeners.

| game | data | people | model | baseline | mean difference [95%] | model wins | share [Wilson 95%] | Holm p | bar |
|---|---|---|---|---|---|---|---|---|---|
| 1 call 10 skips (hits of 10) | Spotify | 12 of 13 | 7.87 | 7.47 best rule | 0.40 [0.20, 0.61] | 9, lose 2, tie 1 | 0.75 [0.47, 0.91] | 0.131 | misses |
| 2 which of 3 sticks (accuracy) | Spotify | 9 of 13 | 0.688 | 0.371 popularity | 0.316 [0.207, 0.427] | 9, lose 0 | 1.00 [0.70, 1.00] | 0.012 | passes |
| 2 which of 3 sticks (accuracy) | ListenBrainz | 149 of 304 | 0.695 | 0.456 popularity | 0.239 [0.204, 0.274] | 132, lose 14, tie 3 | 0.89 [0.83, 0.93] | <0.001 | passes |
| 3 next week's top 5 (precision) | Spotify | 12 of 13 | 0.484 | 0.474 last 4 weeks | 0.009 [-0.022, 0.038] | 5, lose 4, tie 3 | 0.42 [0.19, 0.68] | 1.000 | misses |
| 3 next week's top 5 (precision) | ListenBrainz | 294 of 304 | 0.294 | 0.271 last 4 weeks | 0.023 [0.016, 0.030] | 165, lose 75, tie 54 | 0.56 [0.50, 0.62] | <0.001 | misses |

### Game 1: calling 10 skips

The model calls 7.9 of 10 tracks right on average against 7.5 for the better simple rule, and its interval over people sits above zero. For most people the gain is a third of a track to one full track per block. It misses the bar on the count of people: 9 wins, 2 small losses (0.04 and 0.03 of a track) and one tie, where 11 wins of 12 were needed. The baseline here was picked per person on the held-out rounds, which favours it. Told how many tracks will be skipped, the model's top k beats the artist rate's top k for the same 9 of 12 (8.24 against 8.07 hits).

For a person, the honest framing is "8 of 10 right, where always guessing your habit gets 7.5". That is real and checkable, but the margin is small enough that a user would rarely notice it.

### Game 2: which of three new artists sticks

This is the strong result. After one session with each of three new artists, the model names the one that stays about 69% of the time, against 33% by chance. It beats picking the most popular for every eligible Spotify person and for 89% of eligible ListenBrainz listeners, with the Wilson lower bound at 0.83. Against chance it wins for 9 of 9 and 141 of 149.

Two cautions. Popularity turned out to be a weak baseline (0.37 and 0.46, near chance), so beating it is less demanding than the pre-registration expected. And the exploratory comparison shows where most of the skill comes from: picking the artist played most in its first session scores 0.688 on Spotify, the same as the model (5 wins of 9), and 0.626 on ListenBrainz, where the model still wins for 104 of 149 (0.70, Wilson [0.62, 0.77], mean gain 0.07). So the product statement holds, and the model adds a modest amount over a rule a person could apply themselves. This comparison was not pre-registered and is not part of the Holm family. Only half of the ListenBrainz listeners (149 of 304) had the two stuck artists in their held-out weeks needed to play.

### Game 3: next week's top 5

Naming next week's top 5 from a decayed history gets 2.4 of 5 right on Spotify and 1.5 of 5 on ListenBrainz. Your last 4 weeks do almost as well. On ListenBrainz the small gain (0.023 of precision, about one extra artist every nine weeks) is consistent enough to be significant over 294 people, but the model wins for only 56% of them. This game is not a product claim.

### Which game is the wow

Game 2. It is the only one that passes the bar on both data sets, it has a large gap over chance that a person can feel (two in three against one in three), and it is checkable within eight weeks. The caveat to carry into any copy is that much of that skill is "the one you played most in its first session". The statement is honest; the claim that it needs a model is weaker.

### Per-game summaries

The per-person tables (people as P01 and so on) stay in the local bench/out/game.md and are left out of the public repo. The summaries below are the aggregate lines from it.

#### Game 1, Spotify: call 10 skips: model vs the better of own skip rate and most-skipped artists

12 of 13 people eligible. Mean per person (hits out of 10): model 7.869, best baseline 7.468, difference 0.401 [0.196, 0.612]. Per-person difference min -0.044, quartiles 0.026 / 0.343 / 0.693, max 1.075. The model wins for 9 of 12 (share 0.750, Wilson 95% [0.468, 0.911]), loses for 2, ties 1. Sign test p 0.065, Holm 0.131. Misses the product-grade bar.

#### Game 1, Spotify: call 10 skips, told how many: model top k vs artist skip rate top k

12 of 13 people eligible. Mean per person (hits out of 10): model 8.239, artist top k 8.071, difference 0.169 [0.084, 0.260]. Per-person difference min -0.067, quartiles 0.044 / 0.165 / 0.211, max 0.478. The model wins for 9 of 12 (share 0.750, Wilson 95% [0.468, 0.911]), loses for 2, ties 1. Sign test p 0.065 (secondary, unadjusted).

#### Game 2, Spotify: which of 3 new artists sticks: model vs popularity

9 of 13 people eligible. Mean per person (accuracy): model 0.688, popularity 0.371, difference 0.316 [0.207, 0.427]. Per-person difference min 0.099, quartiles 0.145 / 0.294 / 0.429, max 0.570. The model wins for 9 of 9 (share 1.000, Wilson 95% [0.701, 1.000]), loses for 0, ties 0. Sign test p 0.004, Holm 0.012. Passes the product-grade bar.

#### Game 2, Spotify: which of 3 new artists sticks: model vs chance

9 of 13 people eligible. Mean per person (accuracy): model 0.688, chance 0.333, difference 0.354 [0.266, 0.437]. Per-person difference min 0.105, quartiles 0.271 / 0.315 / 0.451, max 0.535. The model wins for 9 of 9 (share 1.000, Wilson 95% [0.701, 1.000]), loses for 0, ties 0. Sign test p 0.004 (secondary, unadjusted).

#### Game 2, Spotify: which of 3 new artists sticks: model vs most played in its first session (exploratory, added after the run)

9 of 13 people eligible. Mean per person (accuracy): model 0.688, first session 0.688, difference -0.001 [-0.105, 0.092]. Per-person difference min -0.272, quartiles -0.030 / 0.022 / 0.134, max 0.148. The model wins for 5 of 9 (share 0.556, Wilson 95% [0.267, 0.811]), loses for 4, ties 0. Sign test p 1.000 (secondary, unadjusted).

#### Game 2, ListenBrainz: which of 3 new artists sticks: model vs popularity

149 of 304 people eligible. Mean per person (accuracy): model 0.695, popularity 0.456, difference 0.239 [0.204, 0.274]. Per-person difference min -0.437, quartiles 0.105 / 0.215 / 0.390, max 0.886. The model wins for 132 of 149 (share 0.886, Wilson 95% [0.825, 0.928]), loses for 14, ties 3. Sign test p <0.001, Holm <0.001. Passes the product-grade bar.

#### Game 2, ListenBrainz: which of 3 new artists sticks: model vs chance

149 of 304 people eligible. Mean per person (accuracy): model 0.695, chance 0.333, difference 0.362 [0.330, 0.393]. Per-person difference min -0.234, quartiles 0.254 / 0.380 / 0.515, max 0.667. The model wins for 141 of 149 (share 0.946, Wilson 95% [0.898, 0.973]), loses for 8, ties 0. Sign test p <0.001 (secondary, unadjusted).

#### Game 2, ListenBrainz: which of 3 new artists sticks: model vs most played in its first session (exploratory, added after the run)

149 of 304 people eligible. Mean per person (accuracy): model 0.695, first session 0.626, difference 0.069 [0.049, 0.089]. Per-person difference min -0.205, quartiles -0.003 / 0.047 / 0.128, max 0.617. The model wins for 104 of 149 (share 0.698, Wilson 95% [0.620, 0.766]), loses for 38, ties 7. Sign test p <0.001 (secondary, unadjusted).

#### Game 3, Spotify: next week's top 5: decayed score vs last 4 weeks' top 5

12 of 13 people eligible. Mean per person (precision at 5): model 0.484, last 4 weeks 0.474, difference 0.009 [-0.022, 0.038]. Per-person difference min -0.114, quartiles -0.025 / 0.000 / 0.056, max 0.075. The model wins for 5 of 12 (share 0.417, Wilson 95% [0.193, 0.680]), loses for 4, ties 3. Sign test p 1.000, Holm 1.000. Misses the product-grade bar.

#### Game 3, ListenBrainz: next week's top 5: decayed score vs last 4 weeks' top 5

294 of 304 people eligible. Mean per person (precision at 5): model 0.294, last 4 weeks 0.271, difference 0.023 [0.016, 0.030]. Per-person difference min -0.229, quartiles -0.025 / 0.025 / 0.050, max 0.229. The model wins for 165 of 294 (share 0.561, Wilson 95% [0.504, 0.617]), loses for 75, ties 54. Sign test p <0.001, Holm <0.001. Misses the product-grade bar.


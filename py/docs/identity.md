# Identity in k sentences

How many plain sentences about the way someone listens does it take to tell them apart from everyone
else, and how much of what singles a person out survives from one stretch of their history to the
next? Round 2 of the statements lane (`statements.md`) showed that three statements, each chosen on
its own for being rare, confident and stable, pick a listener's later history out of a lineup of five
67.5% of the time on fresh people, and that a fingerprint over all 57 measures does it 89.5% of the
time. This round asks the question in terms of information, chooses the sentences together instead of
one at a time, measures identification against everyone at once instead of four decoys, and
compares the readable sentences with the item fingerprint (which artists someone plays) that the
literature already knows works (`related_work_identity.md`).

Everything is computed on ListenBrainz listeners found by the same similar-users walk as rounds 1 and
2, with the same cleaning and the same person rule. The code is `src/aas/statements/identity.py` and
`scripts/run_identity.py`; the tests are `tests/test_identity.py`.

## The maths

### Windows

Each person's training weeks (everything before their last 8 weeks) are cut at the midpoint in time
into a first half H1 and a second half H2. H1 is cut again at its midpoint into H1a and H1b. All 57
families of round 2 are measured on each window. Sentences are always made from H1 and checked on H2;
nothing about a person's H2 is used to choose their sentences.

### Bits of identity

A statement s (a family, a direction and one of the family's round thresholds) is true of a window x
when x_f >= th ("at least") or x_f <= th ("no more than"); a value that cannot be measured makes it
false. Against a reference of n other people, of whom c_S fit every statement in a set S, the set
carries

    b(S) = -log2((c_S + 1/2) / (n + 1))

bits. This is the share of others that fit, with the Krichevsky-Trofimov half count so that a set
nobody else fits is worth log2(2n + 2) bits instead of infinitely many. Picking one person out of N
takes log2 N bits: 9.3 for the 636 people explored here.

If the statements in S were independent, the bits would add: b_ind(S) = sum over s of b({s}). The gap
between b_ind(S) and b(S) measures how much the sentences repeat each other.

A sentence is only worth its bits if it is still true later. The surviving bits of a set are b(S)
computed on the others' second halves when the person's own second half still fits all of S, and 0
when it does not.

### How likely a sentence is to hold

Every value is turned into a normal score z = Phi^-1(p), where p is its mid-rank percentile among the
population's values on the same kind of window. Per family f, r_f is the correlation across people of
z(H1a) with z(H1b): the part of a window's score that belongs to the person rather than to the window.
A person's score is modelled as their own level plus window noise, z = mu + e, with Var(mu) = r_f.
Their noise variance starts from the family's 1 - r_f and is moved towards what their own two quarters
show:

    tau^2 = (lambda (1 - r_f) + (z(H1a) - z(H1b))^2 / 2) / (lambda + 1),   lambda = 2.

Given z1 = z(H1), the next window's score has mean m = z1 r_f / (r_f + tau^2) and variance
v = kappa (r_f tau^2 / (r_f + tau^2) + tau^2), with kappa = 1.44. The chance that a sentence true on H1
is still true next time is

    q = Phi(sign (m - z(th)) / sqrt(v)),

with sign +1 for "at least" and -1 for "no more than", and z(th) the normal score a value equal to the
threshold would get. The factor r_f / (r_f + tau^2) pulls extreme people back towards the middle,
which is what extreme people do; with tau^2 = 1 - r_f the formula is the bivariate normal
Phi(sign (r_f z1 - z(th)) / sqrt(1 - r_f^2)). Only H1, H1a and H1b enter. lambda and kappa were set on
the exploration set by calibration of q against whether rare statements (true of 20% or fewer) still
held on H2 (log loss 0.635, against 0.76 for a model without the pull to the middle and 1.04 for round
2's confidence, which assumed no regression to the mean); they were not tuned on lineup accuracy.

### Choosing the sentences together

The lineup matcher is round 2's: it counts how many of the target's sentences are true of each
candidate's second half and picks the highest, an m-way tie scoring 1/m. Suppose the target's count is
a sum of independent Bernoulli(q_s) and each decoy is drawn from the reference, reference person j
counting sentence s with their own q_js (the same formula, applied to them). Write P_T(t) for the
target's count distribution, and G(t) and E(t) for the chance that a decoy counts fewer than t and
exactly t. With D decoys the expected credit of a set S is

    credit(S) = sum over t of P_T(t) ((G(t) + E(t))^(D+1) - G(t)^(D+1)) / ((D + 1) E(t)),

and G(t)^D when E(t) = 0. This is the expected 1/m over ties, summed in closed form over how many decoys
tie (the test checks it against brute-force enumeration). Joint selection is greedy: start empty and
add, one at a time, the true sentence that most raises credit(S), with at most one sentence per family
and at most two per group of families, until there are k. Ties go to the rarer, then the surer
sentence. For speed the greedy steps after the first look only at the 200 sentences that do best alone.

Two things follow from the objective. A sentence the target only just clears has a low q and is passed
over for a slightly blunter one they clear with room to spare, so the threshold adapts to the person
(any round number on the family's grid can be chosen, not one per family). And a second sentence that
says much the same as the first adds almost nothing to credit(S), so overlapping sentences are not
stacked. For a lineup of five D = 4, and the reference is everyone outside the lineup. For 1-of-N the
same rule is used with D = N - 1 against everyone else.

Every sentence is one of round 2's: a share, a rate or a typical length with a round threshold, about
behaviour only, with no artist, track, release, name or raw play count, and nothing about health,
religion, sexuality or politics. Nothing new is added to what a sentence may say.

### 1 of N

For each person, their sentences (made from H1 against all N - 1 others) are counted on every one of the
N second halves; the rank of their own second half, with ties broken at random in expectation, gives
top-1 and top-5 accuracy. The rank bits, log2 N - E[log2 rank], say how much of the log2 N needed the
matcher recovered on average.

### Fingerprints

The fingerprint compares all 57 measures at once. Round 2's version is the mean absolute difference of
percentiles. The learned one puts every value on the normal-score scale of the first halves (an
unmeasurable value counts as 0, the median) and uses the distance

    d(x, y) = (z_x - z_y)^T W^-1 (z_x - z_y),

where W is the within-person covariance of window-to-window changes, estimated from the same set's
H1a and H1b: W = (Za - Zb)^T (Za - Zb) / (2P). A measure that under 10% of people can be scored on in
both quarters gets variance 1 and no covariance, and eigenvalues are floored at a thousandth of the mean
variance. Measures that wobble from quarter to quarter count for less, and two measures that wobble
together count once. No second half is used to learn W. A diagonal version (weights 1 / W_ff) and plain
Euclidean distance on the normal scores are reported alongside.

### The item baseline

The item fingerprint is what the literature uses: each half becomes the vector of square roots of its
artist play shares, and the similarity of two halves is sum over artists of sqrt(p_a p'_a) (the
Hellinger affinity). It names artists, so it could never be printed on a card; it is here to price
readability.

### Time decay

Window B is the last 26 weeks of the training weeks. Window A is 26 weeks long and ends 0, 1, 3, 6 or 12
months before B starts. On the people whose history reaches back far enough for the 12 month gap, the
same people at every gap, 3 joint sentences are made from A (with A's two halves standing in for H1a
and H1b) and checked on B, in lineups of five and 1 of N, and the learned fingerprint and the item
fingerprint are matched from A to B.

## Exploration on the 636

All of this was built and looked at on the 303 listeners of round 1 and the 333 of round 2's
confirmation set together: 636 people, every one of whom has now been seen. Nothing in this section is
a confirmation. Lineups are 20 per person with seed 7, the same lineups for every method. Every number
is in `docs/identity_results.json` under "exploration", and the curves are in
`docs/identity_exploration_lineup.csv`, `_one_of_n.csv`, `_bits.csv` and `_decay.csv`.

### What was tried

The first version of the hold probability used each person's own quarter-to-quarter wobble with no pull
towards the middle. It was badly overconfident for rare statements, the ones that matter: statements it
gave a 90% chance held on the second half about 70% of the time, because people at the edge of the
population drift back towards it. Adding the pull (the r_f / (r_f + tau^2) factor) fixed the
calibration, and on a subsample of 64 people joint selection with it went from 69% to 76% on three
sentences in a lineup of five (round 2's rule: 65%). With the hold probability fixed, the remaining choices made little difference on the 636: letting
any number of sentences come from one group of families (75.1%), allowing only sentences true of half
the others or fewer (75.2%), counting decoys by their first-half truth instead of their own hold
probability (73.7%), and kappa = 1 instead of 1.44 (75.3%). A 20% rarity line, round 2's, cost three
points (72.5%), because it stops the greedy step from pairing two moderately rare sentences that are
rare together. The rule kept is the one described above: no rarity line, at most two per group, decoys
counted by their hold probability.

### Lineups of five

Three sentences chosen together pick the person out of five 75.2% of the time (95% interval 73.0% to
77.4%), against 68.2% (65.6% to 70.7%) for round 2's three chosen one at a time on the same lineups. The
paired difference is 7.0 points (4.9 to 9.2). The gap opens with every sentence added: the joint rule
reaches 84.8% with seven, while round 2's rule flattens at 73.3%, partly because it runs out of
sentences (12% of its lineups have fewer than seven) and partly because its later sentences repeat the
earlier ones.

| sentences | joint | round 2 rule |
|---|---|---|
| 1 | 59.0% | 55.4% |
| 2 | 70.0% | 64.0% |
| 3 | 75.2% | 68.2% |
| 4 | 78.8% | 70.7% |
| 5 | 82.0% | 71.9% |
| 6 | 83.6% | 72.4% |
| 7 | 84.8% | 73.3% |

The fingerprint over all 57 measures does better than any number of sentences here: 89.8% with round
2's percentile distance and 88.6% with the learned distance. In a lineup of five the learned distance
does not help.

### One out of all 636

Against everyone at once the picture changes. Chance is 0.16%. Three joint sentences put the right
second half first 13.3% of the time (11.0% to 15.7%) and in the top five 25.0%; round 2's three manage
6.7% and 20.1%. Seven joint sentences reach 24.0% top-1 and 42.5% top five. The fingerprint does far
better: 44.5% top-1 with round 2's percentile distance, 50.3% (46.4% to 54.4%) with the learned
distance, a paired gain of 5.8 points (2.2 to 9.4). Weighting each measure by its own noise alone gives
43.2% and plain distance 40.3%, so the gain comes from counting measures that move together once.

The item fingerprint, which artists a person plays, puts the right person first 95.0% of the time and
in the top five 97.3%. That is the price of readability: the 57 measures that could be printed on a
card find half the people the artist list finds, and three sentences find about one in seven of them.

| matcher (1 of 636) | top-1 | top-5 | rank bits |
|---|---|---|---|
| 1 joint sentence | 3.6% | 11.9% | 3.1 |
| 3 joint sentences | 13.3% | 25.0% | 4.6 |
| 3 sentences, round 2 rule | 6.7% | 20.1% | 4.5 |
| 7 joint sentences | 24.0% | 42.5% | 5.9 |
| fingerprint, percentiles (round 2) | 44.5% | 64.6% | 7.2 |
| fingerprint, learned distance | 50.3% | 68.1% | 7.4 |
| items (artists), Hellinger | 95.0% | 97.3% | 9.1 |

Rank bits are log2 636 - E[log2 rank], out of the 9.3 needed.

### Bits per sentence

The best single true sentence about a person fits, on average, a share of the others worth 5.5 bits.
When the sentences are chosen for 1 of N (D = 635), the first is worth 5.3 bits, two 8.1 and three 9.5,
against the 9.3 needed and the 10.3 that the estimate can show for a set nobody else fits. If the three
were independent they would carry 10.7 bits, so joint choice loses about 1.2 bits to overlap (some of
it the ceiling). Round 2's three, chosen one at a time, carry 6.4 bits together against 10.2 if they
were independent: almost 4 bits, more than a third, are the same information said twice.

On paper, then, three well-chosen sentences are nearly enough to single out one person in 636. In
practice the bits do not last. Only 19% of people still fit all three of their joint 1-of-N sentences
on their second half (42% for one sentence), and the bits that survive average 1.7 for three sentences
and 1.9 for one. Round 2's blunter sentences keep better (38% fit all three) and keep 2.1 bits. The
rarest description of a person is mostly a description of the time it was made in. The lineup objective
trades bits for staying power, which is why it is the one that wins on lineups and 1 of N.

| k | joint bits | independent sum | keep all on H2 | surviving bits | round 2 bits | round 2 keep |
|---|---|---|---|---|---|---|
| 1 | 5.3 | 5.3 | 42% | 1.9 | 3.8 | 63% |
| 2 | 8.1 | 8.5 | 25% | 1.8 | 5.3 | 49% |
| 3 | 9.5 | 10.7 | 19% | 1.7 | 6.4 | 38% |
| 5 | 10.0 | 14.3 | 15% | 1.4 | 8.0 | 25% |
| 7 | 10.1 | 17.3 | 13% | 1.2 | 8.9 | 17% |

### Time decay

469 of the 636 have enough history for a 26-week window ending a year before their last 26 training
weeks. On them, with the same people at every gap:

| gap between windows | 3 sentences, lineup of 5 | 3 sentences, 1 of 469 | fingerprint, 1 of 469 | fingerprint, lineup | items, 1 of 469 |
|---|---|---|---|---|---|
| none | 78.7% | 13.6% | 58.2% | 89.3% | 95.7% |
| 1 month | 75.1% | 14.1% | 53.3% | 87.1% | 93.4% |
| 3 months | 73.6% | 13.0% | 50.3% | 85.0% | 92.8% |
| 6 months | 70.0% | 9.8% | 39.0% | 82.1% | 92.8% |
| 12 months | 65.8% | 9.4% | 29.2% | 79.2% | 89.1% |

Everything fades with time, the behavioural measures faster than the artists. Over a year the learned
fingerprint's top-1 halves, from 58% to 29%, while the artist list loses six points. Three sentences
lose about 13 points in a lineup over the year.

## Pre-registration: the third set

This section was committed and pushed before any listener of the third set was measured. The code it
names is frozen at that commit (`src/aas/statements/identity.py`, `scripts/run_identity.py`), and the
script refuses to make the confirmation run twice.

### Who

The similar-users walk was continued on 5 November 2026 from where round 2 left it, with the same rules
(5,000 to 2,000,000 listens, a year or more between first and latest listen, a listen in the last 120
days), until 1,300 accounts qualified. The third set is the first 400 qualifying accounts, in pseudonym
order, that neither earlier download ever started (`LB_SET=third`, folders `raw_third/` and
`plays_third/`, same salt, same download rules: newest first, 60,000 listens, extended to two years for
heavy listeners). Cleaning is round 2's, including the single-artist exclusion, and so is the person rule
(2,000 or more listens; the last 8 weeks dropped, with 1,000 or more listens before them and 200 or more
inside them). Whoever is left is the third set; accounts that fail to download are left out and nobody
is replaced by hand. If fewer than 250 people are left the run waits for more downloads.

### The frozen algorithm

As written above under "The maths", with the constants in `identity.py`: lambda = 2, kappa = 1.44,
r_f clipped to [0.05, 0.99], at most two sentences per group, 200 sentences kept after the first greedy
step, decoys counted by their own hold probability, no rarity line; the learned fingerprint with no
shrinkage and a ridge of a thousandth. Every population quantity (normal scores, r_f, W, the rarity
counts) is computed on the third set itself, from first halves and their quarters only. Nothing is
carried over from the 636 except the rule and its constants. Lineups: 20 per person, seed 7, four
decoys from the third set.

### Primary test

Three jointly chosen sentences in a lineup of five, scored by round 2's counting matcher. It passes when
accuracy is at least 75%, the lower end of its 95% bootstrap interval (4,000 resamples over people) is
above 67.5% (round 2's confirmed figure), and the permutation p (second halves shuffled, 1,000 times) is
below 0.01. On the 636 it scored 75.2% (73.0% to 77.4%), so the 75% bar sits at the exploration estimate
and a pass on the point estimate is close to a coin toss even if nothing was overfitted; the interval
condition is the part the exploration says should hold.

### Secondary tests

1. Learned fingerprint, 1 of N (N = everyone in the third set): top-1 at least 40%, and its paired gain
   over round 2's percentile fingerprint with a 95% interval above 0. (Exploration: 50.3% at N = 636,
   gain 5.8 points, 2.2 to 9.4.)
2. Three joint sentences, 1 of N: top-1 at least 10%, and the paired gain over round 2's three with a
   95% interval above 0. (Exploration: 13.3% against 6.7% at N = 636.)
3. Joint against round 2's selection in lineups of five, three sentences: paired difference with a 95%
   interval above 0. (Exploration: 7.0 points, 4.9 to 9.2.)

Reported without a bar: the lineup and 1-of-N curves for k = 1 to 7; bits per sentence (bits of the
joint and round 2 sets on the others' first halves, the independence sum, the share still fitting on
the second half and the surviving bits), with the expectation that joint sets lose much less to overlap
than round 2's; the item fingerprint and the cost of readability; the time-decay curve on the people
with enough history, expected to fall with the gap for every matcher and fastest for the behavioural
fingerprint; and the other fingerprint distances.

## Results on the third set

Run once on 6 November 2026 with `run_identity.py confirm`, after the pre-registration above was
committed and pushed (commit b7c3cdf) and with the frozen files unchanged since. Every number is in
`docs/identity_results.json` under "confirmation", and the curves are in
`docs/identity_confirmation_*.csv`.

### Who was in it

All 400 accounts downloaded without a failure. Cleaning kept 387: 6 were dropped for listens faster than
listening, 5 for 500 or more listens on too many days, 3 for scripted pacing, 2 each for one track
dominating and under a year kept, and 1 each for bulk-import timestamps, too few listens and too few
active weeks (some hit more than one rule); none for the single-artist rule. The person rule then left
352. None of them is in either earlier set. One person in 352 takes log2 352 = 8.5 bits.

### The verdict: the primary test fails

Three jointly chosen sentences pick a third-set listener's second half out of a lineup of five 68.8% of
the time (95% interval 65.5% to 72.1%, permutation p = 0.001). The bar was 75% with the interval's lower
end above 67.5%, and both conditions fail. On the 636 people it was built on the rule scored 75.2%; on
new people it scored 6.4 points less.

The comparison inside the run says why, and it is the part that held. Round 2's rule, run on the same
352 people and the same lineups, scored 61.5% (57.9% to 65.2%), six points below the 67.5% it was
confirmed at on round 2's fresh listeners. The third set is harder for every sentence rule: its people
are less often still themselves on their second half (19% keep all three joint 1-of-N sentences, 29% all
three of round 2's). Against that baseline joint selection gained 7.2 points (4.2 to 10.4), the same 7.0
it gained on the exploration set. So the improvement from choosing sentences together replicated; the
absolute level the bar asked for did not, because the bar was set against round 2's number on a
different set of people and the level moves by several points from one set to the next.

| test (third set, 352 people) | result | 95% interval | bar | verdict |
|---|---|---|---|---|
| 3 joint sentences, lineup of 5 (primary) | 68.8% | 65.5% to 72.1% | >= 75%, lower end > 67.5%, p < 0.01 | fail |
| joint minus round 2 rule, lineup of 5 | +7.2 points | 4.2 to 10.4 | lower end > 0 | pass |
| 3 joint sentences, 1 of 352, top-1 | 13.6% | 10.6% to 16.8% | >= 10% | pass |
| joint minus round 2 rule, top-1 | +5.1 points | 2.5 to 7.7 | lower end > 0 | pass |
| learned fingerprint, 1 of 352, top-1 | 42.3% | 36.9% to 47.4% | >= 40% | |
| learned minus percentile fingerprint, top-1 | +4.5 points | -0.003 to 9.1 | lower end > 0 | fail |

The third secondary (joint beats round 2 in lineups) and the second (three sentences find one person in
352 at least 10% of the time, and better than round 2's three) pass. The first fails by a hair: the
learned fingerprint cleared 40%, but its gain over round 2's percentile distance has an interval whose
lower end is 0.00003 below zero. On these people the learned distance is about four points better and
the data cannot rule out none.

### Lineups and 1 of N

| sentences | joint, lineup | round 2 rule, lineup | joint, 1 of 352 top-1 | round 2 rule, top-1 | joint, top-5 |
|---|---|---|---|---|---|
| 1 | 52.8% | 49.5% | 3.9% | 2.9% | 15.0% |
| 2 | 62.2% | 58.3% | 8.9% | 6.0% | 20.3% |
| 3 | 68.8% | 61.5% | 13.6% | 8.4% | 27.6% |
| 4 | 72.5% | 64.1% | 17.1% | 10.2% | 33.5% |
| 5 | 74.8% | 65.2% | 18.5% | 11.2% | 38.2% |
| 6 | 76.5% | 66.2% | 20.8% | 13.1% | 42.7% |
| 7 | 77.6% | 67.0% | 23.3% | 13.6% | 46.1% |

The joint curve keeps rising with each sentence; round 2's flattens. Two joint sentences (62.2%) do about
as well as round 2's three (61.5%), and from five sentences on the joint rule is ten points ahead.

| matcher (1 of 352) | top-1 | top-5 | lineup of 5 | rank bits |
|---|---|---|---|---|
| 3 joint sentences | 13.6% | 27.6% | 68.8% | 4.1 |
| fingerprint, percentiles (round 2) | 37.8% | 61.1% | 84.3% | 6.2 |
| fingerprint, noise weights only | 39.2% | 59.7% | 83.8% | 6.1 |
| fingerprint, learned distance | 42.3% | 62.8% | 84.5% | 6.3 |
| items (artists), Hellinger | 94.0% | 96.3% | 98.5% | 8.2 |

The cost of readability replicates. The artist list finds 94% of people in 352; the 57 printable
measures find 42%, and three sentences 14%. In rank bits, out of the 8.5 needed, the artists recover
8.2, the measures 6.3 and three sentences 4.1: three plain sentences keep about half of the identity the
artist list carries and two thirds of what all 57 measures carry.

### Bits per sentence

| k | joint bits | independent sum | keep all on H2 | surviving bits | round 2 bits | round 2 sum | round 2 keep |
|---|---|---|---|---|---|---|---|
| 1 | 5.3 | 5.3 | 38% | 1.6 | 3.8 | 3.8 | 55% |
| 2 | 7.8 | 8.2 | 22% | 1.4 | 5.4 | 7.2 | 37% |
| 3 | 8.8 | 10.4 | 19% | 1.4 | 6.4 | 10.3 | 29% |
| 5 | 9.2 | 14.1 | 15% | 1.2 | 7.7 | 16.0 | 17% |
| 7 | 9.2 | 17.0 | 14% | 1.1 | 8.5 | 21.2 | 11% |

The estimate tops out at log2(2 x 351 + 2) = 9.5 bits here, and joint sets reach it by about the fourth
sentence. The first joint sentence is worth 5.3 bits and the first two 7.8, against the 8.5 needed: on
the first half, two or three sentences chosen together are nearly enough to name one person in 352. Round
2's three, chosen one at a time, carry 6.4 bits against 10.3 if they were independent, so close to 4
bits are repetition; joint choice loses 0.4 bits at two sentences (more at three, where the ceiling
starts to bite). As on the exploration set, the bits mostly do not last: fewer than one person in five
still fits all three of their sentences on the second half, and the bits that survive average 1.4.

### Time decay

261 of the 352 have enough history for the year gap. Same people at every gap:

| gap between windows | 3 sentences, lineup of 5 | 3 sentences, 1 of 261 | fingerprint, 1 of 261 | fingerprint, lineup | items, 1 of 261 |
|---|---|---|---|---|---|
| none | 76.7% | 16.7% | 64.0% | 87.8% | 96.9% |
| 1 month | 74.5% | 18.1% | 59.4% | 85.5% | 97.7% |
| 3 months | 71.2% | 14.1% | 47.9% | 82.1% | 96.9% |
| 6 months | 71.0% | 11.4% | 42.9% | 79.7% | 94.6% |
| 12 months | 61.7% | 8.3% | 29.5% | 72.2% | 90.4% |

The shape matches the exploration set. A year between the windows halves the learned fingerprint's top-1
(64% to 30%) and takes 15 points off three sentences in a lineup, while the artist list loses six and a half.
Habits of listening drift faster than the artists a person plays.

## What this round shows, and its limits

Choosing the sentences together, for how likely each is to stay true and how little it repeats the
others, is worth about seven points in a lineup of five and about five points of top-1 against everyone,
on two independent sets of people. That improvement is the confirmed result. The absolute figure the
primary test asked for (75%) was not reached: on the third set three sentences scored 68.8%, which is
above round 2's 67.5% but with an interval that reaches down to 65.5%.

- The level moves between sets more than the method does. Round 2's rule scored 67.5% on round 2's fresh
  set and 61.5% on the third; a bar pinned to the earlier set's number mixed the two. A paired comparison
  on the same people (the third secondary) is the test that measures the method.
- The rule was built on 636 people and lost 6.4 points on new ones, against about two points for round
  2's rule. Some of that is the third set being harder for every rule (round 2's lost six points too),
  so the overfit itself looks small, but this run cannot separate the two.
- The hold probability assumes a sentence's chances are independent of each other and that the
  population stays where it was; neither is exactly true.
- The bits are on first halves. Most of a rare description does not survive to the second half; what a
  sentence is worth for recognising someone later is its surviving bits, about 1.4 for a set of three.
- 1 of N here means N of a few hundred. Rank bits and the decay with N say nothing yet about tens of
  thousands of listeners, where the item fingerprint is known to keep working.
- All three sets come from the same similar-users walk over ListenBrainz, so they are the same kind of
  listener (self-hosters, Last.fm veterans). Other populations are untested.
- Whether a person recognises their own three sentences, or a stranger can use them, is untested on any
  person; everything here is matching by the data alone.

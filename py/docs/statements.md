# Statements: what your listening says about you and almost nobody else

The v1 personality estimates are honest and they read like a horoscope. A fused r near 0.16 means most people see a
z near zero, and a sentence that fits nearly anyone cannot surprise anyone. This lane tries the other direction.
Instead of placing a person on a scale, it looks for things that are true of them and false of nearly everyone
else, written so that the person can check them against their own memory: "83% of your plays went to artists
already in your top 50 four to six months before", with the note that this is true of fewer than 1% of listeners.

A statement is only worth showing if three things hold. It has to be specific, which here means true of the person
and of at most 10% of a reference population. It has to be checkable, which means it should still be true on a
different stretch of the same person's history. And it has to carry information about who the person is, which is
what the lineup test measures: given a few statements made from one half of someone's history, can a blind matcher
pick that person's other half out of a lineup of five?

Code: `src/aas/statements/` (`history.py` the window, `families.py` the measures and sentences, `engine.py` the
filter, `lineup.py` the test). Run: `cd py && uv run python scripts/run_statements.py all`. Tests:
`tests/test_statements.py`.

## Pre-registration (written and committed before the lineup test was run)

This section was committed before any lineup, stability check or Barnum control was run on real people. Before
it, the family values were measured on the 304 listeners once (`run_statements.py values`) to check that every
family can be computed and that the threshold grids cover the population's range; that step printed only the 10th,
50th and 90th percentile of each family and its split-half r. Two grids were widened after it (more round numbers
at the top of `effective_artists` and `depth`, and 0.5% added to the percentage grid). Nothing else was changed
after looking at data.

### Data

The 304 ListenBrainz listeners of the music reader (`bench/lbusers`, `docs/reader_music.md`): at least 2,000
listens, with 1,000 or more before their last 8 weeks and 200 or more in them. The training weeks are used (the
last 8 weeks are dropped, as in the reader's norms). Artists are keyed by MusicBrainz id when a listen carries
exactly one, otherwise by the lower-cased credit; a track is the artist key plus the title. Times are UTC start
times. ListenBrainz has no time zone, no play length and no skips, so nothing below uses the hour on the clock,
the name of a weekday, how long a track played or whether it was skipped.

### Families

34 families in 9 groups (`families.py`). Each is a share, a rate per month, or a typical length, measured on a
window of time. Measures that would grow with the length of the window (the longest streak ever, the number of
artists ever heard) are taken per 13-week season and averaged, or per 30 days, so a person with ten years of
history is compared fairly with one who has one. Each family has a fixed grid of round thresholds and two
sentences, one for "at least" and one for "no more than". A statement is a family, a direction and a threshold
from the grid. Raw play counts and artist names are never statements.

| group | families |
|---|---|
| discovery | new share, new artists per month, one and done, stick rate, discovery bursts, comebacks per month |
| loyalty | old friends, season turnover, same top artist, dropped cold |
| concentration | top 1 share, top 5 share, effective number of artists, one artist's share of a week, artist days |
| repetition | back to back, same song within a day, heavy rotation, album runs, depth |
| runs | artist run length, long runs, artist day streak |
| clock | peak four hours, centre shift, two adjacent days |
| rhythm | weekly swing, active days, plays per active day, silent weeks, listening streak |
| sessions | session length, marathon share, sessions per day |

A window reads the plays inside it and the plays before it (to know which artists were already familiar). It
never reads later plays. The first half of a history is everything before the midpoint in time, the second half
everything after.

### The "only you" filter

For a person and a reference population (everyone else, or everyone else except a lineup's decoys), a statement
qualifies when it is true of the person and true of at most 10% of the reference. For each family and direction
the threshold with the highest score is kept, then the better direction per family.

Score is confidence times rarity. Rarity is the surprisal −log2((c + 1) / (n + 1)) of the c of n reference
people the statement is also true of. Confidence is Phi((value − threshold) / s) (sign flipped for "no more
than"), where s is the standard deviation of the family across the four quarters of the window, floored at a
tenth of the family's population standard deviation, and set to half of the population standard deviation when
fewer than two quarters can be measured. A top-k list takes the highest scores, at most two from one group.

### Stability

For every person, all only-you statements made from the first half (reference: the other 303 first halves) are
re-checked on the second half. Reported: the share still true, the share still true and still true of at most
10% of the others' second halves, the share that cannot be checked (the family cannot be measured on the second
half, counted as not holding), and the same for each person's top 3. No pass bar; this is descriptive.

### The lineup test

For each of the 304 targets, 20 lineups. Each lineup is the target and 4 decoys drawn at random from the other
303 (seed 5 + k). The target's top-k statements are made from the target's first half against the first halves
of the 299 people outside the lineup, so the decoys' data plays no part in choosing the statements. A blind
matcher counts, for each of the five people, how many of the statements are true on that person's second half,
and picks the highest count. A tie is broken at random; the reported accuracy uses the expected value of the
tie-break (an m-way tie that includes the target scores 1/m). A target with no qualifying statement scores 1/5. A
target's accuracy is the mean over its 20 lineups, and the overall accuracy is the mean over targets.

- Chance is 20%.
- 95% interval: bootstrap over targets, 4,000 resamples, percentile interval.
- Permutation null: the second halves are shuffled across the 304 people (statements and decoys unchanged), 1,000
  times; p = (1 + number of shuffles at or above the observed accuracy) / 1,001.
- k = 3 is the primary test; k = 5 is reported alongside.

Pass bar, fixed now. The statement engine passes when top-3 lineup accuracy is at least 60%, the lower end of its
95% interval is above 40%, and the permutation p is below 0.01. Anything less is reported as a fail.

### The Barnum control

The same lineups with Barnum statements in place of only-you ones: for each family and direction, the threshold
that is true of the target and of between 50% and 90% of the reference, the one the target meets most surely
(highest confidence); top k by confidence with the same two-per-group rule. These are the horoscope sentences,
true of the person and of most people. Expected: accuracy near chance, at most 30%. If the Barnum control
reaches 30% or more, that is reported as a finding: generic statements would then carry some identity.

### Coverage and example cards

Reported on whole histories (reference: the other 303 whole histories): the share of people with at least 1, 3
and 5 only-you statements, and the same on first halves. Three people are drawn at random (seed 5) for example
cards; their pseudonymous ids are cut to 8 characters and no artist name, username or track appears in the
cards. The most striking statements are listed separately and are picked by how few others they are true of,
which is a selection and is labelled as one.

## Results

Run on 5 November 2026 with `run_statements.py all` (48 seconds on a laptop). Every number below is in
`docs/statements_results.json`. After the run, two display fixes were made to the sentences and nothing else: values
in hours and days are printed to one decimal, and the artist streak sentence no longer read "56 days or more days".
No threshold, family or rule changed, and the numbers are the same before and after.

### The verdict: fail, narrowly

Three statements made from the first half of someone's history pick out their second half from a lineup of five
55% of the time, against 20% by chance. That is far above chance and below the 60% the test was set to clear, so
the statement engine fails its pre-registered bar. The lower end of the interval (51%) is above 40% and the
permutation p is at its floor, so the miss is on the size of the effect alone.

| test | accuracy | 95% interval | shuffled second halves (mean, p) |
|---|---|---|---|
| only you, top 3 (primary) | 54.9% | 50.7% to 58.9% | 20.0%, p = 0.001 |
| only you, top 5 | 57.6% | 53.3% to 61.8% | 20.1%, p = 0.001 |
| Barnum, top 3 | 31.9% | 30.7% to 33.2% | 20.0%, p = 0.001 |
| Barnum, top 5 | 38.3% | 36.7% to 39.8% | 20.0%, p = 0.001 |

The shuffle lands on 20% every time, which is what it should do: with somebody else's second half in the target's
place, the statements have nothing to find.

What costs the test is coverage. Statements are made against the 299 people outside each lineup, and 34 of the 304
targets had no statement true of them and of 10% or fewer of the others in any of their lineups; by the rules they
score chance. In 36% of lineups the target had fewer than three statements. When the target did have three, the matcher
found them 66.6% of the time (3,871 lineups), and for the 191 targets who had three in every lineup the accuracy is
66.7%. With two statements it was 49.8%, with one 36.9%. Those splits were not pre-registered and are reported to
explain the miss. They do not rescue it.

### The Barnum control: above its expected ceiling

The horoscope sentences (true of the target and of 50% to 90% of others) found the target 31.9% of the time with
three and 38.3% with five. The pre-registration expected 30% or less, so this is the finding it said it would
report: a handful of sentences that each fit most people still narrow a lineup when they are combined, because each
one rules out the people it does not fit. Even so, the specific statements are clearly better. Three only-you
statements do better than five Barnum ones (54.9% against 38.3%), and the target is alone at the top of the count in
43% of only-you lineups against 4% of Barnum lineups.

### Coverage

On whole histories, against the other 303 listeners, 85% of people have at least one only-you statement, 64% have
three or more and 46% have five or more (median 4). On first halves the figures are 67% with three or more and 50%
with five or more. Every one of the 34 families produced statements for between 21 and 60 people, so no single
family carries the result.

### Stability

The first halves gave 1,591 only-you statements. 46.5% of them are still true on the second half, and 40.7% are
still true and still true of 10% or fewer of the others' second halves. One could not be checked. Of each person's
top three, 47.8% hold. For comparison, the same statements are true of 7.1% of the other listeners' second halves,
so a statement that held is about six and a half times as likely to describe its own person as a stranger.

Half failing looks worse than it is, and the reason is in the design. The filter puts the threshold just inside the
person's value, at the edge of the top or bottom tenth of the population, so ordinary drift between halves pushes
many people back across the line. The confidence score sees some of it (not pre-registered, from the same run):
statements with confidence 0.5 to 0.7 held 38% of the time, 0.7 to 0.9 held 53%, and 0.9 or more held 55% to 58%.
Families differ too. The most stable were sessions per day (64% held), artist run length (63%), album runs (62%) and
same song within a day (59%). The least stable were the two adjacent days (19%), back to back repeats (24%) and the
artist day streak (29%).

### How much the sentences give up

As a reference (not pre-registered), the same lineups were matched on all 34 family values at once, choosing the
person whose second half is nearest to the target's first half in mean absolute z. That finds the target 86.8% of the
time (83.9% to 89.4%). So the halves of one person's history are very alike, and almost all of that likeness is in
patterns a person could read. Three threshold sentences keep a good part of it (55%); the rest is lost to the
threshold, which throws away how far past it a person is, and to the people with fewer than three statements.

### Example cards

Three listeners drawn at random (seed 5), whole histories against the other 303. In the public copy the
pseudonymous ids are replaced by letters. "True of" counts the other listeners the same sentence fits.

Listener A, 17,363 plays over 1.7 years. One statement:

- Your listening day moves: a typical day's centre sits at least 5 hours from your usual time. Yours: 5.5 hours.
  True of 8 of the 303 others (2.6%), confidence 0.71.

Listener B, 56,865 plays over 2.2 years. Three statements:

- Your listening keeps a fixed time: a typical day's centre sits within 1.5 hours of your usual time. Yours: 1.2
  hours. True of 28 of 303 (9.2%), confidence 0.97.
- Your typical session runs 60 minutes or longer. Yours: 67 minutes. True of 28 of 303 (9.2%), confidence 0.85.
- No more than 7.5% of your plays go to artists you first heard in the four weeks before. Yours: 7.1%. True of 18
  of 303 (5.9%), confidence 0.53.

Listener C, 57,076 plays over 13.7 years. Two statements:

- At least 7.5% of your plays are the same song again, straight after itself. Yours: 8.7%. True of 28 of 303
  (9.2%), confidence 0.58.
- Your typical session is 20 minutes or shorter. Yours: 20 minutes. True of 30 of 303 (9.9%), confidence 0.50.

These are typical. Random people mostly get two or three statements near the 10% line, some with low confidence,
and they are true and dull. The striking ones belong to people at the far edge of a family. Sorted by how few others
they fit (a selection, and labelled as one; the full card files are kept out of the public repo):

- One listener: two days in a row each week hold at least 55% of your plays (yours 72%). True of none of the 303 others.
- Another: your listening day moves, a typical day's centre sits at least 6 hours from your usual time (yours 7.4
  hours). True of 1 of 303.
- Another: every season some artist gets played on 56 or more days in a row (yours 75). True of 2 of 303.
- Another: at least 90% of your plays happen in sessions of four hours or more (yours 97%). True of 3 of 303.

The single rarest statement in the bench is "in a typical season one artist takes at least 65% of your plays", for
a listener at 100%. That account plays one artist and nothing else, probably an odd use of the scrobbler, and the bench's cleaning
rules (one track over 30%) do not catch it.

## Limits

- The bar was missed. The engine as specified says something specific about most people and identifies them well
  above chance, and it does not reach the 60% that was set as "wow". Raising coverage (more families, or a 15% line
  for people with nothing under 10%) or keeping only statements with high confidence are the obvious next tries, and
  each would need its own pre-registration because these results have now been seen.
- Half the statements made from one half of a history do not survive to the other. A person shown a card should
  read it as true of the period it was measured on.
- The reference population is 304 ListenBrainz users: enthusiasts, self-hosters, Last.fm veterans, heavy on rock,
  metal and electronic. "True of 2% of listeners" means 2% of them. A Spotify listener compared with this bench
  would look different on any family that depends on how plays are logged (repeats and sessions especially).
- No time zone, so nothing about the hour of the day or the day of the week by name. "Night owl" and "weekend
  listener" statements would likely be the most recognisable kind, and this data cannot make them.
- Halves differ in length from person to person (six months to several years each), and families that need long
  windows (old friends, season turnover, dropped cold) have fewer chances to qualify for people with short
  histories.
- Families are correlated (old friends against new share against turnover, for instance), so three statements
  from different families are not three independent pieces of evidence. The two-per-group rule limits this and does
  not remove it.
- Nothing has been shown to a person. Whether a true, rare statement feels surprising to the person it describes is
  untested.

## Round 2: pre-registration (committed before the fresh listeners were analysed)

Round 1 missed its bar, and the reasons it missed were found by looking at the 304 listeners after the
test. Anything tuned on those people would flatter itself if it were scored on them again, so round 2
was built on the old listeners and is scored once on listeners nobody has looked at. This section was
committed while the fresh listeners were still downloading and before any of their histories had been
measured. The code it names is frozen at that commit (`src/aas/statements/v2.py`,
`src/aas/statements/families_v2.py`, `scripts/run_statements_v2.py`), and the script refuses to make
the confirmation run a second time.

### What was tried on the old listeners

Every number in this list comes from round 1's listeners, with one account removed (the one that plays
a single artist, see below), 303 people, the same lineups of five and 20 lineups per target. These are
exploratory and were used to choose the rule, so they overstate how well it does.

The round 1 rule on these 303 people scored 55.1%. Adding 23 families (57 in all) raised it to 57.9%,
mostly by giving more people a statement: the share of lineups where the target had fewer than three
fell from 36% to 23%. Weighting the score by how well each family holds from one half of a history to
the other (its split-half correlation, squared) gave 59.5%. Letting people with nothing under the 10%
line fall back to 15% and then 20% gave 62.4%, and preferring thresholds the person clears comfortably
(confidence cubed in the score) gave 64.6%. Moving the line itself to 20% while keeping both of those
preferences gave 68.9%; a 15% line with a 20% fallback gave 67.2% and a 25% line 68.8%. Dropping
families with low split-half correlation, a confidence floor, a matcher that weights statements by
rarity or by confidence, and other limits per group all did no better. Requiring a longer first half
(3,000 plays and nine months) gave 66.2% against 64.6% on the same rule, on 268 people instead of 303,
and was not adopted. For the fingerprint, a mean absolute difference of percentiles found the target
90.4% of the time, against 87.0% for round 1's z scores, and weighting families by reliability changed
nothing.

### The fresh listeners

The same walk over ListenBrainz's similar-users graph, continued from where round 1 stopped on 5
November 2026 with the same rules (5,000 to 2,000,000 listens, a year or more between first and latest
listen, a listen in the last 120 days). It qualified 817 accounts in all. The fresh set is every
qualifying account that round 1's download never started, in pseudonym order, up to 370 accounts.
They are downloaded the same way (`LB_SET=fresh`, newest first, up to 60,000 listens and extended to
two years for heavy listeners), stored under `bench/data/lbusers/raw_fresh` and `plays_fresh`
(gitignored), pseudonymised with the same salt, and cleaned with round 1's rules plus one new
exclusion, applied to both sets: an account is dropped when one artist takes more than 75% of its
listens or it has fewer than 50 different artists. On the old listeners this removes exactly the
account that plays one artist and nothing else. Then round 1's person rule: 2,000 or more listens, the
last 8 weeks dropped with 1,000 or more listens before them and 200 or more inside them. Whoever is
left is the fresh set; the count is reported, accounts that fail to download are left out and none is
replaced by hand. If fewer than 200 people are left, the run waits for more downloads before it is
made.

### The frozen rule

57 families: round 1's 34 and 23 new ones in five new groups (catalogue, variety, returns, pace, day)
plus one each in discovery, rhythm and repetition. The new ones are the share of plays that are songs
first played in the four weeks before, songs first played over a year before, artists played for over
two years, songs come back to after six months away, how many new artists get five plays on their first
day, how many of a season's songs are played once, how much of a season is different songs, the top ten
songs' share, the share of plays to artists played once or twice in a season, artists per season, the
typical gap before a song or an artist comes back, the typical gap between plays inside a session (about
the length of a track), plays per session, single-song sessions, artists per long session, hours between
sessions, how long a listening day runs from first play to last, how many hours of a day have music,
how often a day is three times the usual, the longest break in a season, how often the week's busiest
day is the usual one, and the share of plays in sittings of eight or more from one release. All are
shares, rates or typical lengths with grids of round numbers; none names anything, and none reads the
clock hour or the weekday name.

A statement now qualifies when it is true of the person and of at most 20% of the reference
population, a fifth instead of a tenth. That is a real change in what "only you" means, made because
the 10% line left too many people with too little to say. The surprisal in the score still rewards
rarer statements, and the cards print how many others each one fits. Score is confidence cubed times
surprisal times the family's frozen split-half correlation squared (the correlations are written into
`v2.py`). One statement per family, a top-k by score with at most two per group, the same confidence
and the same matcher as round 1.

### Tests on the fresh set

The primary test is round 1's lineup with the v2 rule: for each fresh target, 20 lineups of the target
and four decoys from the other fresh listeners (seed 5 + k), top-3 statements made from the target's
first half against the first halves of everyone outside the lineup, a blind count of how many hold on
each second half, ties shared. It passes when top-3 accuracy is at least 60%, the lower end of its 95%
bootstrap interval (4,000 resamples over targets) is above 50%, and the permutation p (second halves
shuffled, 1,000 times) is below 0.01. Top 5 is reported alongside.

The first secondary test is the fingerprint: on the same lineups, every family value of every first and
second half is put on the percentile scale of the fresh first halves, and the matcher picks the person
whose second half is nearest the target's first half in mean absolute percentile difference over the
families both can be measured on. It passes at 75% or more.

The second secondary is a stand-in for "a reading a person would recognise": the share of fresh people
with three or more statements made from their first half (against the other first halves) that are
still true on their second half and still true of at most 20% of the others' second halves. There is
no pass bar; on the old listeners it is 75%, and a figure well below that on fresh people would say the
statements were fitted to the old ones.

The Barnum control is round 1's (the horoscope statements true of 50% to 90% of others, ranked by
confidence) over the 57 families. Round 1 found it above its expected 30% ceiling, and on the old
listeners with 57 families it scores 32.3%, so it is expected near 32%. It is reported and is not a gate.

Coverage on whole histories and the stability of first-half statements are reported as in round 1,
with the share of selected statements that are true of 10% or fewer of others, so the cost of the 20%
line is visible.

### Cards

`docs/statements_cards.md` (written locally, kept out of the public repo) is rewritten from the fresh run: three fresh listeners drawn at random with
seed 5 and, separately and labelled as a selection, the three others whose top three statements are
together the rarest. Each card shows the top three statements on the whole history, with ids cut to
eight characters and no artist, track, release or username.

### The frozen rule on the old listeners (exploratory)

Run with the pre-registered seeds before the fresh run (`run_statements_v2.py old`, in
`docs/statements_v2_results.json` under "old"): top-3 lineup accuracy 69.7% (66.0% to 73.2%), top 5
72.3%, fingerprint 90.1% (87.7% to 92.4%), Barnum 32.3% with three and 38.2% with five, 75% of people
with three or more stable statements. These numbers were made on the people the rule was chosen on.

## Round 2: results on the fresh listeners

Run once on 5 November 2026 with `run_statements_v2.py fresh`, after the pre-registration above was
committed and pushed (commit 33792a8). Every number is in `docs/statements_v2_results.json` under
"fresh". Nothing was changed between the commit and the run.

### Who was in it

All 370 fresh accounts downloaded without a failure. Cleaning kept 359 of them: 7 were dropped for
listens faster than listening, 5 for 500 or more listens on too many days, 2 for scripted pacing and 1
each for bulk-import timestamps, one track dominating, too few listens and under a year of history
(some hit more than one rule). None was caught by the new single-artist rule. Round 1's person rule then
left 333, after 26 had too little listening before or inside their last 8 weeks. The fresh set is those
333 people; none of them is in round 1's bench.

### The verdict: pass

Three statements made from the first half of a fresh listener's history pick out their second half from
a lineup of five 67.5% of the time, against 20% by chance and a 60% bar. The interval runs from 63.8% to
71.0%, so its lower end clears the 50% the pre-registration asked for, and none of the 1,000 shuffles
came near (p = 0.001, the floor). The statement engine passes on people it was not built on. It scored
69.7% on the old listeners it was tuned on, so the drop to new people is about two points.

| test (fresh, 333 people) | accuracy | 95% interval | shuffled (mean, p) |
|---|---|---|---|
| top 3 statements (primary) | 67.5% | 63.8% to 71.0% | 20.0%, p = 0.001 |
| top 5 statements | 71.5% | 68.0% to 74.8% | 20.0%, p = 0.001 |
| fingerprint, all 57 families | 89.5% | 87.1% to 91.7% | |
| Barnum, top 3 | 31.7% | 30.3% to 33.0% | 20.0%, p = 0.001 |
| Barnum, top 5 | 38.7% | 37.0% to 40.5% | 20.0%, p = 0.001 |

Coverage, which sank round 1, is no longer the problem: the target had fewer than three statements in
1.3% of lineups (36% in round 1), and in 57% of lineups the target was alone at the top of the count.

### The fingerprint

Matching on every family at once finds the target 89.5% of the time, well over its 75% bar and close to
the 90.1% on the old listeners. The halves of one person's listening stay alike on the measures a
person could read, on new people as on old ones. Three sentences recover about two thirds of what the
full profile carries.

### A reading a person would recognise

75% of fresh people (74.8%) have three or more statements made from the first half of their history
that still hold on the second half and still fit a fifth or fewer of the others there. On the old
listeners the figure was 75.2%, so this did not shrink on new people. Of all statements made from first
halves, 57% held on the second half (round 1: 46.5%).

### What the 20% line costs

The pass came partly from widening "only you" from a tenth to a fifth of listeners. On whole histories
nearly everyone has something to be told (98% have three or more statements, median 16), but only 39% of
the statements on a person's card are true of 10% or fewer of the others, and 47% of people have three
or more at the old 10% line. So a typical card line fits somewhere between one in ten and one in five
other listeners. A card says how many others each sentence fits, so the reader can see this.

### The Barnum control

The horoscope sentences scored 31.7% with three and 38.7% with five, the same as round 1 and the same as
on the old listeners. As before, a handful of sentences each true of most people still narrows a
lineup a little, and the specific statements are far ahead of them (67.5% against 31.7%, with the
target alone on top in 57% of lineups against 5%).

### The cards

`docs/statements_cards.md` (local only) shows three fresh listeners drawn at random and three picked as striking.
The random three each have three statements, all true of under 10% of the others and most with
confidence near 1: one listener's days run 19 hours from first play to last, another meets about 600
new artists a month and gives only 6% of plays to artists that were favourites half a year before, a third comes back to music seven
times on a listening day and repeats songs heavily. The striking three include two unusual accounts (one
with music on 1.4% of days over 14 years, almost all of it new songs; one where 99% of plays fall in the
same four hours and a day has about one session). They passed every cleaning rule, but they may be
logging habits more than listening habits, which is another reason the striking cards are labelled as a
selection. The cards as first written printed some values without a unit ("Yours: 6.62" where 6.6
minutes was meant). That was fixed afterwards in the display only, and the cards were redrawn from the
run's cached measures (`run_statements_v2.py cards`); the same people and statements appear, and no
score or verdict was recomputed.

### The frozen rule on the old listeners (exploratory, for comparison)

On the 303 old listeners the rule was chosen on: top 3 69.7% (66.0% to 73.2%), top 5 72.3%, fingerprint
90.1%, Barnum 32.3% and 38.2%, 75.2% with three or more stable statements, 56.5% of first-half
statements holding. The fresh numbers sit a little below these on every count, which is the expected
cost of having tuned on the old people, and small.

### Limits of round 2

- The pass is on ListenBrainz people found the same way as round 1's (the similar-users walk), so the
  fresh set is new people from the same kind of population. Other kinds of listener are untested.
- "Only you" now means a fifth of listeners or fewer. At the 10% line the round 2 engine was not run on
  the fresh set as a primary test, so nothing here says it would pass there.
- 61% of the statements on cards fit between 10% and 20% of others, which a reader may find
  unsurprising. Whether a card feels like it describes its reader is still untested on any person.
- The stable-reading figure is a proxy for recognition, measured on the data alone.

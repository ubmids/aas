# Recognition test: can people pick out their own card?

The statements and identity rounds showed that three sentences about how someone listens pick that
person's later listening out of a lineup, when a program does the matching. This test asks the human
version of the question: shown their own card and two other people's, does a person pick their own?
It is small (5 to 10 friends) and exploratory. It cannot confirm anything; it can tell us whether the
cards mean something to the people they describe, and how much of that is the Barnum effect (people
seeing themselves in anything vague enough).

The tool is `scripts/my_card.py`; the card logic is `src/aas/statements/card.py`; the tests are
`tests/test_my_card.py`.

## What a card says

Three sentences, chosen with the identity round's joint rule, each with the person's own value and
units and the share of 636 reference listeners it is also true of. Then two lines from the music
reader: the explore or stay dial and the old favourites line. No artist, track, album, play count,
date or place appears on a card. Example, for the invented person in `src/aas/example.py`:

    1. No more than 2% of your plays sit in runs of ten or more by the same artist.
       Yours: 0%. True of 6% of listeners.
    2. You play music on at least 98% of all days.
       Yours: 100%. True of 11% of listeners.
    3. Your songs run long: inside a session a new play starts 4.25 minutes or more after the one before.
       Yours: 7.4 minutes between starts. True of 12% of listeners.

    Explore or stay: you lean toward staying with old favourites (-2.36, where 0 is the typical listener). Further toward staying than 99% of listeners.
    Old favourites: 83% of your plays go to artists who were already in your top 50 four to six months before. Higher than 99% of listeners.

## Running it

Everyone needs their own listening history: the Spotify extended streaming history (spotify.com,
Account, Privacy settings, "Extended streaming history"; it takes a few days to arrive) or a
ListenBrainz export (listenbrainz.org, Settings, Export). A card needs about 2,000 plays; the dial and
the old favourites line need eight or nine months of history.

There are two ways to collect cards. The second keeps every export on its owner's machine.

1. The organiser collects exports into one folder, one per person, named by a label (for example
   `ada.zip`, `ben/`, `cy.json`).
2. Each person runs, on their own computer,

       cd py
       uv run python scripts/my_card.py PATH_TO_EXPORT --save-card ada

   which prints their card and writes `ada.card.json`. That file holds the card's text and a Barnum
   card's text, nothing else. They send only that file.

Then the organiser runs

    uv run python scripts/my_card.py --lineup FOLDER --out sheets --barnum --seed 1

which writes `sheets/sheet_<label>.txt` for each person, `sheets/answer_key.csv` and an empty
`sheets/answers.csv`. Each sheet shows the person's own card and two other people's in shuffled order,
labelled A, B and C. People sit in a random cycle and each sees the next two people's cards, so every
card appears as a decoy exactly twice. With `--barnum` the sheet ends with a fourth card, D, to rate
but not to pick: three sentences that are true of the person and of 50% to 90% of the reference
listeners (round 1's Barnum band), most widely shared first, with the same two dial lines.

Give each person their sheet alone, before they have seen anyone else's card, and ask them not to
compare sheets until everyone has answered. The organiser should not stand over them or hint. Fill
in `answers.csv` (the pick as A, B or C, and each rating 1 to 5), then

    uv run python scripts/my_card.py --score sheets/answers.csv --key sheets/answer_key.csv

## Consent

Read or send this before anyone shares anything:

> This is a small test of a program that describes how people listen to music. If you take part, you
> make a card from your own Spotify or ListenBrainz listening history. The program runs on a computer,
> not online, and uploads nothing. You can make the card on your own computer and send only the
> card, a few sentences with numbers and no artists, songs, dates or places. If you would rather send
> your export file, it is used only to make the card and is deleted when the cards are made. You then
> see three cards, one of them yours, and say which you think is yours and how much each one sounds
> like you. Two other people in the test will see your card, without your name on it. Only your answers
> are kept, under a label, not your history. You can stop at any time, and ask for your answers to be
> deleted, without giving a reason.

## What is recorded

Per person: the label, which card they picked, and a rating from 1 (not me at all) to 5 (exactly me)
for each card on the sheet, including the Barnum card when it is used. The answer key records which
letter was whose. Nothing else: no history, no measures, no card values outside the sheets. If exports
were collected, they are deleted once the sheets are made.

## The bar, set before anyone takes part

This section is committed before any real person's card is made.

Primary: the share of people who pick their own card is at least two thirds, twice the one in three
expected by chance. Reported with the one-sided binomial p against one third and its 95% interval. With
10 people that is 7 or more right (p = 0.02 if picks were at chance); with 5 people it is 4 or more
(p = 0.045).

Secondary, reported without a bar: the mean rating of the own card against the mean rating of the two
decoys; and, with `--barnum`, the own card's rating against the Barnum card's. If people rate the Barnum
card as highly as their own, the ratings say little and only the pick counts.

This is small n and exploratory. Ten people can show a large effect or a clear absence of one, nothing
in between: one person more or less right moves the share by ten points. Friends who know each other's
habits may recognise a decoy as someone else's, which helps the pick without the card describing
anyone. A pass is a reason to run a proper test, not a result.

## How the card is made

The 57 round 2 measures are taken on the whole export and on its two halves in time. Spotify plays
under 30 seconds are left out first (ListenBrainz only records listens, so this keeps the two sources
alike); a ListenBrainz artist with a single MusicBrainz id is keyed by that id, as the bench does.

The three sentences come from `joint_select` in `src/aas/statements/identity.py`, imported unchanged:
greedy choice of the set that most raises expected credit in a lineup of five (four decoys), at most
one sentence per measure and two per group, with no rarity line. That rule needs every reference
person's truth and hold probability for every sentence, and the 636 reference listeners (rounds 1 and
2) cannot be shipped one by one. So the card ships aggregates only, in `src/aas/data/card_norms.json`
(built by `scripts/build_card_norms.py` from the local bench caches): per measure, the 1st to 99th
percentiles on whole histories and on halves, how many could be measured, the split-half correlation of
normal scores; per sentence, how many of the 636 it is true of and the threshold's own percentile; and
the 57 by 57 correlation of normal scores. From these the card draws a stand-in reference of 2,000
synthetic people (a Gaussian copula with the real correlations, each measure missing at its real rate,
each sentence true with exactly its real share), with each synthetic person's hold probability from the
identity round's formula at the measure's typical noise. The real person gets the full formula, their
two halves standing in for H1a and H1b.

This is an approximation of the frozen rule, not the rule itself. Dependence beyond pairwise
correlation is lost, missing values are not tied to history length, and the person is compared on a
whole export (often many years) with reference histories of the training weeks (a median of about four
years). Reference listeners are ListenBrainz users, who record more deliberately than the average
Spotify listener. The shares printed on cards are the exact counts among the 636, not the stand-in's.
When `identity.py` is reconciled after the third-set confirmation, `card.py` should be checked against
it; it imports `joint_select`, `all_statements`, `KAPPA` and `LAMBDA` and nothing else.

The explore or stay dial and the old favourites figure come from the music reader
(`docs/reader_music.md`) and its own norms (304 listeners).

# AAS

Show me your digital world and I'll tell you who you are.

## What it is

Every big app runs an algorithm that seems to know you. Spotify knows the song you'll want next, TikTok knows what will keep you scrolling. Those algorithms are built from your data, but you never get to see what they've worked out about you, or how sure they are.

AAS turns that around. You download your own data from the services you use (Spotify, Google, YouTube, Reddit, X all let you request a copy), and AAS reads it on your own machine and tells you what it can honestly say about you. Nothing is uploaded.

It's a research engine, not an app yet. The point of this first version was to find out which things can really be read from a person's data and which can't, and to build the reading so that it never claims more than it can back up.

## The rule it follows

Anyone can write a reading that sounds true. "You're loyal to the things you love, but you crave something new now and then" fits almost everybody, which is why horoscopes work. AAS is built to avoid that.

So every reader in AAS has to earn the right to speak. Before it's allowed to say anything about you, it's tested on hundreds or thousands of people it never saw during training, and its accuracy is measured against a plain baseline such as guessing the average. When it does speak, it gives a value, a range it expects you to fall in, and the accuracy it reached in that test for your kind of data. When your data can't support an answer (too little of it, or a kind it was never shown to work on), it says so and explains why instead of guessing.

## How it works

1. Parsers read the export files each service gives you and turn them into one shared shape: things you wrote, things you played, things you watched, each with a time.
2. Readers look at one kind of data each. The text readers read your writing, the music reader reads your listening history, and the listening engine models how you play music day to day.
3. Each reader returns estimates: a value, a range, its tested accuracy, and a sentence explaining where the number came from.
4. A fusion step combines readers that answer the same question, weighting each by its own tested accuracy.
5. The profile puts it all together and lists what each reader said and what it declined to say.

## What you get back

Here is part of what it says for an invented test person (all the data in this example is made up, see `py/docs/example_profile.md` for the full output):

```
music  old_friends  83% of your plays went to artists who were already among your
                    top 50 four to six months before. That's more than 99% of the
                    304 ListenBrainz listeners we measured.
music  dial_explore Explore against stay: -2.36, so you lean staying with old favourites.
text   openness     not shown to work on this kind of writing, so no number is given.
                    It has only been shown to work on essay-like long prose, which is
                    22% of your words (at least 50% needed).
```

The listening lines are sharp and specific. The personality lines, for most people, are honest but weak, and the section below says why.

## What it found so far

Listening behaviour is very predictable from listening behaviour. Whether you will skip a track, whether a new artist will still be in your rotation a few weeks later, and how adventurous or loyal you are as a listener can all be read from a year of plays, and they hold up on held-out weeks. On 13 detailed Spotify histories the skip model beats each person's own per-artist skip rate for all 13 (AUC 0.85 against 0.62). On 301 public ListenBrainz histories the "will it stick" model reaches AUC 0.79 after the first session, against 0.62 for popularity. Listening traits fall onto two stable dials, exploring against staying with old favourites, and how fixed your listening hours are.

Personality is much harder, and the honest answer is that the signal is small. Language carries most of it. A bag of words trained on student essays still reaches AUC 0.54 to 0.57 on tweets, and the fused v1 reader reaches a correlation of about 0.16 with the Big Five in simulation, with intervals that are calibrated but wide. Phone sensors, a fitness tracker and Steam profile counts showed nothing that survives correction at 46 to 149 people. None of the readers has yet been checked against real people's own questionnaires; `py/docs/study/` holds the protocol for that study.

The stage after that has started, and two results came back on 5 November 2026. The first is about statements: sentences true of you and of few others, such as "your songs run long: inside a session a new play starts 5.5 minutes or more after the one before", each with how many other listeners it also fits. To check that such statements describe a person, three statements made from the first half of someone's history are shown with the second halves of five people, and a blind judge has to pick the right one, so chance is one in five. The bar was set in advance at 60%. Round one missed it at 54.9%. The rules were then changed (more measures, a looser rarity line) and frozen, and the frozen version was run once on 333 new ListenBrainz listeners nobody had looked at. It picked the right person 67.5% of the time (95% interval 63.8% to 71.0%). Comparing people on all 57 measures at once, the fingerprint, picks them 89.5% of the time (87.1% to 91.7%). Generic statements true of most people, the horoscope kind, pick them 31.7% of the time. Details are in `py/docs/statements.md`.

The second is a prediction game. After one session with each of three new artists, the model names the one that will still be in your rotation about 69% of the time, against one in three by chance and 37% to 46% for picking the most popular. It beats popularity for all 9 eligible Spotify people and 132 of 149 ListenBrainz listeners, which clears the bar set in advance. Much of that skill is simply the artist you played most in its first session, which does as well on Spotify and only a little worse on ListenBrainz. The other two games missed: calling which of the next 10 tracks you will skip beats the better simple rule for 9 of 12 people where 11 were needed, and naming next week's top five beats your last four weeks for only 56% of ListenBrainz listeners. The rules and every number are in `docs/GAME.md`.

A third result came back on 6 November 2026, about how few sentences it takes to tell people apart. Choosing the three sentences together, instead of one at a time, avoids sentences that repeat each other, and a hold probability fitted on the first half allows for extreme people drifting back toward the middle. The bar set in advance was 75% in lineups of five, on a third set of 352 ListenBrainz listeners in neither earlier set. It failed: 68.8% (65.5% to 72.1%). The improvement itself held. On the same people and lineups the joint choice beat round two's rule by 7.2 points (4.2 to 10.4), and from three sentences it found one person among 352 13.6% of the time against 8.4% (chance is 0.3%). Round two's rule scored 61.5% on this crowd against 67.5% on the last one, so absolute levels move between population draws and the fair comparison is head to head on the same people. Artist lists alone still pick the person out of 352 94.0% of the time. Details are in `py/docs/identity.md`, with related work in `py/docs/related_work_identity.md`.

`py/scripts/my_card.py` turns your own Spotify or ListenBrainz export into a listening card of these sentences, scored against aggregate norms in `py/src/aas/data/card_norms.json`. `py/docs/recognition_test.md` describes the test of whether people recognise their own card.

The full account, including what failed, is in `docs/REPORT.md` and `docs/MATH.md` for the engine and `py/README.md` and `py/docs/MATH.md` for the Python side.

## What's next

Personality scores this weak read like a horoscope for any one person, so the work now leans on what holds up: statements that are true of one person and of few others, and predictions people can check for themselves. Both have passed a test on data, and the identity round showed that bars for these tests are better set as head to head gains on the same people than as absolute levels. The next test is with real people: shown their own card next to other people's, can they pick out their own? The protocol is in `py/docs/recognition_test.md`.

## Layout

- `engine/` is the TypeScript engine. `engine/src/parse/` reads Spotify exports and `engine/src/model/` holds the models (skip, taste map, will it stick, selves, eras, traits).
- `bench/` is the evaluation harness for the engine. It needs listening histories that are not included here (see below).
- `py/` is the Python package `aas`: export parsers, text readers, the music reader, fusion, the IPIP quiz, the feature record a participant would share, and the study design.
- `docs/` and `py/docs/` hold one write up per experiment, with the numbers and how to reproduce them.
- `DESIGN.md` and `SPEC.md` are the working design notes, kept as written.

## Install and run

You need Node 20 or newer and [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 for you).

```
npm ci
npm test                 # engine unit tests

cd py
uv sync
uv run pytest            # Python tests
```

The Python tests that need a trained text reader skip with a message saying which script builds it, so a fresh checkout passes with about two dozen skips.

Some things run straight away on synthetic or bundled data:

```
cd py
uv run python scripts/take_quiz.py           # the 20 question IPIP quiz, scored on 597k person norms
uv run python scripts/inspect_export.py ~/Downloads/<your export>.zip
uv run python scripts/extract_features.py ~/Downloads/<your export>.zip --out record.json
```

Everything else needs a dataset first. Put downloads under `data/raw/<name>/` at the repository root (the folder is gitignored) and follow `DATA.md`. Then, for example:

```
cd py
uv run python scripts/train_text_readers.py --only bigfive   # needs Essays
uv run python scripts/train_text_readers.py --only mbti      # needs the Kaggle MBTI set
uv run python scripts/example_profile.py                      # needs both readers
```

The engine bench (`npm run bench`, `npm run lb3`) needs listening histories in `bench/data/`, described in `DATA.md` and `bench/lbusers/README.md`.

## What is not included, and why

No dataset is redistributed. Each one has its own terms and some forbid sharing, so `DATA.md` says where to get each one instead.

The listening histories behind the engine bench are not included, nor is any list of whose they are. The Spotify bench came from people who had published their own streaming history in public GitHub repositories, and the ListenBrainz bench from public ListenBrainz profiles. Even though the data was public, a list naming those people next to an analysis of their habits is not something to publish, so only aggregate results and pseudonymous summaries are kept (the music reader's norms in `py/src/aas/data/music_norms.json` are percentiles over 304 people, and the card's norms in `py/src/aas/data/card_norms.json` are percentiles and correlations over 636, both with no rows).

Three sets of model weights are left out, while their model cards (`card.json`) and the scripts that build them are kept:

- `py/models/text_bigfive/` is trained on the Essays dataset, which has no stated licence, so the weights are not redistributed.
- `py/models/text_mbti/` is trained on forum posts, and its vocabulary picked up a few forum usernames that people quoted in their posts. Rebuilding it locally is fine; shipping it is not.
- `py/models/text_distilled/` is trained on Claude's ratings of forum users. Anthropic's terms restrict using model outputs to build competing models, and while a five number regressor is unlikely to count, it has not been checked, so the weights stay out. It also failed its own pre-registered test and is not a default reader.

The NRC emotion lexicon used by one text experiment is free for non-commercial research only and is not included; `DATA.md` says where to get it.

## Licence

MIT, see `LICENSE`. A few small word lists under `py/src/aas/data/` keep their own licences, listed there and in `DATA.md`.

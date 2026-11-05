# AAS

AAS reads the data downloads people can already request from the services they use (Spotify, Google, YouTube, Reddit, X) and tries to say true things about the person behind them. It runs on the person's own machine. Nothing is uploaded, and every claim it makes is first scored on people the model never saw, against a plain baseline, before it is allowed to show up.

The project has two halves. The TypeScript engine reads listening histories: which tracks you will skip, which artists you will play next, whether a new artist will stick, when you listen, and a handful of listening traits measured straight from your plays. The Python package asks the harder question of whether any of this, or your own writing, says something about personality, and it reports how well each reader does on kinds of data it was not trained on. Where the data cannot support a number, it refuses and says why.

## What it found

Listening behaviour is very predictable from listening behaviour. Whether you will skip a track, whether a new artist will still be in your rotation a few weeks later, and how adventurous or loyal you are as a listener can all be read from a year of plays, and they hold up on held-out weeks. On 13 detailed Spotify histories the skip model beats each person's own per-artist skip rate for all 13 (AUC 0.85 against 0.62). On 301 public ListenBrainz histories the "will it stick" model reaches AUC 0.79 after the first session, against 0.62 for popularity. Listening traits fall onto two stable dials, exploring against staying with old favourites, and how fixed your listening hours are.

Personality is much harder, and the honest answer is that the signal is small. Language carries most of it. A bag of words trained on student essays still reaches AUC 0.54 to 0.57 on tweets, and the fused v1 reader reaches a correlation of about 0.16 with the Big Five in simulation, with intervals that are calibrated but wide. Phone sensors, a fitness tracker and Steam profile counts showed nothing that survives correction at 46 to 149 people. None of the readers has yet been checked against real people's own questionnaires; `py/docs/study/` holds the protocol for that study.

The full account, including what failed, is in `docs/REPORT.md` and `docs/MATH.md` for the engine and `py/README.md` and `py/docs/MATH.md` for the Python side.

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

The listening histories behind the engine bench are not included, nor is any list of whose they are. The Spotify bench came from people who had published their own streaming history in public GitHub repositories, and the ListenBrainz bench from public ListenBrainz profiles. Even though the data was public, a list naming those people next to an analysis of their habits is not something to publish, so only aggregate results and pseudonymous summaries are kept (the music reader's norms in `py/src/aas/data/music_norms.json` are percentiles over 304 people, with no rows).

Three sets of model weights are left out, while their model cards (`card.json`) and the scripts that build them are kept:

- `py/models/text_bigfive/` is trained on the Essays dataset, which has no stated licence, so the weights are not redistributed.
- `py/models/text_mbti/` is trained on forum posts, and its vocabulary picked up a few forum usernames that people quoted in their posts. Rebuilding it locally is fine; shipping it is not.
- `py/models/text_distilled/` is trained on Claude's ratings of forum users. Anthropic's terms restrict using model outputs to build competing models, and while a five number regressor is unlikely to count, it has not been checked, so the weights stay out. It also failed its own pre-registered test and is not a default reader.

The NRC emotion lexicon used by one text experiment is free for non-commercial research only and is not included; `DATA.md` says where to get it.

## Licence

MIT, see `LICENSE`. A few small word lists under `py/src/aas/data/` keep their own licences, listed there and in `DATA.md`.

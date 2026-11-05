# Data

This repository ships no datasets and none of the listening histories its bench was run on. Every dataset below has to be downloaded by you, under its own terms, into `data/raw/<name>/` at the repository root unless the entry says otherwise. That folder is gitignored. The terms column says what the source states; where the source states nothing, that is said too, and you should read the source yourself before any use beyond research.

The detailed survey of text datasets, including the ones that were looked at and rejected, is in `py/docs/text_datasets.md`. Each slice's write up in `py/docs/` says exactly which files were used and how they were cleaned.

## Personality and behaviour datasets

### Essays (Pennebaker and King 1999)

Stream of consciousness essays by 2,467 students with yes or no Big Five labels from a self report questionnaire. It trains the default text Big Five reader.

- Get it: `curl -L -o data/raw/essays/essays.csv https://raw.githubusercontent.com/SenticNet/personality-detection/master/essays.csv`
- Terms: no data licence is stated. The mirror's MIT licence covers its code, not the essays. It has been used for research for decades; that is the only basis for using it here, and it is why the trained weights are not shipped.
- Used by: `scripts/train_text_readers.py --only bigfive`, `scripts/text_crossdomain.py`, `scripts/llm_crossdomain.py`, `scripts/distill.py` (evaluation).

### MBTI PersonalityCafe (Kaggle `datasnaek/mbti-type`)

The last 50 or so posts of 8,675 forum users with the MBTI type they reported. It trains the text MBTI reader and was the pool for the distillation experiment.

- Get it: `for s in train validation test; do curl -L -o data/raw/mbti/$s.parquet https://huggingface.co/datasets/jingjietan/kaggle-mbti/resolve/main/data/$s-00000-of-00001.parquet; done`
- Terms: the original Kaggle dataset is CC0 (public domain); the Hugging Face mirror is tagged Apache 2.0. It is scraped forum text, so it contains usernames people quoted, which is why the trained vocabulary is not shipped.
- Used by: `scripts/train_text_readers.py --only mbti`, `scripts/run_text.py`, `scripts/reid_features.py`, `scripts/distill.py`.

### FriendsPersona (Jiang, Zhang and Choi 2020)

711 scenes from the TV series Friends with observer ratings of the main speaker's Big Five.

- Get it: `https://raw.githubusercontent.com/emorynlp/personality-detection/master/CSV/friends-personality.csv` into `data/raw/friendspersona/`.
- Terms: the parent project (emorynlp/character-mining) is Apache 2.0; the sub repository has no licence file, and the dialogue comes from the show's scripts. Research use.
- Used by: the cross domain study only (`scripts/text_crossdomain.py`).

### PAN 2015 Author Profiling, English (Rangel et al. 2015)

294 Twitter users with self reported BFI-10 scores and about 100 tweets each.

- Get it: https://zenodo.org/records/3745945 (the training and test zips; keep the English parts) into `data/raw/pan15/`.
- Terms: "open access" on Zenodo, no licence stated, released for the PAN shared task. Research use; do not redistribute.
- Used by: the cross domain study, the LLM rater benchmark and the distillation test.

### YouTube vlogs (Biel and Gatica-Perez 2013, WCPR 2014 release)

Transcripts of 404 vloggers with observer impressions of their Big Five.

- Get it: https://huggingface.co/datasets/facells/youtube-vlog-personality-recognition-wcpr14 into `data/raw/youtube_vlogs/`.
- Terms: CC BY-NC-ND 4.0. Non-commercial research only, and no derivatives, so it is used for evaluation and never to train a saved model.

### PANDORA (Gjurkovic et al. 2021)

Reddit comments from about 1,600 authors with Big Five scores. Not used yet; the loader is ready in `py/src/aas/slices/text_pandora.py`.

- Get it: by request at https://psy.takelab.fer.hr/datasets/all/pandora/
- Terms: research use only, no transfer or reproduction of any part. Re-uploads found on Hugging Face break those terms and must not be used.

### IPIP-FFM answers (Open Psychometrics, 8 Nov 2018 dump)

About a million online answers to the 50 item IPIP Big Five markers, with response times. It sets the quiz norms (597,365 people after cleaning) and the answer timing experiment.

- Get it: https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip, then `cd py && uv run python scripts/build_ipip.py`.
- Terms: published by Open Psychometrics for research; check the terms on the site and in the zip before anything else. Only the aggregate norms are shipped (`py/src/aas/data/ipip_norms.json`). The item key is Goldberg's public domain IPIP (https://ipip.ori.org/newBigFive5broadKey.htm).

### StudentLife (Wang et al. 2014)

Smartphone sensing and the BFI-44 for 48 Dartmouth students over a ten week term.

- Get it: the RData mirror made for the R package `studentlife` (github.com/frycast/studentlife), Zenodo record 3529253, `dataset_rds.zip`, into `data/raw/studentlife/dataset_rds/`.
- Terms: as stated on the Zenodo record and the original StudentLife site. Not checked for this release beyond research use.
- Used by: `scripts/run_phone_body.py`, `scripts/run_pool_sensors.py`.

### LifeSnaps (Yfantidou et al. 2022)

Four months of Fitbit data and the IPIP Big Five for 71 people (50 with the questionnaire).

- Get it: Zenodo record 7229547 (v4), `rais_anonymized.zip`, into `data/raw/lifesnaps/rais_anonymized/`. Record 6832186 (v2) is now restricted.
- Terms: as stated on the Zenodo record. Not checked for this release beyond research use.
- Used by: `scripts/run_phone_body.py`, `scripts/run_pool_sensors.py`.

### Steam profiles and BFI-2 (Lin and Li, Mendeley Data v5)

173 Steam users with profile counts and the BFI-2.

- Get it: doi:10.17632/yyz69kysy3.5, file `Steam0706.csv`, into `data/raw/steam/`.
- Terms: CC BY 4.0, with credit to the authors and a note of what was changed.
- Used by: `scripts/run_play.py`.

## Listening histories (engine bench)

### Spotify Extended Streaming History, public GitHub copies

13 people with enough history for the detailed bench (skips, start reasons, device, country). These are files that their owners chose to commit to public GitHub repositories. The collection script and the list of sources are not published, so these people are not pointed at; the bench expects the files in `bench/data/spotify/`.

- Terms: each repository's own; most state none. The bench treats them as research data that stays on the machine. IP addresses and user agents are dropped on load, people appear only as P01 to P13 in reports, and no list of the repositories is published.

### ListenBrainz public profiles

343 listeners, 15.9 million listens, used for round 3 and for the music reader's norms. `bench/lbusers/README.md` explains how they were found, fetched and cleaned. Seed usernames go in `bench/data/lbusers/seeds.txt`, which you supply; users are stored under a salted hash and no username is committed.

- Terms: ListenBrainz listen data is published as CC0. Respect the API's rate limits.

### MusicBrainz and ListenBrainz lists

Artist lookups and similar artist lists are fetched live from the MusicBrainz and ListenBrainz APIs by the bench. MusicBrainz core data is CC0.

## Lexicons, word lists and models

- NRC Word-Emotion Association Lexicon v0.92 (Mohammad and Turney). Get it from https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip and put `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` in `data/raw/lexicons/`. Free for non-commercial research only; commercial use needs a licence from NRC. Not included.
- wordfreq (Speer 2022). `py/src/aas/data/common_words.json` is derived from it and is CC BY-SA 4.0.
- LDNOOBW English list. Included in `py/src/aas/data/word_blocklist.json`, CC BY 4.0.
- all-MiniLM-L6-v2 sentence embeddings, downloaded by fastembed on first use into `data/raw/models/`. Apache 2.0.

## LLM outputs

The LLM rater and the distillation experiment send text from the datasets above to Claude (or a local model) and cache the answers under `data/llm_cache/` and `data/raw/distill_cache/`. Those caches are not included. Rebuilding them costs money and needs your own key: set `ANTHROPIC_API_KEY` and `AAS_LLM_ALLOW_SPEND=1`, and read `py/docs/llm_rater.md` and `py/docs/distill.md` first. A model trained on those outputs falls under Anthropic's usage terms, which restrict using outputs to build competing models; check them before shipping one.

## Trained weights

None are shipped. Each folder under `py/models/` keeps its `card.json`, which records the data, the selection and every test score. To rebuild:

| reader | command | needs |
|---|---|---|
| text-bigfive | `cd py && uv run python scripts/train_text_readers.py --only bigfive` | Essays |
| text-mbti | `cd py && uv run python scripts/train_text_readers.py --only mbti` | MBTI PersonalityCafe |
| text-distilled | `cd py && uv run python scripts/distill.py --rate`, then `--evaluate` | MBTI, Essays, PAN 2015, vlogs, an Anthropic key |

A rebuilt text-mbti vocabulary will again contain a few forum usernames, so keep rebuilt weights to yourself. The music reader needs no weights; its norms are in `py/src/aas/data/`.

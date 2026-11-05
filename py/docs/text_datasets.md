# Text datasets with Big Five labels: what exists and what we can use

Surveyed 2026-11-01. "Usable" means the text and labels can be downloaded today without a request, and the
published terms (or, where there are none, the way the data was released) allow research use. Only English
sets can train or test an English model. Raw copies live under `data/raw/<name>/` (gitignored).

**Self-rated and observer-rated labels are different things.** A self-report score is what a person says about
themselves on a questionnaire. An observer score is how strangers *see* someone from a scene or a video. They
correlate only moderately (about r = 0.3 to 0.5 for strangers), so a model that predicts one is not
automatically predicting the other. Every table in `text_crossdomain.md` says which kind each domain is.

## The table

| name | size | labels | domain | licence / terms | usable? | why |
|---|---|---|---|---|---|---|
| **Essays** (Pennebaker & King 1999) | 2,467 people, ~650 words each | self-report questionnaire, **yes/no** median split | 1990s student stream-of-consciousness essays | none from the source; mirrors' code licences only | **yes** (already used) | the standard research copy; every public copy is yes/no only, no continuous-score copy exists publicly |
| **FriendsPersona** (Jiang, Zhang & Choi, AAAI 2020) | 711 scenes, ~80 characters by first name (the 6 main ones = 80%) | **observer**: 3 annotators rated the main speaker from the scene (-1/0/1), summed, **yes/no** median split by the authors | TV sitcom dialogue (Friends S1-4); we keep only the main speaker's own lines, median 30 words | parent project emorynlp/character-mining is Apache-2.0 (the sub-repo has no licence file); dialogue text is from the Friends scripts | **yes** (research) | open download; labels are of a fictional character's behaviour in one scene. NEU column is "emotionally stable" (see note) |
| **PAN 2015 Author Profiling**, English (Rangel et al. 2015) | 294 Twitter users (152 train + 142 test, both with truth files), ~100 tweets each | **self-report** BFI-10, **continuous** (-0.5 to 0.5 in 0.1 steps) | tweets, 2014-15 | Zenodo record is "open access", no licence stated; released for the PAN shared task | **yes** (research) | open download, full tweet text included (not IDs). No explicit licence: same status as Essays; we don't redistribute it |
| **YouTube vlogs** (Biel & Gatica-Perez 2013), WCPR 2014 release | 404 vloggers, one transcript each (median 466 words) | **observer**: MTurk raters' impressions after watching a minute of video, **continuous** 1-7 | spoken vlog monologue (transcribed) | Hugging Face re-host `facells/...` tagged **CC BY-NC-ND 4.0**; Idiap's original page (agreement-based) is gone | **evaluation only** | terms allow non-commercial research use; "no derivatives" means we don't ship a model trained on it. Used in the leave-one-domain-out study, excluded from the saved artifact |
| ChaLearn First Impressions v2 | 10,000 15-second clips, transcripts ~43 words each | observer (pairwise MTurk, continuous) | YouTube talking-head clips | behind a login; terms not visible; site warns against decision-making use | **no (not now)** | needs an account registration by the owner; could be added later |
| PANDORA (Gjurkovic et al. 2021) | ~1,600 Reddit users with Big Five | self-report, continuous | Reddit | access by request; "cannot transfer or reproduce any part" | **no (not now)** | request not made yet; loader already exists (`slices/text_pandora.py`) |
| PANDORA re-uploads (`jingjietan/pandora-big5`, `Fatima0923/Automated-Personality-Prediction`, `psetialana/...`) | | | Reddit | violate PANDORA's terms | **no** | not legitimate copies |
| myPersonality and derivatives (WCPR13 Facebook, `holistic-ai/Personality_mypersonality`, `facells/facebook-...`) | | self-report | Facebook | discontinued 2018 | **no** | excluded on ethics: Facebook data the owners withdrew |
| LLM-synthetic sets (Big5-Chat, `agentlans/big-five-personality-traits`, `Ritsuk0/aleph-big5-synthetic`, Big5PersonalityEssays, ...) | | written by a model told a personality | synthetic | various | **no** | not a human's writing or a human's personality |
| PersonalityEvd (EMNLP 2024), CPED | | annotated | **Chinese** TV dialogue | no licence / Apache-2.0 | **no** | not English |
| RealPersonaChat | | self-report | **Japanese** chat | CC BY-SA 4.0 | **no** | not English |
| PELD | 6 Friends characters | averages of FriendsPersona | | none | **no** | six labels in total |
| Pennebaker 2015-2023 essays (BFI-2, used in BIG5-TPoT 2025) | 5,810 | self-report, continuous | essays | not released | **no** | no public copy |
| `farahseddik/Instagram-Profiles-OCEAN-...` | | 0/1, origin unknown | scraped Instagram | none | **no** | unknown provenance, scraped profiles |
| MBTI sets (PersonalityCafe, Twitter MBTI) | | MBTI type | | | not Big Five | used by the separate text-mbti reader |

## URLs and checksums of what was downloaded (2.4 MB in total)

| name | URL | local file(s) | sha256 |
|---|---|---|---|
| FriendsPersona | https://raw.githubusercontent.com/emorynlp/personality-detection/master/CSV/friends-personality.csv ; licence https://github.com/emorynlp/character-mining/blob/master/LICENSE.txt | `data/raw/friendspersona/friends-personality.csv` | `89c9d078...cd3` |
| PAN 2015 | https://zenodo.org/records/3745945 (train and test zips; only the English parts kept) | `data/raw/pan15/*/...english.../` | (zips deleted after unpacking) |
| YouTube vlogs | https://huggingface.co/datasets/facells/youtube-vlog-personality-recognition-wcpr14 | `data/raw/youtube_vlogs/vlogs-wcpr14-{training,test}.csv` | |

## Decisions recorded with the data

- **FriendsPersona NEU direction.** The annotation form asked "Emotionally Stable" and the released column is
  `cNEU`. Using the labels alone (no model), Friends' `cNEU` correlates +0.17 with agreeableness and +0.15 with
  conscientiousness, the pattern of *stability* (in Essays, where cNEU is neuroticism, those are -0.19 and
  -0.15). So we flip it (N = 1 - cNEU). Decided before any model was scored on Friends.
- **Friends text.** Only the main speaker's own lines are kept; other speakers' lines and stage directions
  (written by the script writers) are removed. The annotators saw the whole scene, so the labels use context
  our model never sees.
- **PAN 2015 text.** Retweets and quote-tweets (someone else's words) are dropped. Neuroticism = minus the
  released "stable" score.
- **Vlogs.** Neuroticism = minus "stability". The raters watched the video, so the label also reflects face,
  voice and setting, which the transcript lacks.

# The derived feature record (`aas.features`)

Version: schema 1.0.0, extractor 0.1.0, word list 1.0.0, topic lexicon 1.0.0 (1 Nov 2026).
Design source: `docs/study/PRIVACY.md` section 2. Code: `src/aas/features.py`. Schema:
`src/aas/data/feature_schema.json`. Tests: `tests/test_features_privacy.py`.

## What it is

A participant's exports are parsed on their own computer into a `Person` (every post, search,
video title and play). `features.build(person, readers)` turns that into one JSON object, the
**record**, and the record is the only thing the study ever receives. `features.validate(record)`
rejects anything the schema does not allow, and `features.preview(record)` prints the plain-English
list the participant reads before pressing Send.

```
exports --parsers--> Person --readers--> Estimates --+
                        |                              +--> features.build --> record --validate--> sent
                        +------- counts, shapes -------+
```

The rule that makes it safe: **every value is a number, a true/false, an empty value, or a word
from a closed list.** A string is allowed only where the schema names the exact values it may take
(an enum) or gives a short anchored pattern (a `YYYY-MM` month, a version number, the 64-character
extractor hash). There is no field anywhere that can hold a sentence, so no post, query, title,
channel, artist, track, username, email address or URL can travel, whatever bug the code has: the
validator would refuse the record.

Size: about 380 numbers (9 KB of JSON) without the opt-in word block; about 2,380 numbers (39 KB)
with it.

## The record, field by field

Rounding legend: *share* = a fraction 0 to 1 rounded to 0.01; *bucket* = one of `0`, `1-49`,
`50-199`, `200-999`, `1000-4999`, `5000-19999`, `20000+`.

### Bookkeeping

| field | contents | why | why it can't carry content |
|---|---|---|---|
| `schema_version` | constant `"1.0.0"` | which layout this is | constant |
| `extractor.version`, `.hash` | version string; sha256 of `features.py` + the four data files | PROTOCOL integrity check: a record from unknown or edited code is excluded | fixed pattern |
| `extractor.word_list_version`, `.topic_lexicon_version` | versions | the lists can change only with a version bump | fixed pattern |
| `models[]` | `{reader, version}` for each reader that ran | which trained model produced the estimates | reader is an enum; version is a pattern (`1.0.0`, or `norms-<12 hex>` for the music reader, which has no model card) |
| `consent.word_rates` | true/false | the opt-in is recorded with the data | boolean |
| `sources.<s>` | true/false for `reddit`, `x`, `google_search`, `youtube_search`, `youtube_watch`, `spotify` | which exports were present | boolean |
| `floors.text/google/spotify` | `{passed, reason}`, reason in `ok`, `no_data`, `too_little`, `span_too_short` | PROTOCOL section 4 eligibility, checked locally | enum |
| `spotify.export` | `extended` or `basic` | the basic export lacks skips and shuffle, so some traits are missing for a known reason | enum |

### Reader outputs (`readers.<name>`)

One block per known reader: `text-bigfive` (traits O C E A N), `text-mbti` (I/E N/S T/F J/P),
`music` (the 7 measured traits and 2 dials).

| field | contents | why |
|---|---|---|
| `status` | `ok`, `insufficient_data`, `error`, `not_run` | a reader that could not read says so with a code, never with its message |
| `version` | model card version | refit and audit |
| `estimates.<trait>` | `usable`, `reason` (`insufficient_data` or null), `value`, `low`, `high`, `interval_level`, `percentile`, `validity`, `validity_kind` (`r`/`auc`), `reliability`; all rounded to 3 decimals (percentile 1) | exactly what the person sees in the app, minus the explanation; Q1, Q3 and Q6 are scored on these |
| `raw_scores.<trait>` (text readers) | the linear head's score before calibration, on all the person's prose as one document | PRIVACY block B: lets the study refit the calibration on ordinary people |

Dropped on purpose: `Estimate.evidence`. It is written for the person and quotes things
("most of all Radiohead, Burial and Four Tet", exact item counts). The music reader's evidence names
artists; none of it reaches the record. The reason text from `can_read` and any exception message
are also dropped, because messages can quote data.

A reader not in the schema makes `build()` raise: adding a reader means changing the schema and
bumping the version, deliberately.

### Activity (`activity.<source>`, six sources)

| field | contents | why | why it can't carry content |
|---|---|---|---|
| `items` | bucket | volume (dose-response, Q8) | bucket |
| `first_month`, `last_month` | `YYYY-MM` | date range, coarsened | months only, no day or time |
| `span_months` | integer | how long the history is | integer |
| `active_day_share` | share of days in the span with any activity | regularity | one number |
| `hours_utc[24]` | share of items in each UTC hour | rhythm (chronotype is a known correlate of conscientiousness) | a histogram has no order and no content |
| `weekdays[7]` | share of items per weekday, Monday first | weekly rhythm | same |

With fewer than 20 dated items, the histograms and `active_day_share` are null: with one or two
posts, an exact hour plus a month could point at a single public post.

### Writing (`writing`, Reddit + X own posts, searches excluded)

| field | contents | why |
|---|---|---|
| `words_rounded` | total words, to the nearest 100 | floor checks, dose-response |
| `items_by_kind` | bucket for `comment`, `post`, `tweet`, `reply`, `other` | genre mix |
| `style.*` | 15 numbers: mean words per item; shares of items with a question mark, an exclamation mark, a URL, an @mention; per 1,000 words: first person singular, first person plural, second person, negations, profanity (LDNOOBW words), emoji; share of ALL-CAPS words, share of 7+ letter words, mean word length; type-token ratio on 1,000-word windows (null under 1,000 words) | the classic LIWC-style correlates of personality, cheap and interpretable |

URLs, emails and handles are removed before anything is counted; only "an item had a URL" survives.
Under 5 posts the style numbers are null (one post's exact length is a fingerprint).

### Searches (`search_form.google_search`, `search_form.youtube_search`)

`mean_words`, `question_share` (starts with how/what/why/... or ends in ?), `near_me_share`,
`english_share` (at least half the words in wordfreq's top 5,000 English words), `repeat_share`
(share of queries that repeat an earlier one). Null under 20 queries. The queries are read, measured
and discarded.

### Viewing (`viewing_form`, YouTube watches)

`distinct_channel_ratio`, `channel_entropy_bits`, `top10_channel_share`, `repeat_watch_share`,
`music_video_share` (title looks like "Official Video", "lyrics", "ft.", "live at", "remix"...),
`no_channel_share`. Null under 20 watches. **No channel names**: channel identity is used only to
count distinct channels, then dropped.

### Topics (`topics.google_search`, `.youtube_search`, `.youtube_watch`)

Share of items in each of 20 broad categories plus `other`; null under 20 items. See the next
section.

### Spotify (`spotify`)

`export` (above) and `shuffle_share` (null for the basic export). Play count and span are in
`activity.spotify`; the measured traits are in `readers.music`. **No artist or track names**.

### Opt-in common-word rates (`word_rates`, only with consent)

Present only when `build(..., word_rates_opt_in=True)` and the person wrote at least 500 words.
`word_rates_status` says which: `absent_no_consent`, `below_500_words`, `present`.

| field | contents |
|---|---|
| `tokens_rounded` | number of counted words, to the nearest 100 |
| `per_1k.<word>` | for **every** word of the fixed 2,000-word list (zeros included, so the key set never varies), uses per 1,000 words, 3 significant figures |

URLs, emails, handles and digits are removed first; tokens are letters only, 2+ letters (the
readers' tokeniser). The schema lists the 2,000 words as the only allowed keys and requires all of
them, so a word not on the list cannot be added and none can be dropped to make room.

## The word list (`data/common_words.json`)

Built by `scripts/build_feature_words.py` from wordfreq 3.1.1 (`top_n_list('en', 5000, 'best')`,
data CC BY-SA 4.0, Speer 2022), in frequency order:

1. alphabetic, at least 2 letters;
2. in a trained text reader's unigram vocabulary (so the block can retrain the current models);
3. not a first name (`/usr/share/dict/propernames`, except ordinary words it happens to list such
   as "will", "mark", "the", "think");
4. not in `data/word_blocklist.json`: hand groups for health, sexuality and gender identity,
   religion, politics and unions, ethnicity and nationality, crime, money trouble, places, names and
   brands, foreign fragments; plus the whole LDNOOBW English list (CC BY 4.0) for profanity, sexual
   terms and slurs;
5. the first 2,000 survivors (the last one is rank 2,275 in wordfreq).

Conservative choices, documented so they can be argued with: "black" and "white" are out (one
sense is ethnicity), "husband", "wife", "girlfriend", "boyfriend" are out (with a stated gender they
reveal orientation), "pain", "weight", "diet", "sick" are out (health), "president" and "union" are
out (politics). Every surviving word was read before freezing.

PRIVACY.md predicted "about 2,000" words from the top-5,000 rule alone; in fact that rule leaves
about 4,000, so the list is cut at 2,000 by frequency. The effect: the block covers the most common
words only, which is also the safer end.

## Topic categories and how items are assigned (`data/topic_lexicon.json`)

| category | examples of keywords |
|---|---|
| sport | football, nba, premier league, highlights, gym, workout |
| gaming | minecraft, fortnite, playstation, walkthrough, speedrun |
| food_drink | recipe, baking, pizza, coffee, restaurant, air fryer |
| technology | iphone, laptop, python, wifi, chatgpt, unboxing |
| science_space | physics, nasa, black hole, dinosaur, volcano |
| education | how to, tutorial, exam, homework, lecture, definition |
| music | lyrics, official video, remix, concert, guitar, playlist |
| film_tv | trailer, netflix, episode, anime, ending explained |
| books_reading | novel, kindle, audiobook, poetry, booktok |
| travel_outdoors | flights, hotel, airbnb, hiking, camping, passport |
| vehicles | tesla, motorcycle, oil change, used car, mpg |
| fashion_beauty | outfit, sneakers, makeup, skincare, haircut, perfume |
| home_garden_diy | diy, ikea, renovation, plumbing, gardening, cleaning |
| pets_animals | dog, puppy, kitten, aquarium, wildlife, zoo |
| parenting_kids | toddler, nursery rhymes, stroller, lego, cocomelon |
| work_career | resume, salary, linkedin, interview, freelance |
| money_investing | stocks, etf, bitcoin, budget, discount, price |
| art_design_photo | drawing, procreate, photoshop, photography, knitting |
| comedy_memes | funny, stand up, memes, prank, try not to laugh |
| utilities | weather, translate, near me, opening hours, celsius |
| other | everything else, and everything sensitive |

**Assignment.** The item (a query or a video title) is lower-cased and split into words; every
lexicon keyword of one to five words that occurs adds one vote to its category (a keyword counted
once per item); the item goes to the category with the most votes, ties split evenly; no votes means
`other`. Per source the record holds each category's share of items, rounded to 0.01.

**Sensitive routing.** If the item contains any term on the lexicon's `sensitive` list (about 230
health, sexuality, religion, politics, ethnicity, crime and money-trouble terms) or any LDNOOBW
term, it goes to `other`, whatever else it matches: "christian rock music" is `other`, not `music`;
"depression symptoms" is `other`. How many items were routed this way is **not** recorded, because
that count would itself be a sensitive signal.

There is deliberately no health, religion, sexuality, politics, ethnicity, crime, money-trouble or
news category (PRIVACY.md suggested "news in general"; it was left out because news is where
political content lives). "Finance" became `money_investing` (markets, prices, deals); debt, loans,
benefits and similar go to `other`.

Why a lexicon and not MiniLM centroids: it is deterministic, readable by a reviewer, needs no model
download in the browser, and a person can check by hand why an item landed where it did. The cost is
coverage: phrasing outside the keywords lands in `other`. Expect `other` to be the largest share for
Google searches; the pilot should report its size and, if needed, grow the lexicon (a version bump).

## The privacy tests (`tests/test_features_privacy.py`)

A synthetic person is built with canaries everywhere: made-up tokens (`zqxjvorbel`), an email, two
URLs, two phone numbers, a full name, a Reddit `u/` handle, an `@handle`, a whole sentence, canary
artists and track names in 4,000 Spotify plays, canary video titles and channels in 300 watches,
canary searches, a canary in an unknown source, canaries in the parser notes, a reader whose
evidence quotes canaries, and a reader whose exception message quotes a canary. The real text and
music readers run on it, and every block of the record is filled (the first test checks that, so the
rest are not vacuous). Then:

| test | asserts |
|---|---|
| no canary anywhere | none of 42 canary strings (and fragments: digits of the phone numbers, the URL id) occurs in `json.dumps(record)` or in any key or value, with and without the word block |
| no long input strings | every string in the record is at most 40 characters (the one exception is the extractor hash at its own path), and no 41-character window of any input string occurs in the record |
| every string is a known label | each string is a schema key, enum value, word-list word, a month, a version or the hash |
| schema cannot hold free text | walks the schema: every string slot is an enum/const or an anchored pattern of at most 64 characters; every object is closed (`additionalProperties: false`), except `per_1k`, whose keys are an enum of the 2,000 words and whose values are numbers |
| dates coarsened | no `YYYY-MM-DD`, no `HH:MM`, no `T12`, no number above 10 million anywhere (no timestamps, no exact large counts) |
| validates / validate rejects | the record validates; 16 mutations are refused: an extra top-level or nested field, free text in an enum, a timestamp in a month field, text in a number, NaN, a share above 1, a 25-hour histogram, `evidence` on an estimate, an unknown reader or trait, an off-list word, a missing list word, a wrong schema version, a free-text model version, an unknown topic |
| word rates absent unless opted in | absent without consent; absent under 500 words; when present, exactly the 2,000 list words, 3 significant figures |
| word list is clean | 2,000 unique a-z words, none in the blocklist or LDNOOBW or the first-name list |
| topics | no sensitive category exists; sensitive items go to `other`; no count of them is kept |
| also | deterministic output; unknown readers and traits refused; unusable estimates carry no numbers; tiny sources give no histograms; the parsed fixture exports validate and no input string of 4+ characters appears; the preview text has no canaries; `aas.features` imports no numpy/scipy/sklearn/pandas/joblib; the schema file matches its generator |

Result on 1 Nov 2026: **all 56 privacy tests pass** (one optional cross-check against the
`jsonschema` package is skipped because it is not installed); the full `py/tests` suite also passes.
The tests were also checked against two deliberately leaky versions of the builder (evidence
copied into `reason`, a channel name added to `viewing_form`): 10 tests failed, as they should.

## Re-identification: how identifying is a record? (honest numbers)

`scripts/reid_features.py`, results in `docs/features_reid.json`. Population: 5,948 real
PersonalityCafe forum users (the MBTI training corpus; those with at least 500 words in each half
of their posts; median 724 words per half). Style and word rates are computed with the real code on
their real posts. **Their posts have no timestamps, so the hour/weekday histograms are simulated**
(US/UK time zones, evening-peaked habits, varying regularity, 150 items per half).

**1. Singling out.** Treat each field, rounded exactly as the record rounds it, as a label. In a
population of N, how many fields does it take to make a person's combination unique?

| block | N = 300 | N = 1,000 | N = 5,948 |
|---|---|---|---|
| writing style (15 numbers) | 100% unique; median 2 fields needed | 100%; 2 | 100%; 4 |
| hours + weekdays (31, simulated) | 100%; 3 | 100%; 4 | 100%; 5 |
| coarse rhythm (7 numbers: six 4-hour blocks + weekend share, to 0.05) | 95%; 5 | 83%; 5 | 51%; 6 |

Plainly: **every record is unique**, and two to five of its ~380 numbers already single a person
out. That is expected of any rich numeric record and is why PRIVACY.md calls it pseudonymised
personal data, not anonymous data. Uniqueness alone only matters if the file leaks together with a
second dataset to match against, which is the next measurement.

**2. Linkage (the realistic attack).** Each person's posts are split in two. Half A plays the record
the study holds; half B plays an attacker's scrape of the same person's public posts, from which the
attacker computes the same features. For each record, rank all N scrapes by similarity:

| features the attacker matches on | top-1, N = 300 | N = 1,000 | N = 3,000 | top-10, N = 3,000 |
|---|---|---|---|---|
| chance | 0.3% | 0.1% | 0.03% | 0.3% |
| writing style only | 16% | 7% | 5% | 16% |
| hours + weekdays only (simulated) | 15% | 8% | 4% | 18% |
| coarse rhythm only (simulated) | 9% | 3% | 2% | 10% |
| style + hours | 40% | 27% | 17% | 47% |
| **opt-in word rates only** | 26% | 17% | 13% | 29% |
| everything (style + hours + word rates) | 59% | 46% | 36% | 64% |

What this means:

- **The record is linkable.** Someone holding the study file plus a scrape of a few thousand
  candidate Reddit/X accounts with timestamps would find the right account first for about a sixth
  of participants without the word block, and for about a third with it. That is 500 to 1,000 times
  better than chance.
- **The opt-in word block roughly doubles linkage**, which supports keeping it opt-in, separately
  consented, encrypted, and never published (PRIVACY.md 2.1 and 5 already say this).
- **These numbers are probably underestimates for real participants.** Only about 700 words per
  half here; real Reddit/X histories are often 10x longer, and stylometric matching sharpens with
  more text. Real posting rhythms are also real, not simulated. Against that, a real attacker
  searches millions of accounts, not 3,000, which lowers top-1 sharply unless they can pre-filter
  (for example by country or activity period, which `first_month`/`last_month` help with).
- Reader outputs (Big Five, MBTI, music) were not included in the attack and would add more.
- The simple "everything" combination is not optimal; a trained matcher would do better.

**Options for the owner, if this is too much** (none is applied by default; each costs analysis):

1. Replace the 24-hour and 7-day histograms with the coarse rhythm block (6 blocks + weekend share,
   0.05 steps): halves rhythm linkage, keeps "morning vs night person".
2. Drop `first_month`/`last_month` and keep only `span_months` (PRIVACY.md's original wording):
   removes the activity-period pre-filter.
3. Round style numbers more coarsely (e.g. 2 significant figures).
4. Keep the word block opt-in and out of every shared or published file (already the plan).

## Running in the browser (Pyodide)

`aas.features` imports only the Python standard library and `aas.person` (a test enforces it). The
data files are plain JSON read with `pathlib`, which works on Pyodide's in-memory file system when
the package is installed as a wheel. The validator is a small built-in subset of JSON Schema
(type, enum, const, pattern, minimum/maximum, items, min/maxItems, required, properties,
additionalProperties, propertyNames, `$ref` into `$defs`), so no `jsonschema` package is needed.

What still needs work for the browser build:

1. **Readers.** `readers.text` uses numpy, scipy (sparse matrices, `norm.ppf`), scikit-learn
   (`CountVectorizer`) and joblib to load `model.joblib`; `readers.music` uses numpy and scipy. All
   exist in Pyodide but scikit-learn and scipy are large downloads (tens of MB). Cheaper: export each
   model to plain JSON/`.npy` (vocabulary, idf, head coefficients, calibration) and score with a
   20-line numpy tokeniser plus `math.erf` instead of `scipy.stats.norm`. joblib pickles are also
   tied to library versions; a JSON export avoids that.
2. **The import chain.** `readers.text` imports `slices.text` (pandas) and `slices.text_pandora` for
   the leak-stripping regexes. Move `strip_leaks` and `BIG5_LEAK` into a small dependency-free
   module so the browser does not load pandas.
3. **Raw scores.** `features._raw_scores` reaches into a text reader's `.m` and `.vec`. A public
   `reader.raw_scores(person)` method would be cleaner (proposed below).
4. **Hash.** `extractor_hash()` reads `features.py` itself; in a wheel that works. If the browser
   build bundles differently, compute the hash at build time instead and ship it as a constant.
5. **Zips.** `parsers._bundle` uses `zipfile` on a path; in the browser the dropped file arrives
   as bytes, so `Bundle` needs a bytes constructor (`zipfile.ZipFile(io.BytesIO(data))`).

## Proposed changes to other modules (not made here)

- `readers/text.py`: add `raw_scores(person) -> dict[str, float]` (the head score before
  calibration). If the study wants it to match what the reader calibrates for long histories, use
  the same chunk average the reader uses; today the record uses one whole-history document.
- `readers/music.py`: give `MusicReader` a `version` attribute (today the record uses a hash of the
  norms files).
- `parsers/*`: `Person.notes` are free text and include file names; they are never sent, but a
  fixed set of note codes would let the record report parser warnings safely.
- `parsers/reddit.py`: PRIVACY.md 2.2 point 5 says typology subreddits (r/intj, ...) are dropped
  locally and counted; the parser does not keep the subreddit, so this cannot be done yet. It should
  read the subreddit, drop those items, and the record could carry a bucketed `dropped_typology`
  count.
- `person.Estimate`: an explicit `shareable=False` marker on `evidence` would document what the
  feature builder already enforces.

## Regenerating the data files

```
cd py
uv run --with wordfreq python scripts/build_feature_words.py --ldnoobw <LDNOOBW en file>   # word list + blocklist
uv run python scripts/build_feature_schema.py                                             # schema (after any change)
uv run pytest tests/test_features_privacy.py                                              # must pass
uv run python scripts/reid_features.py                                                    # re-identification numbers
```

Any change to the word list, lexicon, schema or `features.py` changes `extractor.hash`; bump the
relevant version and freeze again after the pilot.

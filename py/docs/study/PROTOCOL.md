# Study protocol: one person, several sources, one questionnaire

Version 0.1, 1 Nov 2026. Draft for the owner's decisions (see README.md). Nothing here has been
run, submitted or paid for.

## 1. Why this study exists

AAS reads one slice of a person at a time: what they write (Reddit, X), what they search and
watch (Google, YouTube), what they listen to (Spotify). Each reader was trained or normed on a
different public dataset, with different people in it:

| source | reader today | trained / normed on | best honest number so far |
|---|---|---|---|
| Reddit / X writing | `text-bigfive`, `text-mbti` | 1990s student essays; a typology forum | Big Five AUC 0.57 to 0.68 in domain (implied r 0.16 to 0.40, only O real); cross-domain transfer AUC 0.55 to 0.67 |
| Google searches, YouTube | none (searches are deliberately excluded from the text reader) | | no labelled data anywhere |
| Spotify | `music` (listening traits, not personality) | 304 ListenBrainz listeners, no personality labels | split-half r 0.72 to 0.99 for the traits themselves |

No public dataset has the same person's writing, searches, viewing and listening next to their own
personality questionnaire. So nobody, including us, can say whether combining sources helps, or
whether a combined guess's stated uncertainty is honest. This study collects exactly that, and
nothing more.

## 2. Aims and the specific questions

Primary (confirmatory, pre-registered in PREREG.md):

- **Q1. Does the text reader work on ordinary people?** Correlation between the pre-trained
  `text-bigfive` score and the person's own quiz z, per trait, with no refitting. This is the
  first clean out-of-domain number for the reader that exists today.
- **Q2. Does a second source add anything?** Does a combined model (text reader + a model of the
  second source, fit inside cross-validation on study people) predict the quiz better than the text
  reader alone, on people it never saw? Reported as Delta r and as a paired error test.
- **Q3. Are the stated intervals honest?** Do 80% intervals contain the quiz z about 80% of the time,
  both for the existing text reader's own intervals and for the combined model's?
- **Q4. Does the combined model beat knowing nothing?** Skill = 1 - MAE / MAE(guess average).

Secondary:

- Q5. Validity of each non-text source on its own (Google search/YouTube features, Spotify listening
  traits), learned in study with cross-validation.
- Q6. MBTI reader letters vs an open type questionnaire (OEJTS), AUC per letter.
- Q7. Retest reliability of the quiz at 2 weeks (the ruler's own noise, which caps every r).
- Q8. Dose-response: does validity rise with words of text (testing the 100 / 800-word floors)?
- Q9. Domain shift: Reddit-only vs X-only writers.

Not asked: anything about health, sexuality, religion, politics, ethnicity or any other sensitive
trait. The extractor does not compute features aimed at them (PRIVACY.md section 3).

## 3. Design in one picture

```
Prolific screener (3 min)  ->  Session 1 (25 min)            ->  wait 2 to 30 days      ->  Session 2 (25 min)               -> [Session 3, 10 min, optional]
 which accounts, how much       consent, IPIP-50, OEJTS,           exports arrive by email     open extractor page in browser,      late Spotify extended
 they use them, willing?        demographics, request exports      (reminders 3/7/14/21/28)    drop the zips, review the payload,   history top-up
                                (guided, step by step)                                         send derived features only
                              + random subset: IPIP-50 retest at day 14 (5 min, separate Prolific study)
```

Two sessions are forced by the platforms: a data export cannot be produced inside a survey. Reddit
and Spotify extended history officially take up to 30 days; X says 24 hours or more; Google Takeout
minutes to a day or two (`docs/exports.md`). Session 2 opens as soon as the person has their text
export plus one other, so most people need not wait for the slowest one. Spotify extended history
that arrives after Session 2 is collected in an optional Session 3.

## 4. Participants

Inclusion (all must hold):

1. Adult, 18 or over (Prolific verifies age at sign-up; we re-ask).
2. Lives in the US or UK (Prolific location filter).
3. Fluent in English, writes online mostly in English (the readers are English-only).
4. Has **writing**: a Reddit or X account with own posts or comments, and in the export at least
   **100 words** of own prose after cleaning (the `text-mbti` floor; the Big Five floor is 25 words).
   We aim for at least 800 words (the MBTI "recommended" length) and record the word count so
   analyses can be restricted to it.
5. Has **at least one more source** among:
   - Google: My Activity search history and/or YouTube watch history, with at least **300 searches or
     200 YouTube watches over at least 90 days** (provisional: no reader exists, so these floors are
     our choice, to be checked in the pilot);
   - Spotify: at least **2,000 plays over at least 90 days** (the `music` reader floor). Extended
     history preferred; the basic one-year export is accepted and flagged.
6. Willing to request their exports and run the extractor page on a laptop or desktop browser
   (Chrome, Edge, Firefox or Safari, recent version).

Why writing is required (a deviation from "any 2 of the supported exports"): Q2 asks whether a second
source improves on the best single source. Writing is the only source with a pre-trained Big Five
reader, and every prior number says it is the strongest. If people could join with, say, Google +
Spotify only, the "best single source" would differ from person to person and the increment test
would be split across many small source pairs. Requiring writing gives every participant the same
anchor. The cost: fewer eligible people (Reddit/X writers are a minority) and a sample tilted to
people who post online. The owner can relax this (README decision 4).

Exclusion:

- Fails 2 or more of the 3 attention checks in Session 1 (section 6).
- Quiz speed or straight-lining rules from `quiz.py` cleaning: median time per item under 1 s, all
  50 answers identical, or 15 or more identical answers in a row.
- Extractor reports the exports fail the floors above (person is paid, not analysed).
- Payload fails integrity checks (extractor version unknown, hash mismatch, impossible values, same
  payload submitted twice under two IDs).
- Says at the end that their answers should not be used (a no-penalty "did you answer honestly?" item).

## 5. Measures

### 5.1 Personality instrument: the choice

| option | items, time | licence | for AAS |
|---|---|---|---|
| **IPIP-50** (Goldberg's Big-Five Factor Markers) | 50, about 7 to 8 min | Public domain: "copy, edit, translate, or use them for any purpose without asking permission and without paying a fee" (ipip.ori.org) | Already keyed, normed on 597,365 people, scored by `quiz.score()`; alpha 0.80 to 0.90; same items as the in-app 20-item quiz, so the study ruler and the product ruler are one ruler |
| IPIP-NEO-120 (Johnson 2014) | 120, about 15 to 20 min | Public domain (IPIP) | Adds 30 facets (4 items each, so each facet is noisy). No norms in the repo. Doubles quiz time and fatigue; more dropout before the hard part of the study |
| BFI-2 (Soto and John 2017) | 60, about 10 min | **Not public domain.** "Freely available for researchers to use for non-commercial research purposes"; "for non-commercial uses only"; commercial use needs a request to the Berkeley lab (ocf.berkeley.edu/~johnlab/bfi.html) | Excellent psychometrics, but AAS is meant to become a product, and the models trained on this study would carry that purpose. Using BFI-2 would need written permission first |

**Decision: IPIP-50, all 50 items**, in the standard order, 5-point scale (Very inaccurate to Very
accurate, matching the norms data's 1 to 5). Reasons: no licence risk for a future product; norms and
scoring already exist and are tested; it is the full form of the in-app quiz; short enough to leave
energy for the export steps. Facets are out of scope: the readers predict domains, and four-item
facets at n of a few hundred would be noise.

Secondary instrument: **OEJTS 1.2** (Open Extended Jungian Type Scales, 32 items, about 4 min) so the
MBTI reader can be scored against a type questionnaire instead of only the loose Big Five mapping.
Licence: CC BY-NC-SA 4.0 (openpsychometrics.org). That is **non-commercial**: fine for this research
study, but the MBTI reader must not ship OEJTS items in a product, and if the owner considers the
study itself commercial, drop OEJTS and keep the loose mapping (README decision 6). It is optional.

### 5.2 Retest (the ruler's own reliability)

A random 25% of Session 1 completers (minimum 30, cap 100) are invited 14 days later (window day 12
to 20) to retake the IPIP-50 only, as a separate short Prolific study. Random assignment happens at
Session 1, before anyone knows who will finish Session 2, so the retest sample is not filtered by
data-donation success. Output: retest r and ICC per trait. This gives the ceiling every validity is
read against: a model cannot correlate with the quiz more than about sqrt(retest r).

### 5.3 Attention and quality checks

Prolific's rule for studies of 5 minutes or more: a submission may be rejected only for failing
**more than one** attention check, and comprehension checks may never be grounds for rejection.
So:

- Two instructed-response items inside the IPIP-50 ("For this line, please choose Moderately
  accurate"), written so no good-faith reading gives another answer.
- One instructed item in the OEJTS block.
- Reject (no pay) only when 2 of 3 are failed. One failure: pay, keep, flag.
- Separate from rejection, analysis exclusions use the same speed and straight-lining rules that
  cleaned the 597k norms (section 4), so the study ruler is cleaned exactly like its norms.
- A comprehension check on the privacy page ("Which of these leaves your computer?") is used to
  re-explain, never to reject.

### 5.4 Other self-report

Age band, gender (optional, free text allowed), country, education band, hours online per day,
"How much of what you write online is in English?". Nothing else. No names, handles or emails
(Prolific forbids asking for social media accounts, and we do not need them).

### 5.5 What the extractor computes

Defined in PRIVACY.md section 2, and frozen (versioned and hashed) after the pilot. In short:
reader outputs, counts and time patterns, broad topic shares, and an optional common-word rate block.
No text, no search queries, no titles, no artist or channel names leave the computer.

## 6. The exact data flow

1. **Screener (Prolific, 3 min, paid).** Uses Prolific's free prescreeners (US/UK, age, fluent English)
   plus our own questions: which of Reddit, X, Google, YouTube, Spotify they use and how much; laptop
   access; willingness to request exports. Eligible and willing people get a Session 1 invite by
   Prolific ID allowlist.
2. **Session 1 (our study page, opened from Prolific, 25 min).**
   - Information sheet and consent (PRIVACY.md section 6). Consent is per step: quiz, export requests,
     optional common-word block, optional retest.
   - The page creates a random **study ID** (128-bit) and a **withdrawal code** shown to the person
     (we store only a hash of the code next to their data). The Prolific ID is stored in a separate
     **link table** used only for payment, reminders and withdrawal, never joined into analysis files.
   - IPIP-50 (+2 instructed items), OEJTS (+1), demographics.
   - Guided export requests, one service at a time, with screenshots: Reddit (GDPR, all time), X
     archive, Google Takeout (deselect all, then only My Activity: Search in JSON, and YouTube
     history in JSON), Spotify (tick Extended streaming history). They confirm each request; we
     store which services were requested and when.
3. **Waiting (2 to 30 days).** Prolific messages on day 3, 7, 14, 21 and 28: "Has your export
   arrived? Session 2 is open." Session 2 is published as a follow-up study to the allowlist, open
   until day 35.
4. **Retest subset (day 14).** Separate 5-minute Prolific study, IPIP-50 only, linked by study ID.
5. **Session 2 (extractor page, 25 min).**
   - The page loads the AAS extractor in the browser (Python compiled to WebAssembly via Pyodide,
     the approach of the Port data-donation tool; PRIVACY.md section 2.1). The participant drops
     their zips in. Files are read in the browser's memory; they are never uploaded.
   - The extractor runs `aas.parsers.load()`, the readers, and the feature builder; checks the floors.
   - It shows the participant **exactly what will be sent**: a plain-English summary ("We will send:
     your writing came to 3,420 words; you posted most in the evening; 31% of your YouTube viewing was
     music..."), plus the raw JSON for anyone who wants it. They can untick any source.
   - On "Send", one HTTPS POST to the study server: `{study_id, extractor_version, extractor_hash,
     payload}`. The page's Content-Security-Policy allows network requests only to that endpoint.
   - A completion code returns them to Prolific.
6. **Session 3 (optional, 10 min).** For Spotify extended history that arrived late: same page,
   Spotify only, appended to the same study ID.
7. **Server.** Stores quiz answers and payloads by study ID (PRIVACY.md section 5). The analysis
   export is a single parquet file of study-ID rows with no Prolific IDs.
8. **Payment and close.** Payments within 7 days of each session. The link table is deleted 60 days
   after the last payment (PRIVACY.md section 5).

## 7. Analysis outline

Full plan in PREREG.md. In brief: the text reader is scored as is (no refit) against quiz z; every
model learned from study data is scored on people it never saw (10-fold CV, nested where stacked);
coverage is checked against the interval level each estimate claims; skill against guessing the
average; Holm across the five traits within each family; permutation nulls, read against `null_mean`
as in `evaluate.py`.

## 8. Pilot first

Ten people before anything else (RECRUITMENT_AND_BUDGET.md section 6). The pilot fixes the feature
specification, the export floors for Google, the timings and the pay, and checks that the extractor
page runs on real exports on ordinary laptops. Pilot data are never used in confirmatory analyses.

## 9. Things that must be built before the pilot

Listed so the gap between this document and reality is clear:

1. `aas.features`: the derived feature builder (PRIVACY.md section 2), with tests that assert no
   field can carry free text.
2. A Pyodide build of the extractor (parsers + readers + features). The readers import numpy, scipy and
   (through `slices.text`) pandas, all of which ship with Pyodide; the TF-IDF heads are plain numpy.
   The embedding path (fastembed / ONNX) is not needed and is left out. The import chain should be
   trimmed so the browser does not load training-only modules.
3. The study pages (Session 1 survey, Session 2 extractor page, retest) and a small server endpoint.
   This is the minimum instrument for collecting data, not a product UI.
4. A topic lexicon for searches and video titles (about 20 broad, non-sensitive categories) that runs
   locally.

## 10. Biggest risks in this design

1. **Dropout between sessions.** Data-donation studies report high and non-random dropout (Welbers et
   al. 2024, Computational Communication Research; Keusch et al. 2024, Field Methods). Our cost model
   assumes 50% of Session 1 completers finish Session 2, with 35% as the pessimistic case. If it is
   20%, every cost doubles. The pilot measures it.
2. **The text reader may be near chance on ordinary people.** In domain, only Openness was real
   (AUC 0.68). If text r is about 0.1, Q2's "increment over the best single source" is an increment
   over almost nothing, and the study mainly says that.
3. **The second sources have no reader.** Google and Spotify models must be learned from the study's
   own people. At n = 250 a 10-feature model with a true R of 0.15 scores about 0.06 under
   cross-validation (POWER.md). Small effects are not detectable at the main scale.
4. **Eligibility.** Requiring Reddit/X writing plus a second source plus a laptop shrinks the pool and
   tilts it to heavy online users. Generalisation is to "people who write online", not everyone.
5. **Self-report on self-report.** The quiz is the ground truth but has its own noise (alpha 0.80 to
   0.90); the retest measures how much.

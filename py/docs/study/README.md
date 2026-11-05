# The AAS whole-person study: summary

Version 0.1, 1 Nov 2026. Design only: nothing has been submitted, recruited or paid for.

## What and why

No public dataset has one person's writing, searches, viewing and listening next to their own personality
questionnaire, so nobody can say whether a "whole person" model beats a single source, or whether its
uncertainty is honest. This study collects that, from US and UK adults on Prolific, in two sessions:

1. **Session 1 (25 min):** consent, the IPIP-50 Big Five questionnaire (public domain, already normed in
   `quiz.py`), an optional 32-item type questionnaire (OEJTS), and guided requests for their Reddit/X, Google
   and Spotify exports.
2. **Wait** 2 to 30 days for the exports.
3. **Session 2 (25 min):** a web page runs our extractor **inside their browser**. Their files are never
   uploaded. They see exactly which numbers will be sent (reader outputs, counts, time-of-day patterns, broad
   topic shares, music listening traits; optionally rates of about 2,000 common words), and press Send.

A random subset retakes the questionnaire at 2 weeks to measure the ruler's own noise. A 10-person pilot comes
first.

## Files

| file | what it settles |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | questions, eligibility, instrument choice, data flow, what must be built first |
| [PRIVACY.md](PRIVACY.md) | the derived payload field by field, what it can and cannot reveal, law, storage, consent text, ethics routes |
| [POWER.md](POWER.md) | sample size from simulation (`scripts/study_power.py`, results in `power_results.json`) |
| [RECRUITMENT_AND_BUDGET.md](RECRUITMENT_AND_BUDGET.md) | Prolific fees and pay, funnel, cost per scenario, alternatives, pilot plan |
| [PREREG.md](PREREG.md) | hypotheses, analysis, exclusions, stopping rule, fixed before data |

## Key numbers

The powers below use the validities measured on 2026-11-01 on writing the readers never saw (POWER.md section 9):
text Big Five r 0.097, 0.095, 0.094, 0.158 for C, E, A, N on Reddit and X, openness refused. The main study waits
until a text model retrained on PANDORA reaches r >= 0.20 on held-out Reddit authors for at least three traits.

| scenario | usable n | central cost (pay + Prolific fee) | answers at measured validities |
|---|---|---|---|
| Pilot | 10 | about $600 | does the instrument work; real return rate and timings |
| Minimal | 60 | about $3,600 | text validity only roughly (+-0.24); pooled coverage roughly; nothing about combining |
| **Main** | **250** | **about $14,700** | text validity +-0.12; per-trait coverage +-5 points; text r > 0 with power 0.32 to 0.33 for C, E, A and 0.71 for N; fused r > 0 with power 0.51 to 0.88; an increment of 0.10 with power 0.11 |
| Ideal | 600 | about $35,100 | text r > 0 with power 0.63 to 0.66 for C, E, A; an increment of 0.10 with power 0.43 for one trait and 0.83 over all five; 0.05 still out of reach (0.09) |

Plus about $1,500 to $3,000 for ethics review, needed before the pilot. Cost is about $59 per usable person at
every scale and depends most on how many people come back for Session 2 (assumed 50%; at 35% the main study
costs about $18,100).

## Decisions the owner must make

1. **Budget and scale.** Pilot only (~$2,600 with ethics), main 250 (~$17,300 all in), ideal 600 (~$37,800), or
   250 with a pre-registered second stage to 600. Recommendation: run the pilot now; run 250 (with the option of stage 2) only after the PANDORA gate above is met.
2. **Ethics route.** Commercial IRB exempt determination (recommended, $1,500 to $3,000, 1 to 4 weeks, keeps
   commercial freedom) vs an academic collaborator (free, slower, cheaper Prolific fee, but data ownership and
   commercial use must be settled first).
3. **Pay for Prolific?** Recommended: yes, Prolific for pilot and main study (verified US/UK people, follow-up
   studies built in). CloudResearch Connect's 10-day fee-free window is an option for the pilot. Free channels
   (Reddit, own network) only for usability testing.
4. **Require writing (Reddit or X)?** Recommended yes: it anchors "best single source". No would widen the pool
   but split the increment test across many source pairs.
5. **Common-word block (opt-in).** Include (lets the study retrain the text model on ordinary people; adds a
   re-identification risk by linkage to public posts) or drop (safer, simpler consent, no text retraining).
   Recommended: include as a separate opt-in.
6. **OEJTS for MBTI.** Its licence is non-commercial (CC BY-NC-SA). Keep for the study only, or drop and rely on
   the loose Big Five mapping. Recommended: keep, never ship its items.
7. **UK GDPR representative.** Appoint one (a few hundred dollars a year) or rely on the occasional-processing
   exemption. Ask the reviewer.

## What must be built before the pilot

Built: `aas.features` (the derived payload, stdlib only, with no-free-text and canary privacy tests), the local
topic lexicon (20 non-sensitive topics), and `scripts/study_analysis.py`, run on simulated studies at n = 60, 250
and 600. Still to build: a Pyodide build of parsers + readers + features, the study pages and a small EU server
endpoint. This is the minimum data-collection instrument, not a product UI.

## Riskiest assumptions

1. **Session 2 return of 50%.** Data-donation studies lose many people at the request, download and upload steps,
   and it is not random. If return is 20%, costs double. The pilot measures it; the $8 completion bonus is the
   main lever.
2. **Text reader validity of about 0.20 on ordinary people.** It was trained on 1990s student essays and a typology
   forum; only Openness was clearly real even in domain. If it is near 0.1, "increment over the best single source"
   is an increment over almost nothing.
3. **Non-text sources have no reader.** Google and Spotify models must be learned from the study's own people; at
   n = 250 a true 0.15 signal shows up as about 0.05 under cross-validation. Weak sources will look like nothing.
4. **Increments of 0.05 are the realistic size and are out of reach** below about 1,300 people per trait.
5. **Eligibility tilts the sample** to people who write online, use a laptop and are willing to donate data. Results
   generalise to them, not everyone.
6. **The derived payload is still personal data.** Common-word rates and posting-hour patterns could in principle be
   matched to public Reddit/X accounts. Kept private, never published.
7. **The browser extractor is unbuilt.** Large Google Takeout files, old browsers and Safari memory limits are untested.

## Update from the fusion simulation (docs/fusion.md)

- Before any study data, the zero-data Bayesian fusion (`bayes`) is as good as any trained method at n = 600, so the study does not need to train the combiner. Its job is to check it: real validity per source against stated, interval coverage, and how much text and MBTI errors move together.
- Showing a fused signal is no longer cheap. At measured validities fused r > 0 needs 200 to 493 people for 80% power (200 O, 223 E, 313 N, 429 C, 493 A; POWER.md section 9). Coverage to ±5 points still needs 246, since coverage precision does not depend on validity.
- Proving fusion beats text alone is out of reach at 250: an increment of 0.10 has power 0.11 there (0.83 over five traits at 600), and 0.05 has power 0.06 at 600. It stays a secondary, exploratory outcome.
- Collect every source from every participant. Missing sources can be simulated later by masking; they can't be added back.

## How to run the analysis

The analysis is `scripts/study_analysis.py`. It was written and tested on simulated studies before any
data existed, and PREREG.md section 6 points to it. Run it once on a simulated study (that output is posted
with the registration) and once on the real data folder. Do not change it after the real folder exists; any
change before then goes in PREREG.md "Changes before data collection".

```
cd py
uv run python scripts/simulate_study.py --n 250 --out /tmp/sim250        # fake study, same format, plus truth.json
uv run python scripts/study_analysis.py /tmp/sim250                      # registered run: writes /tmp/sim250/results/
uv run python scripts/study_analysis.py /tmp/sim250 --fast               # quick look (2 repeats), not the registered run
uv run python scripts/simulate_study.py --dryrun 60,250,600 --out /tmp/dry   # simulate + analyse + recovery tables
uv run pytest tests/test_study_analysis.py                               # exclusions, Holm, recovery, determinism (~45 s)
```

Time for the registered run on this laptop: about 25 s at n = 60, 45 s at 250, 75 s at 600.

**Input folder** (one JSON file per person per session; the exact fields are in the FORMAT ADAPTER comment of
the script, and `load_study()` is the only function that reads them, so it is the one place to change when
`aas.features` fixes the payload):

| path | what | from |
|---|---|---|
| `study.json` | frozen extractor versions and hashes, pilot study IDs | written at the pilot's end |
| `session1/<id>.json` | IPIP-50 answers and item times, attention items, "use my answers?", OEJTS letters, retest assignment | Session 1 server |
| `payloads/<id>.json` | the extractor's derived payload (reader estimates, Google and Spotify features, floor flags) | Session 2 POST |
| `retest/<id>.json` | IPIP-50 retest answers and date | retest study |
| `withdrawn.json` | study IDs that withdrew | withdrawal codes |

**Output** (`<folder>/results/` or `--out`): `results.json` (every number, machine-readable) and `report.txt`
(the same in plain text: the exclusion flow table, H1 to H4, S1 to S7, and the PREREG issues list). Both are
identical on every run with the same folder and `--seed` (default 0); `input.folder_digest` in the JSON
fingerprints the input files.

**Reading the report.** "SUPPORTED" means the registered test passed after Holm correction within its
family; anything else is "not detected at this n" with the smallest detectable effect next to it. The
learned models (H2, S1, S4) are scored only on people they never saw, so with no real signal their r runs
below zero; read them against `null_mean`, never against 0.

**Dry-run check (simulated data only).** `simulate_study.py` builds a world with the *measured* reader
validities (POWER.md section 9: text r about .10 for C, E, A and .16 for N; openness refused for everyone
whose writing is not essay-like, 92% of people) and known answers. On three independent simulated studies the
population value fell inside the reported CI for 48 of 50 directly estimable quantities at n = 60, 52 of 53
at n = 250 and 52 of 54 at n = 600 (about 95% expected), and the stopping rule produced exactly n usable people
each time. Learned-model numbers (H2 Delta r, S1) sit below their oracle population values and near their
permutation null at small n, as POWER.md predicts. With these validities the registered primary H2 test was
"not detected" at 60 and 250 and passed at 600 (one simulated study each, not a power estimate).

**Openness.** Because the text reader refuses O for most people, every pooled-over-traits test runs over the
traits a method speaks for in at least 90% of people (C, E, A, N for the text reader; all five for the fused
model, whose O comes from the MBTI letter). The report prints that set, the number of people per trait, and
analyses O separately on the few who have it. This is proposal P22 in PREREG.md.

**Before posting the registration**, the owner decides the "Proposed" items at the end of PREREG.md. The
biggest is P20: the dry run showed the registered OLS stack can amplify noise from a weak source, and
non-negative stacking (`--stack-nonneg`) fixes it. P21 (how r is averaged over CV repeats) and P22
(openness) are already implemented as proposed.

## Update 2026-11-01: measured text validity (read this before deciding)

The text reader's accuracy has now been measured on kinds of writing it never saw (tweets; Reddit is not yet
measured and is given the same numbers). It is much lower than the study was planned around. Details:
`POWER.md` section 9, `../fusion.md` section 5b, `../readers_text.md` "Validity by kind of writing".

| | planned (above) | measured |
|---|---|---|
| text Big Five r on Reddit/X | 0.20 (0.10 to 0.30) | **0.097, 0.095, 0.094** for C, E, A; 0.158 for N; **openness refused** (no evidence it works off student essays) |
| MBTI letters off the forum | AUC 0.60 to 0.70 | about 0.61 to 0.62, **inferred** (no MBTI-labelled test outside the forum exists openly) |
| zero-data fused r (text + MBTI) | about 0.24 to 0.29 | about **0.15** (0.13 to 0.20 by trait) |
| main study n = 250: show text r > 0 per trait | separates 0.1 from 0.3 | power **0.33** for C, E, A; 0.71 for N; none for O |
| main study n = 250: increment of 0.10, omnibus | 0.56 | **0.11** |
| n for 80% to show text r > 0 | | 832 to 887 (C, E, A), 313 (N) |
| n for 80% to show fused r > 0 | 62 to 194 | 200 to 493 |

**Recommendation now:** run the 10-person pilot (it tests the instrument, not the model), **hold the main study**
until a text model retrained on PANDORA reaches about **r 0.20 on held-out Reddit authors for at least three
traits** (the smallest r a 250-person study can detect is 0.18). Request PANDORA access first: it is free and the
loader is written. If the retrained model stays near 0.10, the study's honest purpose becomes collecting data to
train a better model, not testing the current one, and PREREG should say that before anyone is paid.

New and changed risks:

- **Riskiest assumption 2 has come true.** Text validity on ordinary people is about 0.10, the pessimistic planning
  value. "Increment over the best single source" is now an increment over almost nothing.
- **Openness has no text reading for Reddit/X users.** The reader refuses it; any openness result rests on the MBTI
  N/S letter through an untested link.
- **The MBTI off-forum numbers are inferred, not measured.** Tweet spot checks were weaker than the inference for
  N/S and J/P. PANDORA (which has MBTI flair too) would measure them.
- **Reddit itself is not measured.** Its numbers are borrowed from tweets. PANDORA would fix this as well.

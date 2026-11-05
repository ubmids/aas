# AAS algorithm design (agreed in conversation, built after)

Status: designing. Each piece is agreed here before code.

## Decisions so far
1. **Runs on a server**, not in the browser. Reason: cross-user learning (personality needs many quiz-labelled users), big reference data (taste map, sound features, weather history).
   Privacy: strip in the browser before upload (IP -> rough city on device, raw IP never sent; no user agents), raw upload deleted after processing, one-tap delete-everything, model-training contribution is a separate opt-in. UK GDPR.
2. **Taste map**: artists as points (songs later for frequent ones), ~64 dimensions, learned mainly from **session co-listening** (captures mood), plus "same people like both" for discovery. Source: ListenBrainz public data. Known gap: non-Western artists may be missing; measure on real exports.
3. **Sound**: AcousticBrainz features (energy, tempo, valence, acoustic/electronic). Frozen since 2022, newer songs missing.
4. **Surroundings per play**: hour, weekday, season, holidays, gap since last session, session length; device (phone/desktop/car/speaker/cast); country + rough city; shuffle/offline/private; picked vs autoplay; daylight from date+location; **weather is core** (rain, cloud, humidity, wind, temperature, daylight hours, sudden changes), from historical weather data.
5. **Psychology layer** (measured, never pseudo-neuro): chronotype, mood-matcher vs mood-shifter, burn-out curve (Berlyne inverted-U), discovery age, nostalgia, life-change signals (show the change with receipts, never guess the cause), personality (Big Five) proven against a 10-question test.
6. **Selves**: unit = listening session (30 min gap). Each session = what (taste-map position + spread) + sound + behaviour. **Option A: music defines a self; surroundings describe it.** Soft mixture model, number of selves chosen by fit, typically 3 to 6; a self must recur across 3+ different weeks. Surroundings are NOT used to build selves so they can be tested: predict the self of a held-out session from its context alone (bench task). Handle: the "default you" (not the headline), shared accounts (gentle wording), selves born/dying feed eras. Names written by an LLM from measured facts only, with receipts. Card shows at most 5.

7. **Eras**: weekly snapshot (taste-map position + sound, mix of selves, rhythm, place, season/weather) -> change-point detection; an era must hold ~**season-sized** (min ~8 to 12 weeks, a few per year). Described with receipts. Rabbit holes (short intense one-artist/track bursts) are separate and can sit inside an era. Life-change signal = era boundary + rhythm change + city change: say "something changed", never the cause. **Silence counts**: gaps in listening and music->podcast switches are part of the story (gentle wording). Test: era-aware forecast of the next weeks must beat all-time taste (bench); plus a thumbs check from the user.

## Open
- TikTok. Personality is replaced by measured traits (below); a quiz could add labels later. ListenBrainz map built 21 Oct (below): coverage solved, falls ranking not, falls shown as a look back instead.

## v1 results (2026-10-20, 13 real people, held-out = last 8 weeks)
Run `npm run bench` (scores) or `npm run describe` (scores plus each person's selves and eras). Everything tuned is tuned on the last 8 weeks of each person's training period, never on the held-out weeks.

What was built (TypeScript, `engine/src/model/`):
- `map.ts` taste map v0: artist co-listening inside sessions (30 min gap, proper plays, runs of one artist collapsed, window 5, weight 1/distance) -> PPMI with context smoothing 0.75 -> top 48 eigenvectors (randomised) -> unit vectors. Built from everyone's training weeks only (7,828 artists). Falls use a leave-one-person-out map: the scored person never shapes it.
- `falls.ts` falls and the ranker: features are nearness to recent taste (last 8 weeks), all-time taste and the drift-extrapolated point, nearness to the 30 recent favourites, fit to the person's own jump size (mean and spread of how far their earlier falls sat from their taste, shrunk to everyone's), and popularity among other people. A logistic ranker is trained on 8-week windows inside other people's training periods.
- `selves.ts` + `gmm.ts`: session vector = map mean (PCA to 8) + spread + pick share, skip rate, repeat rate, length; standardised; full covariance Gaussian mixture, k by BIC in 2..7; components seen in fewer than 3 weeks are dropped. Context classifier = softmax regression over local hour (3h bins), weekday, device class, shuffle share, country.
- `eras.ts`: weekly rows (map position PCA 4, self mix, log plays, 4 hour-of-day shares, podcast share, silent flag; empty weeks kept and flagged) -> optimal partition with a 10 week minimum segment and penalty 1 x d x log n. Forecast weights plays in the current era 1 and earlier plays gamma, with decay; gamma, half-life and penalty chosen on validation.

| task | model | baseline(s) | wins |
|---|---|---|---|
| 3 the game, all rounds | 0.733 with map | 0.687 old scorer | 6/13 (3 ties) |
| 3 the game, new-artist rounds | 0.179 with map | 0.152 old scorer, chance 0.200 | 6/9 |
| 6 falls, held-out (52 falls, 31% on the map) | recall@20 0.019, @50 0.038 | popular 0.019 / 0.038, centroid 0.019 / 0.019 | 1/9 at @50 |
| 6 falls, rolling windows (407 falls, 39% on the map) | recall@20 0.020, @50 0.039 | popular 0.022 / 0.032, centroid 0.010 / 0.015 | @50: 4/10 vs popular, 6/10 vs centroid |
| 7 selves from context | accuracy 0.347, log-loss 1.753 | majority 0.222, prior 1.812 | 7/13 (3 ties), log-loss 10/13 |
| 8 era-aware forecast | NDCG@10 0.813 | decay 0.813, all-time 0.801 | 0/13 vs decay (9 ties) |
| 8 current era only | NDCG@10 0.824 | decay 0.813 | 4/13 |

Old tasks unchanged: skip AUC 0.848 vs 0.617 (13/13), next artists NDCG 0.813, clock 0.117 bits, novelty AUC 0.765.

What worked
- The map reads right by eye: neighbours of rock, hip hop, Bollywood, J-pop and lofi artists are their genre and mood mates, from 13 people alone.
- Selves are readable and recur (median 7 per person, each across tens to hundreds of weeks), and context predicts them above the majority class for most people. One person: 0.579 vs 0.129 majority, the late-night self is very predictable from the hour.
- Eras read as real seasons: silences, volume jumps and favourite turnover line up with the boundaries.
- The map lifts the game by about 5 points and gets new-artist rounds up a little.

What did not work (yet)
- Falls are not predicted. Only 31 to 39% of falls are on a 13-person map at all, and recall@50 barely beats popularity. The map is the bottleneck, not the ranker. Falls are also rare (median 1 per person in 8 weeks), so the held-out number is noisy; the rolling windows are the fairer read.
- New-artist game rounds stay under chance (0.179 vs 0.200): known-artist decoys still win, and the true new artist is often off the map.
- Era-aware forecasting does not beat plain decay: validation usually picks gamma 1 (= decay). "Current era only" wins for 4 people, so eras are useful as a story and maybe as a feature, not as a forecast on their own.
- BIC nearly always picks the cap of 7 selves; a share floor or a stronger penalty is needed to hit the 3 to 6 we want. Local hours use a rough country offset table without DST.

Next
1. ~~ListenBrainz map~~ done 21 Oct, see below. Options were: the public similar-artists endpoint (labs.api.listenbrainz.org similar-artists by MBID) for the artists in an export, or the ListenBrainz session dumps for a real co-listening map. Needs name -> MBID mapping (MusicBrainz search, 1 request/s) or the canonical artist dump. Should lift the falls ceiling from about 35% to most falls. Must stay under a few hundred MB.
2. AcousticBrainz sound features (energy, tempo, valence, acoustic/electronic) per track, frozen 2022.
3. Weather per session from historical data (rough city + date), as context for selves and eras.
4. Personality needs quiz-labelled users; nothing to fit until then.
5. Selves: share floor or k penalty; eras: use era features inside the next-artists and falls rankers rather than as a standalone forecast.

## ListenBrainz map results (2026-10-21, same 13 people, same splits)
Build: `npx tsx bench/scripts/lb_build.ts` (resumable, caches everything in bench/data/lb/, gitignored). Then `npm run bench` scores v0, the ListenBrainz map and the blend side by side (`--no-lb` for v0 only). The full bench now takes about an hour, almost all of it falls on the big maps.

How it was built
- Names to MusicBrainz ids: labs.api.listenbrainz.org acr-lookup on each artist's top tracks, raw and with Spotify suffixes stripped (" - 2005 Remaster", "(feat. X)"), 100 per request. An exact credit match counts; another spelling (often another script, Hitorie and ヒトリエ) needs two tracks agreeing. Then a MusicBrainz artist search for the misses with 10+ plays (name, alias or the sort name read forwards; one clear match only). MusicBrainz throttles a plain user agent to about one answer per 10 seconds, so the search ran alongside and stopped at 723 of 1,014; rerun without `--no-search` to finish it. 9,995 of 16,287 artists resolved (61%), 6,013 of 7,657 seeds. 76% of falls resolve.
- Lists: labs similar-artists, session based (`session_based_days_7500_session_300_contribution_5_threshold_10_limit_100_filter_True_skip_30`), top 100 per artist, one request per artist (the POST form only answers the first). Fetched for every seed (3+ proper plays in someone's training weeks) and for every artist on a seed's list, so where an artist sits never depends on whether someone in the bench played it. 32,499 lists, 95,602 artists. Raw cache 143 MB, table.json 28 MB.
- Map (`engine/src/model/lbmap.ts`): the lists become one graph (score over the list's top score), then the same PPMI and eigenvector steps as v0, 64 dimensions. A point needs its own list or two lists naming it. Falls use a leave-one-person-out map built from other people's seeds and one hop out (about 69k artists); the game uses everyone's seeds (69,806). The blend joins v0 and ListenBrainz half and half. New ranker features: how many lists name the artist, and how strongly the person's recent artists list it.

| task | v0 map | ListenBrainz | blend |
|---|---|---|---|
| falls on the map, held-out (52) | 0.308 | 0.635 | 0.654 |
| falls on the map, rolling (407) | 0.388 | 0.624 | 0.654 |
| falls recall@50 held-out, ranker | 0.038 | 0.019 | 0.038 |
| falls recall@50 rolling, ranker | 0.039 | 0.025 | 0.020 |
| rolling @50 baselines: popular / centroid / lists | 0.032 / 0.015 / 0.032 | 0.032 / 0.010 / 0.039 | 0.032 / 0.022 / 0.039 |
| game, all rounds (old scorer 0.687) | 0.733 (6/13) | 0.687 (2/13) | 0.742 (7/13) |
| game, new-artist rounds (old 0.152, chance 0.200) | 0.179 | 0.172 | 0.179 |
| held-out new artists on the game map | 0.348 | 0.485 | 0.556 |

"lists" is ListenBrainz's own answer with no learning: rank unheard artists by how strongly your recent artists list them.

What worked
- Coverage. Falls on the map went from about a third to about two thirds, which was the goal. The rest are mostly artists MusicBrainz could not resolve (Japanese and other non-Latin names, small producers) or that no list names.
- The blend is the best game scorer so far, 0.742 against 0.733, 7 of 13 people beating the old scorer.

What did not work
- Falls are still not predicted. Finding the right artist on the map was not the real problem; picking it out of 69,000 is. With the bigger pool the ranker does worse than on v0 (rolling recall@50 0.025 against 0.039), and nothing beats popularity by much. Checked directly: 40% of falls are on a recent artist's list, but at a median rank of 918, and the plain "lists" ranking gets 0.035 recall@50 over all 459 falls. A fall is usually not a near neighbour of what you play now.
- New-artist game rounds stay under chance on every map.
- The ListenBrainz map alone is worse than v0 for the game: within a person's own taste, co-listening from real exports beats a public similarity graph.

Read: as a falls predictor this is a dead end with what we have. ListenBrainz is worth keeping for coverage (placing artists nobody in the bench played) and in the blend for the game. Before any more falls work, ask whether falls should be a card at all, or be shown as "where your new artists came from" (looking back, with receipts) rather than a forecast.

## Where your new artists came from (2026-10-21, same 13 people)
Falls are now a look back, not a forecast. `npm run doors` prints the table below; `npm run doors -- --story` also prints each person's summary and stories. Engine: `engine/src/model/doors.ts`.

What it reads, for every artist first heard after a person's first 12 weeks:
- a fall: 10+ proper plays on 3+ days within 8 weeks of the first play; tried and dropped: 2 proper plays or fewer, ever.
- the door, at the first play only (nothing after it, so the facts cannot leak the outcome): how the play started (clickrow and playbtn count as picked; trackdone and fwdbtn as came on), the artist just before it in the same session, that artist's ListenBrainz rank for it, the best rank among the person's 30 most played artists of the 8 weeks before, shuffle, and whether the first play was a skip. For falls, days to the 10th proper play.
- one door per artist: picked; came on after a similar artist (the artist before lists it); next to a favourite (a recent favourite lists it); came on by itself; out of nowhere.

Is it Barnum? Every fall is compared with the 5 artists the same person tried and dropped whose first plays were nearest in time (within 60 days), and, for the list facts, with 5 never-played ListenBrainz artists named by as many lists. A door counts as evidence when the pooled rate ratio is at least 1.5, the 95% bootstrap interval over people sits above 1, and falls show it more in most people. 1,821 falls, 1,801 with controls, 17,374 dropped artists.

| fact at the first play | falls | tried and dropped | ratio [95%] | people | verdict |
|---|---|---|---|---|---|
| picked | 0.423 | 0.377 | 1.12 [0.89, 1.51] | 9/11 | no difference |
| came on | 0.554 | 0.598 | 0.93 [0.75, 1.09] | 1/11 | no difference |
| artist before lists it | 0.191 | 0.106 | 1.80 [1.16, 2.37] | 7/10 | more for falls |
| a recent favourite lists it | 0.422 | 0.237 | 1.78 [1.29, 2.19] | 8/11 | more for falls |
| same, in its top 25 | 0.224 | 0.091 | 2.47 [1.72, 3.49] | 9/11 | more for falls |
| artist before is a recent favourite | 0.285 | 0.234 | 1.22 [1.13, 1.40] | 11/11 | small but steady |
| picked straight after a favourite | 0.114 | 0.080 | 1.43 [1.23, 1.78] | 10/11 | small but steady |
| shuffle on | 0.432 | 0.383 | 1.13 [0.94, 1.38] | 6/11 | no difference |
| first play was a skip | 0.193 | 0.495 | 0.39 [0.33, 0.44] | 0/11 | less for falls |

| door | falls | tried and dropped | ratio [95%] | verdict |
|---|---|---|---|---|
| picked | 0.423 | 0.377 | 1.12 [0.89, 1.51] | no difference |
| came on after a similar artist | 0.084 | 0.035 | 2.40 [1.15, 3.55] | evidence |
| next to a favourite | 0.129 | 0.052 | 2.47 [1.57, 3.07] | evidence |
| came on by itself | 0.355 | 0.514 | 0.69 [0.57, 0.80] | less for falls |
| out of nowhere | 0.008 | 0.022 | 0.37 [0.10, 0.77] | less for falls |

Against never-played artists of matched list count, a recent favourite lists 42% of falls against 8% of controls (5.0x, 12 of 12 people), median best rank 24 against 43. Falls grow fast: median 15 days from first play to the 10th proper play (quartiles 5 to 31).

What it means
- The doors that are real are the list ones: a new love is 2 to 2.5 times more likely than a dropped artist to sit next to what you already play, or to come on after an artist that lists it. That is the honest version of "the algorithm found it for you".
- Picked or came on, on its own, says nothing across the bench. It does for some people: for 5 of them the falls were picked far more often than their own dropped artists (73% against 11% for one), so the summary judges each person against their own dropped artists first, and only then against the bench.
- Hearing it all the way through the first time matters most: a fall's first play was a skip 19% of the time, a dropped artist's 50%.

How the cards use it
- Every fall gets a story made only of facts with receipts, e.g. "You first heard Funkadelic on 2025-10-02: it came on after Parliament. Parliament lists Funkadelic at #1 of its similar artists on ListenBrainz. You'd played it properly 10 times 2 days later."
- The per-person line names a door as "how new music sticks for you" only when it is 1.5x more common for that person's falls than their dropped artists (10+ falls, 20+ dropped), or when it is a bench-level evidence door. Otherwise it only counts and compares ("Your 302 new loves came in many ways. The most common way in (110 of them): they came on by themselves after something else, less often than for artists you tried and dropped (65%).").

Limits: reason_start cannot tell autoplay from a playlist or album running on, so "came on" never says autoplay. One person has no reason_start at all; only list facts apply there. 39% of artists do not resolve to ListenBrainz, so list facts are compared on resolved artists only.

## How many selves (2026-10-21)
BIC nearly always picked the cap of 7. `npx tsx bench/selves_k.ts` compares a stronger BIC penalty (the parameter term times 2, 4, 8 or 16), a 5% share floor, and a penalty chosen per person on validation (fit on the training period before its last 8 weeks, pick the mixture that best explains those 8 weeks, among fits with 3 to 6 selves; then refit on all training). Scored on task 7, the held-out weeks.

| selves rule | selves (median) | accuracy | majority | context gain (nats) | log-loss wins |
|---|---|---|---|---|---|
| BIC (before) | 7 | 0.347 | 0.222 | 0.036 | 10/13 |
| BIC + 5% floor | 6 | 0.364 | 0.222 | 0.036 | 10/13 |
| penalty x2 | 5 | 0.370 | 0.270 | 0.034 | 9/13 |
| penalty x4 | 4 | 0.470 | 0.407 | 0.034 | 8/13 |
| penalty x8 | 2 | 0.556 | 0.256 | 0.030 | 8/13 |
| chosen on validation (kept) | 5 | 0.461 | 0.256 | 0.036 | 9/13 |

Accuracy is not comparable across rows: fewer selves make every guess easier, majority included. The fair read is the context gain (log-loss of the prior minus the model's): how much the surroundings tell you about which self is listening. It barely moves, 0.036 nats with 7 selves and 0.036 with the validated 5, and only drops once the penalty forces 2. So fewer selves cost nothing measurable, and 5 is what a card can show.

Kept: `fitSelvesValidated` (penalty from 1, 2, 4, 8, 16 by validation likelihood, 3 to 6 selves), median 5 selves per person, accuracy 0.461 against majority 0.256, 7 of 13 above majority. The share floor changed nothing once the penalty was right (no self under 5% survived), so it is an option, off by default. Context still explains little about which self is on: the selves are real in the music, and the surroundings we have only nudge them.

## Local hours with daylight saving (2026-10-21)
The rough standard-time offset table is gone. Each country maps to its main IANA zone (`ZONE` in `engine/src/model/session.ts`), and `zoneOffset` reads the real offset at each play from Intl (cached per half hour), so local hours follow DST. Unknown countries fall back to the person's usual one. The clock task now reads local hours (it used UTC before), and the skip model's hour feature does too. Quick bench (`--no-lb`), same splits:

| task | before (offset table; clock on UTC) | after (IANA zones with DST) |
|---|---|---|
| 4 when you listen, bits/play over uniform | 0.117 (UTC) | 0.123 local, 7/13 people better than UTC |
| 4 vs the all-time clock | 0.131 | 0.135 |
| 7 selves from context, accuracy (majority 0.256) | 0.461 | 0.482 |
| 7 log-loss (prior 1.529) | 1.453 | 1.450 |
| 1 skip AUC | 0.848 | 0.848 |

Small but in the right direction everywhere. Limit: multi-zone countries (US, CA, MX, BR, AU, RU, ID) read as their main zone; a person in Los Angeles is three hours off all year. The fix is asking for a zone in the app, or reading it from the listening rhythm.

## Traits, measured instead of inferred (2026-10-21)
Decided with Subomi: no inferred Big Five. AAS shows measured listening traits instead, each a number read straight from the history, with its receipt and a percentile against other people (the 13 bench people now, the app's users later). Engine: `engine/src/model/traits.ts`; bench: `npm run traits` (`--cards` prints one person's lines).

| group | trait | what it is |
|---|---|---|
| adventurousness | newShare | share of proper plays going to artists first heard in the 4 weeks before |
| adventurousness | spread | 1 minus the length of the play-weighted mean of artist vectors on the taste map |
| loyalty | oldFriends | in each 8-week stretch, share of plays going to artists in your top 50 two or three stretches earlier (4 to 6 months before) |
| patience | finished | share of plays whose reason_end is trackdone |
| patience | skipSeconds | median seconds played before a skip (under 30 s, ended by forward or unknown) |
| clock | clockHour | circular mean of local listening time, weighted by minutes, read on a 4 am to 4 am day (DST-correct, needs a country) |
| clock | nightShare | share of listening minutes between midnight and 5 am, local |
| era turnover | turnover | share of the top 10 replaced from one 13-week season to the next |

Stability: each trait measured on the two halves of every person's training time separately (the second half only uses the first to know which artists were already familiar), then correlated across people. Stable needs Pearson and Spearman both at 0.5 or more; flagged when only one is (shown with "read it as now, not always"); dropped when neither is (measured, never shown).

| trait | people | r | rho | status | bench range (median) |
|---|---|---|---|---|---|
| newShare | 10 | 0.723 | 0.758 | stable | 0.035 to 0.333 (0.136) |
| spread | 12 | 0.628 | 0.476 | flagged | 0.222 to 0.533 (0.387) |
| oldFriends | 10 | 0.680 | 0.479 | flagged | 0.202 to 0.903 (0.425) |
| finished | 11 | 0.864 | 0.782 | stable | 0.130 to 0.751 (0.570) |
| skipSeconds | 12 | 0.988 | 0.902 | stable | 1.07 to 15.16 s (2.07) |
| clockHour | 11 | 0.531 | 0.700 | stable | 12:51 to 02:30 (18:04) |
| nightShare | 11 | 0.348 | 0.364 | dropped | 0.048 to 0.250 (0.168) |
| turnover | 10 | 0.883 | 0.855 | stable | 0.200 to 0.747 (0.569) |

The first loyalty measure, "share of a season's top 20 still played 6 months on", failed badly (r -0.01, rho 0.25) and was replaced by oldFriends; two other tries (top 20 still played next season, comebacks after a silent season) were worse. People with too little history in a half get no value, hence n under 13.

Overlap: the traits are not independent. Across people, oldFriends runs against newShare (rho -0.77), spread (-0.81) and turnover (-0.77), and spread with turnover (0.76); clockHour and nightShare are nearly the same thing (0.93), another reason to drop nightShare. With 13 people this is a sketch; it suggests two underlying axes (explore vs stay, and when), worth rechecking on real users before cards say "you are both very loyal and very adventurous".

Percentiles: share of other people below the value, ties half; with 12 others they move in steps of about 8 points, so cards round and say "more than N% of listeners we've measured".

People like you tend to: a line only where published research backs it, worded as a tendency about people in general, never a claim about the person, never a sensitive trait. Adventurousness, patience and turnover have no line: the personality research we know ties openness to genres, not to how much someone explores, so nothing published backs a line. Citations checked against the published records on 21 Oct 2026, and each line says no more than its paper found.
- oldFriends: "Most people pick familiar songs more often than they themselves expect to." Ward, Goodman and Irwin (2014).
- clockHour: "People's natural daily timing differs by several hours from one person to the next." Roenneberg, Wirz-Justice and Merrow (2003).
- nightShare (dropped, kept for when it is stable): "Across millions of listeners, late-night listening tends to be calmer and more relaxed than daytime listening." Park et al. (2019).

Sources
- Ward, M. K., Goodman, J. K. and Irwin, J. R. (2014). The same old song: the power of familiarity in music choice. Marketing Letters, 25(1), 1 to 11.
- Roenneberg, T., Wirz-Justice, A. and Merrow, M. (2003). Life between clocks: daily temporal patterns of human chronotypes. Journal of Biological Rhythms, 18(1), 80 to 90. doi:10.1177/0748730402239679
- Park, M., Thom, J., Mennicken, S., Cramer, H. and Macy, M. (2019). Global music streaming data reveal diurnal and seasonal patterns of affective preference. Nature Human Behaviour, 3(3). doi:10.1038/s41562-018-0508-z

Quiz hook: `TraitLabel` (trait group, answer, scale, when asked) is the shape for self-report answers from a later in-app quiz, so traits can be checked against what people say about themselves. Nothing collects them yet.

Example card lines (one bench person): "Tracks played to the end: 71%. 74726 of 105581 plays ran to the end of the track. That's more than 91% of listeners we've measured." and "The middle of your listening day, local time: 19:02. ... That's later than 73% of listeners we've measured. Research: People's natural daily timing differs by several hours from one person to the next."

## Statistics, the two-stage ranker and the maths (2026-10-26)
Direction from here: the algorithm and its maths only, no app work, so it can become a case study. `npm run report` runs the bench, doors, traits, the two-stage ranker and the will-it-stick model and writes bench/out/REPORT.md (a snapshot without names is in docs/REPORT.md). docs/MATH.md defines every model, its objective and its result.

What the statistics say with 13 people. Every headline number now has a bootstrap interval over people, and every model is compared with its baseline by an exact sign test, Holm-adjusted across 23 comparisons. Two results survive: the skip model (13 of 13 people, adjusted p 0.006) and the new-artist model (12 of 12, adjusted p 0.011). The skip model is also well calibrated (ECE 0.010, Brier 0.109 against 0.176 for the artist rate, lower for every person). The game beats chance for 12 of 13 people but misses after the adjustment (0.072). Selves from context help on average (mean log-loss gain interval 0.019 to 0.140 nats) but only 9 of 13 people agree. Next artists, the clock, the map in the game, eras and every new-loves ranker do not survive.

Two-stage ranker for new loves (engine/src/model/rank2.ts, bench/rank2.ts). Candidates are the unheard artists on recent favourites' ListenBrainz lists plus the 300 nearest map artists; a logistic ranker scores them from list evidence (listers, weighted reciprocal ranks, scores), popularity, map nearness and the person's own history of list-borne loves, leave one person out. Rolling recall@50 pooled 0.054 against 0.042 for the lists alone and 0.032 for popularity, 7 of 10 people above popularity (p 0.18); held-out 0.038, the same as popularity. Stage one only covers 22% of loves. It does not clear the bar, so new loves stay a look back (doors), and no more ranker work is planned.

Doors under the stricter reading: the two passing doors hold for 8 and 9 of 11 people, short of a multiple-test adjustment; the never-played comparison (12 of 12) survives. Traits: only skip seconds has both stability intervals above 0.5; new share, finished and turnover are clearly above zero.

## Will a new artist stick (2026-10-26)
The forecast that failed (which of tens of thousands of unheard artists someone will love next) is replaced by one that can be checked: after someone hears an artist for the first time, will it become a new love? Code in engine/src/model/stick.ts, bench in bench/stick.ts (`npm run stick`, also inside `npm run report`), definitions in docs/MATH.md section 14.

Every first play after a 12-week warm-up is a row (26,401 in the training periods, 6.7% stuck). Two moments: the end of the first session, and the end of the first 7 days (rows already past the bar by day 7 are left out). Features read nothing after that moment; a 1% sample is re-read from the history cut at the horizon on every run and must match (614 rows, 0 differed). Logistic regression, trained on other people only.

| after | model AUC (median per person) | best baseline | survives 32 comparisons |
|---|---|---|---|
| first session | 0.773 [0.721, 0.812] | popularity 0.628, first-play skip 0.593, own base rate 0.484 | against the base rate (12/12); against skip and popularity only before adjustment (11/1) |
| first session, held-out weeks | 0.831 [0.766, 0.918] (6 people) | popularity 0.634 | 6/6, not after adjustment |
| first 7 days | 0.899 [0.868, 0.922] | first-week play counts 0.880 | against popularity (12/12); the gain over counts is weak (9/2) |

Labels shuffled across all training rows give 0.53 and 0.45, so the score is not leakage. Shuffling only inside each person gives 0.58: that part comes from people differing in how often new artists stick. Calibrated: pooled ECE 0.007.

What carries it is the first session itself (how much of the artist is played through, repeats, coming back to it): dropping it costs 0.051, alone it gets 0.736. ListenBrainz lists add 0.009 and how it started 0.006; the map, popularity, time of day and the first play alone add nothing once the session is in. After 7 days the first week's counts do nearly all of it. Boosted trees were tried as a check and were no better (0.764, 0.895), so the logistic regression stays.

So the story is: where an artist came from explains a little (the doors), and what you do with it in the first session decides most of it. The card can make a checkable claim the day you meet an artist, and say a week later whether it was right.

## Round 3: 343 ListenBrainz listeners (2026-10-26)
The 13-person Spotify bench could not tell small real effects from noise. Round 3 re-tests everything that does not need skips on 343 public ListenBrainz histories (bench/lbusers), 304 of them long enough for the same 8-week split. Run: `npm run lb3` (bench/lb3: prep builds popularity and five fold maps, person reads each person on their own and caches it, models fits the stick model and the ranker per fold, analyse writes bench/out/REPORT_LB.md; `npm run report` folds it into REPORT.md). Each step caches and resumes. Every learned thing is fitted on four of five folds of people and scores the fifth. Definitions and the full table are in docs/MATH.md section 17.

What this data cannot test: finish or skip, selves, the game, local clock hour and night share, finished and skip seconds, and the doors that need start reasons. Those still rest on 13 people.

Results, Holm across the round's 26 sign tests:
- Will a new artist stick holds without skip data. With only the columns ListenBrainz can fill, the model reaches 0.792 [0.784, 0.800] after the first session against 0.466 for the base rate and 0.618 for popularity (301/2 and 283/20 people), 0.811 on the held-out weeks, and 0.884 after 7 days against 0.860 for the week's counts (257/46). On the Spotify bench the dropped columns (skips, play lengths, start reasons, shuffle, hour) add 0.014 after the first session and nothing after 7 days. So the result rests on what the person does in the first session, not on Spotify's skip data.
- Two dials. Every trait this data can measure is stable (both split-half intervals above 0.5). Parallel analysis keeps two components in every bootstrap resample: explore against stay (new share, spread and turnover against old friends, loadings 0.79 to 0.94) and timing regularity (clock spread against clock regularity, 0.96). The second dial is about how fixed listening time is, since clock hour cannot be read without a time zone, and it rests on two measures that are close to one quantity. Spread and old friends move from flagged to stable.
- Doors. "A recent favourite lists it" against never-played artists: 5.3 times, 302 of 302 people. Against tried-and-dropped artists: 1.9 times (2.5 for the top 25 of the list). "Came on straight after an artist that lists it", read without start reasons: 1.4 times, consistent but under the 1.5 bar.
- Flipped with more people: decay beats all-time for next artists (218/72), the clock beats uniform (268/36, UTC hours), and the two-stage ranker beats popularity on rolling windows (recall@50 0.047 against 0.024, 132/24 with 137 ties). The ranker is still tiny in absolute terms (stage one covers 18% of loves), so it stays off the cards. Current era only is now worse than decay (107/192); era-aware forecasting still ties decay.
- List leak. ListenBrainz builds its similar-artist lists from its users' sessions, these people included. The stick model loses 0.002 without the list features. The heaviest third of listeners shows a larger list door than the lightest third (5.6 against 4.5 times), which is what a leak would look like, but the lightest third still shows it for 101 of 101 people. The door is real; part of its size may come from the leak.

New in the engine: `clockShape` (clock spread and clock regularity, both unchanged by a time zone offset), `mapFromSequences` (the co-listening map from streamed id sequences, used for the fold maps), `weekly` without selves, `noSkipFeaturesOf` for the stick model, and normal scores, parallel analysis and varimax in stats.ts. The verdict rule now requires the model to win more people than it loses before a significant sign test counts for it, and calls a significant loss "worse than baseline".

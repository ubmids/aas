# ListenBrainz user bench

A second, bigger bench of real listening histories, from public ListenBrainz profiles. The
Spotify bench has 13 people with full detail (skips, how each track started, device, country).
This one has a few hundred people with less detail per play, so it can test the parts of AAS that
only need what was played and when.

Nothing here is redistributed. The data stays in `bench/data/lbusers/` (gitignored). Users are
stored under a salted hash; the salt and the username list never leave that folder, and no
username appears in any committed file or report.

## Running it

```
npx tsx bench/lbusers/discover.ts 600        # find qualifying users (resumable)
npx tsx bench/lbusers/fetch.ts 360 60000 8   # download their listens (resumable, per user)
npx tsx bench/lbusers/normalise.ts           # clean, exclude, write plays/<pseudonym>.json.gz
```

Load from code with `bench/lbusers/load.ts`: `loadLbUsers()` yields `{ id, plays }` in the engine's
`Play` shape.

Requests go to api.listenbrainz.org with the User-Agent `aas-bench/0.1` and no contact address,
no token, inside the limits the API announces in its `X-RateLimit-*` headers. The transport is curl
because Node's own sockets to that host time out on this Mac while curl and Python connect.

## How users were found

A walk over ListenBrainz's similar-users graph, starting from ten long-standing public accounts of
the project's own developers (listed in the gitignored `bench/data/lbusers/seeds.txt`), 40 users
expanded per round. A user qualifies when:

- they have at least 5,000 listens and at most 2,000,000,
- their first and latest listens are at least 365 days apart,
- their latest listen is within 120 days of the walk (26 Oct 2026).

The walk saw 1,477 accounts and checked 1,059 of them: 631 qualified, 326 had too few listens,
55 too short a history and 47 were inactive.

The walk follows taste similarity, so the sample leans towards the kind of people who use
ListenBrainz: self-hosters (Navidrome, Jellyfin), Last.fm veterans, Western rock, metal and
electronic listeners. Users are then taken in pseudonym order, not walk order.

## What was downloaded

Each user's most recent listens, newest first, 1,000 per page, up to 60,000 listens; a heavy
listener whose 60,000 cover under two years is extended until two years are covered or 150,000
listens are reached. Pages are trimmed (cover art and streaming links dropped) and stored gzipped.

360 of the 631 were downloaded (the first 360 by pseudonym). 350 finished; 10 gave up after
repeated network drops and server errors and are left out (a rerun of `fetch.ts` resumes them).
Users under 60,000 listens have their whole history; 30 heavy listeners were extended past 60,000
to cover two years. Raw pages take 1.3 GB, the cleaned plays 0.76 GB, both gzipped.

## Cleaning and exclusions

Per user, in order:

1. Overlap between pages removed by timestamp and listen id.
2. Listens without an artist or title, before 2002 (before scrobbling existed) or in the future dropped.
3. Duplicate scrobbles dropped: the same track again within 30 seconds (two clients, retries, the
   Last.fm importer run twice).
4. The user is excluded when any of these hold:
   - more than 10% of listens share their exact second with 3 or more others (a bulk import that
     collapsed timestamps),
   - more than 25% of gaps between different tracks are under 30 seconds (faster than anyone listens;
     ListenBrainz only records a listen after half the track or four minutes),
   - more than 2% of active days have 500 or more listens,
   - one track is more than 30% of all listens (loops, sleep sounds, test accounts),
   - one exact gap length is more than 30% of all gaps under an hour (scripted pacing),
   - fewer than 5,000 listens, under 365 days or under 26 active weeks remain.

Of the 350, 343 are kept and 7 excluded:

| reason | users |
|---|---|
| gaps under 30 s between different tracks above 25% | 6 |
| 500+ listens on more than 2% of active days | 2 |
| one gap length above 30% of short gaps | 2 |
| bulk-import timestamps | 1 |

(Some users hit more than one rule.) Nobody failed the volume, span or one-track rules after the
two-year extension. 323,991 duplicate scrobbles and 14 impossible timestamps were removed in all.

The kept bench: 343 users, 15,885,726 listens. Per user, listens have quartiles 22,822 / 58,295 /
59,881 (5,217 to 149,841); the kept span has quartiles 895 / 1,561 / 2,964 days (at least 372);
active weeks 113 / 190 / 313. Listening runs from February 2005 to 26 October 2026; the median user's
first kept listen is June 2022.

## What this data can and cannot test

It has: when each listen started, artist credit, track, release, MusicBrainz ids for most listens,
the submitting client, and whether the listen was imported later.

It does not have: time played (`ms_played`), skips, how a track started (`reason_start`) or ended,
shuffle, device or country. ListenBrainz only stores completed listens, so every listen is a proper
play and there is no skip signal at all. `load.ts` sets those fields to null and sets `ms` to the
track length when known (else 180 s), so length filters treat each listen as played.

So it can test: next artists, new loves and the will-it-stick model without first-play skip or
route features, eras, the listening clock (in UTC, or with a time zone the user's clock implies;
there is no country), the traits that need only plays and times (adventurousness, loyalty,
turnover, clock), and the two-dials structure with a few hundred people.

It cannot test: skip prediction, patience traits, the doors that need `reason_start` (picked it
yourself, came on by itself), selves from device or shuffle context.

## Data-quality notes

- Most history is imported. For the median user 81% of listens were inserted more than 7 days
  after they were played, almost all by the Last.fm importer (the top client for 216 users) or the
  ListenBrainz archive importer (39). Last.fm keeps each scrobble's own time, so the timestamps are
  real; the `imported` flag only says the listen arrived later. 52 users are over 95% imported,
  38 under 5%.
- Imports are where the duplicates come from: running the old and the new Last.fm importer over
  the same history doubles it, so 35 users had more than 5% of their listens removed as duplicates.
- Track length is rare: only 87 users have it on more than half their listens, so nothing here
  should lean on duration.
- MusicBrainz ids are on 93% of the median user's listens, which makes the ListenBrainz
  similar-artist lists and the map usable without name matching.
- Exact one-second clusters are rare; one user was excluded for them. Timestamps before 2002 or
  in the future were 14 listens in all.
- The automated-looking accounts were dense, fast and evenly paced (gaps of the same few seconds,
  hundreds of listens a day). Thresholds are deliberately loose, so a borderline real heavy
  listener is kept rather than dropped; the per-user signals are in `summary.json` for a stricter cut.
- Sample bias: found through the similar-users graph and from developers' accounts outward, so
  these are ListenBrainz people (self-hosters, Last.fm veterans), not a cross-section of Spotify
  listeners. Results here say how well a model generalises across people, not to the general public.

## The fresh set (statements round 2)

`LB_SET=fresh` selects a second, separate download used to confirm the statements lane on people
nobody had looked at (`py/docs/statements.md`, round 2). On 5 November 2026 the walk was continued
from where it stopped (`discover.ts 770`, same rules); it reached 817 qualifying accounts. The fresh
set is the qualifying accounts that the first download never started, in pseudonym order, up to 370:

```
LB_SET=fresh npx tsx bench/lbusers/fetch.ts 370 60000 6
LB_SET=fresh npx tsx bench/lbusers/normalise.ts
```

It shares the salt and `candidates.json`, and keeps its pages, plays and summary in `raw_fresh/`,
`plays_fresh/` and `summary_fresh.json`, so the first bench is untouched. One exclusion was added to
the cleaning rules for both sets: one artist above 75% of listens, or fewer than 50 different
artists (the first bench had one such account; its plays file is left in place and the statements
code applies the same rule when it loads people). A first try with 20 parallel downloads lost
contact with every MetaBrainz host for a few minutes, so the fresh download runs with 6.

All 370 downloaded without a failure (1.5 GB of raw pages). Cleaning kept 359: 7 excluded for
listens faster than listening, 5 for 500+ listens a day, 2 for scripted pacing, 1 each for bulk-import
timestamps, one track dominating, too few listens and under 12 months (some hit more than one rule),
none for the single-artist rule. 477,608 duplicate scrobbles and 545 impossible timestamps were
removed; the kept set holds 16,843,110 listens, median 57,640 per user over a median 1,620 days.

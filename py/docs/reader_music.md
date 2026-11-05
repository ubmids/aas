# Reader: music (how you listen, with receipts)

`aas.readers.music.MusicReader` turns `Person.plays` into `Estimate`s of `kind="measured"`. It describes how someone listens. It does not guess personality: music taste predicts the Big Five only weakly, so the October rounds (TypeScript engine, `engine/src/model/traits.ts`, `docs/MATH.md` section 15) measured listening traits instead, kept the ones that are stable over time, and checked them on 304 ListenBrainz listeners. This reader is a faithful Python port of those.

```python
from aas.readers.music import MusicReader
r = MusicReader()
ok, why = r.can_read(person)
for e in r.read(person):
    print(e.trait, e.value, e.percentile, e.usable, e.evidence)
```

## What it measures

A "full play" is one of 30 seconds or more (Spotify's own threshold). Plays without a length (`ms_played=None`, e.g. ListenBrainz) count as full plays, weighted as 3 minutes on the clock. Times are start times (`Play.when` minus `ms_played`), read in UTC.

| trait | definition | split-half r | reference |
|---|---|---|---|
| `new_share` | share of full plays going to artists first heard in the previous 28 days, after a 12-week warm-up | 0.754 | 304 LB listeners |
| `old_friends` | in each full 8-week stretch, the share of full plays going to artists in the top 50 of the stretches 2 and 3 before (4 to 6 months earlier), averaged over stretches with more than 100 plays | 0.811 | 304 |
| `season_turnover` | mean share of the top 10 artists replaced between consecutive 13-week seasons | 0.802 | 301 |
| `clock_spread` | 1 − R, R = length of the minutes-weighted mean of unit vectors at each full play's time of day on the 24-hour dial (minutes capped at 10) | 0.720 | 304 |
| `clock_regularity` | length of the mean of the daily mean directions, over days with 5+ full plays; each day starts 12 hours from the person's centre | 0.725 | 304 |
| `finish_share` | share of plays whose `reason_end` is `trackdone` | 0.86 (13 Spotify people) | none |
| `skip_seconds` | median seconds played before a skip ended by `fwdbtn` or unknown | 0.99 (13 Spotify people) | none |
| `dial_explore` | explore (+) against stay (−), from new share, old friends, turnover | | 301 |
| `dial_regular` | regular listening time (+) against any hour (−), from the two clock traits | | 301 |

The clock traits do not depend on the time zone: a fixed offset turns every angle by the same amount and changes neither length. That is why they work from UTC.

Each usable estimate's `evidence` carries the person's own receipts: counts, dates, and the artists behind the number ("4% of your full plays went to artists you'd been listening to for under 4 weeks (6,151 of 138,402, Oct 2017 to Feb 2025), most of all A, B and C. In the last 12 months alone it was 2%. That's more than 2% of the 304 ListenBrainz listeners we measured.").

## Not ported, and why

- **Map spread** (how far apart your artists sit on the taste map). It needs the co-listening map built from the 304-listener bench (an artist-by-48-dimension table over ~33,000 artists, built from other people's listening). That table is not shipped here, so spread is skipped. Its place in dial 1 is covered by the other three traits (see below).
- **Clock hour and night share.** They need local time, and `Play` has no country or time zone. Night share was dropped in October anyway (unstable).

## Minimums

- The whole reader: at least **2,000 plays** over at least **90 days**. Below that every estimate comes back `usable=False` with the reason.
- Per trait (from the TS engine): `new_share` 300 full plays after the 12-week warm-up; `old_friends` two scored 8-week stretches, which takes about 8 months; `season_turnover` two season changes with a full top 10, about 9 months; `clock_spread` 500 full plays; `clock_regularity` 30 days with 5+ full plays; `finish_share` 300 plays with `reason_end`; `skip_seconds` 100 skips with real lengths. A dial needs all of its traits.

## The dials

The October factor check (MATH.md section 15) used six traits including map spread. Its loadings were saved (`bench/out/lb3.json`), but score weights were not, and spread can't be computed here. So `py/scripts/build_music_norms.py` refits on the five traits this reader has, on the same 301 people, with the same method: normal scores, Horn's parallel analysis (1,000 shuffles), principal components, varimax. The script first reproduces the published six-trait loadings from the TS engine's saved values to three decimals (eigenvalues 3.100, 1.863, 0.475), which checks the factor code.

Five-trait result: eigenvalues 2.568, 1.861, 0.273, 0.178, 0.120. Parallel analysis keeps 2 (95th percentile under shuffling 1.238, 1.125). Loadings:

| trait | explore | regular |
|---|---|---|
| new share | +0.91 | −0.07 |
| old friends | −0.94 | +0.01 |
| season turnover | +0.92 | +0.05 |
| clock spread | −0.03 | −0.96 |
| clock regularity | −0.05 | +0.96 |

Scores. Each trait is turned into its percentile on the norms table and then a normal score, clipped at 1/302. The dial is a weighted sum (regression weights R⁻¹L, keeping only the dial's own traits, scaled to SD 1 on the bench):

- `dial_explore = 0.360·z(new share) − 0.375·z(old friends) + 0.365·z(turnover)`
- `dial_regular = −0.528·z(clock spread) + 0.527·z(clock regularity)`

Dropping the other dial's near-zero weights changes nothing that matters: the scores correlate 0.9999 and 0.9997 with the full component scores. 0 is the typical listener, and about two in three people fall between −1 and +1. The dial's percentile comes from the bench's own dial scores.

## Data files

- `src/aas/data/music_norms.json`: for each trait and dial, n and the 1st to 99th percentiles (99 numbers). No per-person rows and no pseudonyms. The extremes (below the 1st and above the 99th) are deliberately left out.
- `src/aas/data/music_dials.json`: traits, weights, loadings, the correlation matrix and the parallel-analysis numbers.

Rebuild both with `cd py && uv run python scripts/build_music_norms.py`. This needs the local, gitignored bench (`bench/data/lbusers/plays`, 758 MB) and takes about a minute. Per-person values go to the system temp dir, never to the repo.

## Check against the TypeScript engine

The build script measures every bench person's training weeks in Python, exactly as `bench/lb3/person.ts` did (MusicBrainz-id artist keys, last 8 weeks held out), and compares with the TS engine's saved values in `bench/out/lb3/person/*.json`:

| trait | agree within 1e-4 | max abs diff |
|---|---|---|
| new share | 304/304 | 0 |
| old friends | 304/304 | 0 |
| season turnover | 301/301 | 0 |
| clock spread | 304/304 | 0 |
| clock regularity | 303/304 | 0.0039 |

The one difference is deliberate. One listener has 6 days whose plays cancel out exactly on the dial. Each such day's vector has length about 1e-16, so its "direction" is rounding noise, and JS and C `cos` disagree in the last bit. The TS engine kept those days unless the sum was exactly zero. Python leaves out days whose vector is shorter than 1e-9 per play.

## Caveats

- The reference population is ListenBrainz users, who are mostly enthusiasts who self-host or scrobble. ListenBrainz logs a listen once half the track or 4 minutes has played, while Spotify's full play starts at 30 seconds. So a Spotify history counts a few more partly heard tracks as plays than the reference does. Read percentiles as approximate, not to the point.
- The norms are measured on each reference person's whole history minus their last 8 weeks (median 4.1 years). A person with one year of history is compared with people measured over several. The traits are shares and averages, so this mostly adds noise.
- `finish_share` and `skip_seconds` have no reference population (ListenBrainz records neither), so they have no percentile.

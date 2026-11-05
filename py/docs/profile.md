# Profile: the evaluation core

`src/aas/profile.py`. Runs the readers on a person, checks each guess against the
person's own quiz answers, and scores the readers across many people. No UI on purpose:
the model gets finished and scored before anything is shown to anyone.

```
prof = profile.build(person, readers)          # Estimates + why a reader couldn't read
comp = profile.compare(prof, quiz.score(ans))  # one person: guess vs their own quiz
ev   = profile.evaluate([(prof, qres), ...])   # many people: is the reader any good?
print(profile.dump(prof, comp))                # bare text for debugging
```

## Rules

- An `Estimate` with `usable=False` never becomes a claim. It is stored only as a reason
  (`Profile.unread`), with its `evidence` as the reason text.
- A reader that says `can_read -> False`, or crashes in `can_read`/`read`, becomes a reason
  too. One broken reader can't take the profile down.
- Two readers on the same Big Five trait are kept side by side in `Profile.bigfive[trait]`.
  `profile` itself never averages them. The only combination is the `fused` reader
  (`aas.fusion.FusedReader`, see `fusion.md`): when it is in the readers list its estimate
  sits beside the single readers under the name `fused`, with `validity = None` because
  fusion has not yet been tested on people with several sources and a quiz.
- The MBTI to Big Five links (trait, sign and strength) live in one place,
  `fusion.MBTI_LINKS`; `profile.MBTI_TO_BIGFIVE` is derived from it.

## One person: `compare`

For every Big Five estimate whose trait the quiz scored:
predicted z, interval, quiz z and percentile, `inside` (quiz z within `[low, high]`,
`None` if the reader gave no interval), `same_side` (both above or both below average),
`error` (guess minus quiz).

MBTI estimates get a **loose** check only: I/E against E, N/S against O, T/F against A,
J/P against C (the McCrae and Costa 1989 overlaps, |r| about 0.4 to 0.7). The check is
"does the lean point to the same side of average as the quiz". It is a hint, not a test,
and `evaluate` ignores it.

## Many people: `evaluate`

Per reader and Big Five trait:

| field | meaning | what good looks like |
|---|---|---|
| `coverage` | share of people whose quiz z fell inside the stated interval | close to the stated level (0.80 by default). Lower = the reader is overconfident. Higher = intervals wider than needed. |
| `mean_width` | average interval width in z | narrower is better, but only once coverage is right |
| `mae` | mean absolute error, guess vs quiz, in z | below `mae_baseline` |
| `mae_baseline` | the same error if you guessed "average" (z = 0) for everyone | |
| `skill` | `1 - mae / mae_baseline` | above 0 means the reader beats knowing nothing |
| `r` | correlation between guess and quiz | the same number the slice write-ups report |
| `same_side` | share where guess and quiz were on the same side of average | above 0.5 |

`skipped` counts, per reader, the people it couldn't read at all or for some Big Five trait.

The baseline is z = 0 because quiz z is already relative to the ~600k-person norms, so
"average" is the honest know-nothing guess.

## Sample output

`dump` for one fake person (two readers on extraversion, one unusable MBTI axis, one
reader with no data, one that crashed):

```
sources: reddit.zip, spotify.zip
readers run: text-bigfive, music-bigfive, text-mbti, music
note: 2 files skipped (not text)
bigfive O (z)
  text-bigfive     O                  +1.000 [+0.60, +1.40]  validity r 0.30  (1,204 comments)
bigfive E (z)  several readers, shown separately (the fused reader, if run, is the combination)
  text-bigfive     E                  -0.500 [-1.20, +0.20]  validity r 0.30  (1,204 comments)
  music-bigfive    E                  +0.400 [-0.30, +1.10]  validity r 0.12  (1,204 comments)
mbti (value = P(first letter))
  text-mbti        I/E                +0.700 [+0.60, +0.80]  validity auc 0.73
  text-mbti        T/F                +0.780 [+0.70, +0.86]  validity auc 0.83  (1,204 comments)
measured
  music            skip rate          +0.310 no interval  validity untested  (9,810 plays)
unread
  text-mbti J/P: only 40 words about plans
  watch-reader: no YouTube history in these exports
  broken: failed while reading (RuntimeError: model file missing)
vs quiz
  text-bigfive     O guess +1.00 quiz +0.20 err +0.80 OUTSIDE
  text-bigfive     E guess -0.50 quiz -0.90 err +0.40 inside
  music-bigfive    E guess +0.40 quiz -0.90 err +1.30 OUTSIDE
  text-mbti        E loose via I/E: quiz -0.90 same side
  text-mbti        A loose via T/F: quiz +0.30 other side
  no reader for: C N
  1 of 3 Big Five guesses had the quiz score inside their interval. Loose MBTI checks: 1 of 2 on the same side of average.
```

`evaluate` on simulated people (tests `_simulate`). An honest reader, error sd 0.5 and
80% intervals of +/- 1.28 x 0.5:

```
4000 people
sim              E  n=4000 coverage 0.79 (stated 0.80, n=4000)  MAE 0.40 vs baseline 0.80 (skill +0.50)  r +0.89  same side 0.86
deaf: couldn't read 4000 of 4000
```

An overconfident one (error sd 1.0, intervals only +/- 0.3): coverage 0.25 against a
stated 0.80, and no skill over guessing average despite r 0.70. A correlation alone would
have looked fine; coverage and skill are what catch it.

```
sim              E  n=2000 coverage 0.25 (stated 0.80, n=2000)  MAE 0.79 vs baseline 0.79 (skill -0.00)  r +0.70  same side 0.74
```

## Stated interval level

`Estimate.interval_level` (default 0.80) says how often `[low, high]` claims to contain the
truth. `evaluate` reads the level from each estimate, raises if one reader and trait state
more than one level, and uses its `level` argument only when no estimate sets one. Music
estimates have no interval and set `interval_level = None`.

## Caveat on the scale

Readers trained on a dataset's own z-scores (each slice z-scores within its dataset) are
compared here against quiz z on the IPIP norms. If a reader's training population differs
from the IPIP norm population, its z is shifted; `mae` and `coverage` will show it, `r`
will not.

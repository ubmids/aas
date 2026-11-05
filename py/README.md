# AAS, Python

Show me your digital world and I'll tell you who you are: predict a person's
personality from their own data, one slice of life at a time, and score every
claim on people the model never saw.

```
uv sync
uv run pytest
uv run python scripts/run_play.py         # Steam
uv run python scripts/run_phone_body.py   # StudentLife + LifeSnaps
uv run python scripts/run_text.py         # MBTI forum posts (PANDORA once access lands)
uv run python scripts/take_quiz.py        # the 20-question quiz, scored on 597k-person norms
```

Raw and processed data live in `../data/` (gitignored, never redistributed).

- `src/aas/schema.py`: the contract. Every dataset becomes `persons` (labels) and `features` (behaviour), one row per person.
- `src/aas/evaluate.py`: the referee. Cross-validated r (or AUC), permutation p, bootstrap CI, Holm. Read r against `null_mean`, not 0.
- `src/aas/quiz.py`: IPIP-50 key, norms, short form, scoring.
- `src/aas/slices/`: one loader per dataset.
- `docs/`: one write-up per slice.

## Round 1 (1 Nov 2026)

| Slice | Data | N | Best honest result | Survives? |
|---|---|---|---|---|
| Text | MBTI forum posts, type words stripped | 8,675 | T/F AUC 0.83 to 0.88, I/E and N/S 0.70 to 0.78, J/P 0.64 to 0.70 | yes |
| Phone | StudentLife | 46 | E r = 0.24, O r = 0.32 | no (n too small) |
| Body | LifeSnaps (Fitbit) | 52 | E r = 0.24 | no (n too small) |
| Play | Steam profile counts | 149 | best r = 0.14 | no |
| Answer timing | IPIP response times | 20,000 | r 0.08 to 0.13 on all five | yes, but tiny |
| Quiz ruler | IPIP-50, 597k people | | alpha 0.80 to 0.90, 5 clean factors, short form 0.91 to 0.95 | |

The phone, body, play and timing rows were re-run on 5 Nov with the per-repeat r of the 1 Nov `evaluate.predict`
fix (before it: phone E 0.27, O 0.34; body E 0.26; play 0.15); no verdict changed.

What it says: language is where the signal is; the leak on the MBTI forum was
worth 0.06 to 0.16 AUC; sensors hint at extraversion twice but 50 people can't
settle it; coarse counts (Steam) carry nothing.

## v1 (5 Nov 2026)

v1 is the engine as it stands on 5 November 2026: readers that take a person's own exports and return estimates
with an interval and a stated accuracy, a fused reader that combines them, and a refusal with a reason wherever the
data cannot support a number. `docs/MATH.md` is the full account, and `docs/example_profile.md` shows what v1
returns for one invented person (`scripts/example_profile.py`, readers in `aas.example.default_readers`).

The default readers are these. Text Big Five is a bag of words trained on 2,467 student essays; it states the
accuracy measured on the kind of writing it is given and refuses openness unless most of the words are essay-like.
Text MBTI is trained on 8,675 forum users, and its accuracy off the forum is inferred from a test on essays. Music
measures how someone listens and makes no personality claim. The fused reader combines the two text readers with a
Bayesian model that needs no joint training data.

On writing the readers never saw, the numbers are small. Text Big Five on tweets reaches AUC 0.54 for C, E and A
and 0.57 for N (`docs/text_crossdomain.md`). In simulation with the measured validities, v1 fusion reaches r 0.196
O, 0.146 C, 0.172 E, 0.121 A and 0.148 N (mean 0.157), with its 80% intervals covering 80% on every trait
(`docs/MATH.md` section 5.6). Those intervals are almost as wide as the one that fits anyone, so most people see a z
near 0. No reader has yet been checked against real people's own questionnaires.

Several things failed. Phone, wearable and Steam data showed nothing that survives correction at 46 to 149 people. A local
1.7B model rating text copied the prompt's example. Claude Haiku 4.5 beat the bag of words on unseen writing (AUC
0.618 against 0.558) but runs off the device, so it is a benchmark only. A small on-device student distilled from
Haiku's ratings of 2,000 forum users failed its pre-registered rule narrowly (0.586 against a mark of 0.588 on
tweets and vlogs, 0.583 against 0.596 over three sets; `docs/distill.md`). It is saved and not a default. In
simulation it would raise fused E and N and lower C, which MATH.md section 5.6 reports as a sensitivity row.

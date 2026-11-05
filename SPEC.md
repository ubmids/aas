# AAS

Drop in your data downloads (Spotify, TikTok, later YouTube/Netflix). AAS reads them in the
browser, proves it knows you by predicting things you can check, then tells you what it found,
as shareable cards like Wrapped.

## Rules
- Nothing leaves the device. The engine runs in the browser; no upload, no account.
- Every insight has receipts: the plays/videos behind it.
- It earns trust before it talks: the reveal opens with predictions the user can verify.
- No sensitive inferences: no sexuality, health, pregnancy, religion, politics. Not modelled, not shown.
- "Accurate" means measured: every model is scored on held-out time against a baseline, on real people.

## Accuracy bench
`bench/` holds real public exports (gitignored, identifying fields stripped, never redistributed).
Each person is split by time: the engine sees everything before the cutoff and is scored on the
held-out weeks after it. A model only ships if it beats its baseline on most people.

Spotify tasks (held-out = last 8 weeks):
1. finish or skip: AUC of predicted skip probability on held-out plays. Baseline: the user's overall skip rate per artist.
2. next week's artists: hit@10 / NDCG@10. Baselines: all-time top artists, last-4-weeks top artists.
3. the game: one track the user really played in the held-out weeks against 4 decoys of matched popularity. Accuracy vs 20% chance.
4. when you listen: log-likelihood of held-out play hours vs a uniform clock and vs the all-time clock.
5. new or familiar: AUC of predicting whether a held-out play is an artist never heard before.
6. next falls: recall@20/50 of artists never played before that the user then plays properly 10+ times on 3+ days. Baselines: most popular unheard artists, nearest to all-time taste. Also scored on rolling windows inside training with a map and ranker that never saw the person.
7. selves: predict a held-out session's self from its surroundings only. Accuracy vs majority self, log-loss vs self frequencies.
8. eras: era-aware forecast of the held-out weeks' artists vs the decay model and all-time.

TikTok tasks (once link metadata is resolved): which watched videos you liked, binge sessions, the hours.

## Engine layout (TypeScript, runs in browser and Node)
- `engine/src/parse/` export parsers (Spotify extended, Spotify account data, TikTok JSON/TXT)
- `engine/src/model/` the models (taste profile, skip model, rhythm, drift, rabbit holes)
- `engine/src/insights/` turns model output into cards with receipts
- `bench/` evaluation harness: `npm run bench` prints per-task scores per person and the median

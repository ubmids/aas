// Export the round 3 "will it stick" model (first session, the columns a ListenBrainz history can
// fill) as a small JSON for the portfolio's interactive figure. Fitted on the same seeded rows as
// models.ts (up to 800 training-period rows per person) but on all 304 people at once, so it is a
// description of the model, not a score: every reported number comes from the folded fits.
// Besides the weights it writes each column's median over the fitting rows (the "typical" value the
// figure holds a column at when it is not a control) and the median rank score among rows whose
// list flag is on. Reads bench/out/lb3/{person,stick}/. Run: npx tsx bench/lb3/export_stick.ts <out.json>
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cachePath, readJsonGz, hash, log } from './common.js';
import { featuresOf, noSkipFeaturesOf } from '../../engine/src/model/stick.js';
import { fitLogReg, predict, auc } from '../../engine/src/model/logreg.js';
import { rng } from '../../engine/src/model/linalg.js';

const out = process.argv[2];
if (!out) throw new Error('usage: export_stick.ts <out.json>');
const PER_PERSON_FIT = 800;
const people = readdirSync(cachePath('person')).filter((f) => f.endsWith('.json')).map((f) => JSON.parse(readFileSync(cachePath('person', f), 'utf8'))).filter((p) => !p.skipped);
const names = featuresOf('session'), cols = noSkipFeaturesOf('session'), ix = cols.map((c) => names.indexOf(c));
const X: number[][] = [], y: number[] = [];
for (const p of people) {
  const f = readJsonGz<{ session: { train: { X: (number | null)[][]; y: number[] } } }>(cachePath('stick', `${p.id}.json.gz`));
  const T = f.session.train, r = rng(hash(p.id) + 17);
  const idx = T.y.map((_, i) => i).map((i) => [r(), i] as const).sort((a, b) => a[0] - b[0]).slice(0, PER_PERSON_FIT).map(([, i]) => i);
  for (const i of idx) { X.push(ix.map((c) => T.X[i][c] ?? 0)); y.push(T.y[i]); }
}
const m = fitLogReg(X, y, 1);
const med = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0; };
const median = Object.fromEntries(cols.map((c, j) => [c, med(X.map((x) => x[j]))]));
const onMedian = (flag: string, score: string) => med(X.filter((x) => x[cols.indexOf(flag)] === 1).map((x) => x[cols.indexOf(score)]));
const json = {
  model: 'will a new artist stick, after the first session, ListenBrainz columns (round 3)',
  people: people.length, rows: y.length, stuck: y.reduce((a, v) => a + v, 0),
  features: cols, w: m.w, mean: m.mean, sd: m.sd, median,
  rankScoreWhenListed: { prevRankScore: onMedian('prevLists', 'prevRankScore'), favRankScore: onMedian('favLists', 'favRankScore') },
  inSampleAuc: auc(X.map((x) => predict(m, x)), y),
};
writeFileSync(out, JSON.stringify(json, null, 1) + '\n');
log(`${people.length} people, ${y.length} rows, ${json.stuck} stuck, in-sample AUC ${json.inSampleAuc.toFixed(3)} -> ${out}`);

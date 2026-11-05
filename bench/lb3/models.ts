// Round 3, step 3: the models that learn across people, fitted per fold on the other four folds and
// scored on the people in the fold.
//   will it stick: logistic regression on the columns a ListenBrainz history can fill
//     (noSkipFeaturesOf), plus baselines (own base rate, popularity, first-week play counts), a
//     version without the ListenBrainz list features (those lists were computed by ListenBrainz from
//     its users' sessions, which include these people), and a shuffled-label null.
//   two-stage ranker: the round 2 ranker, trained on rolling windows of other people.
// Reads bench/out/lb3/{person,stick,rank2}/, writes bench/out/lb3/models.json.
// Run: npx tsx bench/lb3/models.ts
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cachePath, readJsonGz, FOLDS, hash, log } from './common.js';
import { featuresOf, noSkipFeaturesOf, averagePrecision, Horizon, STICK_GROUPS } from '../../engine/src/model/stick.js';
import { fitLogReg, predict, auc, LogReg } from '../../engine/src/model/logreg.js';
import { reliability } from '../../engine/src/model/stats.js';
import { recallAt, RANK2_FEATURES } from '../../engine/src/model/rank2.js';
import { rng } from '../../engine/src/model/linalg.js';

const PER_PERSON_FIT = 800, NEG = 200, PRIOR_K = 5;
const people = readdirSync(cachePath('person')).filter((f) => f.endsWith('.json')).map((f) => JSON.parse(readFileSync(cachePath('person', f), 'utf8'))).filter((p) => !p.skipped);
const fold = new Map<string, number>(people.map((p) => [p.id, p.fold]));
log(`${people.length} people`);

// ---------------------------------------------------------------- will it stick
type Part = { X: (number | null)[][]; y: number[] };
type StickFile = Record<Horizon, { train: Part; held: Part }> & { leak: { checked: number; failed: number }; prior: number };
const HORIZONS: Horizon[] = ['session', 'week'];
const LIST_COLS = STICK_GROUPS['ListenBrainz lists'];
const specsFor = (h: Horizon): { key: string; cols: string[]; shuffle?: boolean }[] => {
  const ns = noSkipFeaturesOf(h);
  const s = [
    { key: 'full', cols: ns },
    { key: 'noLists', cols: ns.filter((c) => !LIST_COLS.includes(c)) },
    { key: 'base', cols: ['baseLogit'] },
    { key: 'pop', cols: ['logPeople', 'logListed'] },
    { key: 'shuffled', cols: ns, shuffle: true },
  ];
  if (h === 'week') s.push({ key: 'wkCounts', cols: ['wkProper', 'wkDays'] });
  return s;
};
const num = (v: number | null) => (v === null ? 0 : v);

// fitting rows: a seeded subsample of each person's training-period rows
const fitRows = new Map<string, Record<Horizon, { X: number[][]; y: number[] }>>();
let leakChecked = 0, leakFailed = 0;
for (const p of people) {
  const f = readJsonGz<StickFile>(cachePath('stick', `${p.id}.json.gz`));
  leakChecked += f.leak.checked; leakFailed += f.leak.failed;
  const r = rng(hash(p.id) + 17), o = {} as Record<Horizon, { X: number[][]; y: number[] }>;
  for (const h of HORIZONS) {
    const idx = f[h].train.y.map((_, i) => i).map((i) => [r(), i] as const).sort((a, b) => a[0] - b[0]).slice(0, PER_PERSON_FIT).map(([, i]) => i);
    o[h] = { X: idx.map((i) => f[h].train.X[i].map(num)), y: idx.map((i) => f[h].train.y[i]) };
  }
  fitRows.set(p.id, o);
}
log(`stick rows loaded; leakage re-read ${leakChecked} rows, ${leakFailed} differ`);
if (leakFailed) throw new Error('features read past the horizon');

const metrics = (p: number[], y: number[]) => {
  const pos = y.reduce((a, v) => a + v, 0), ok = pos >= 3 && y.length - pos >= 3, rel = reliability(p, y);
  return { auc: ok ? auc(p, y) : NaN, ap: ok ? averagePrecision(p, y) : NaN, brier: rel.brier, ece: rel.ece, n: y.length, pos };
};
const stickOut: Record<string, Record<string, number>> = Object.fromEntries(people.map((p) => [p.id, {}]));
const pooled: Record<string, { p: number[]; y: number[] }> = {};
const weights: Record<string, Record<string, number>> = {};
for (const h of HORIZONS) {
  const names = featuresOf(h), tag = h === 'session' ? 'st_s' : 'st_w';
  for (const spec of specsFor(h)) {
    const cols = spec.cols.map((c) => names.indexOf(c));
    for (let f = 0; f < FOLDS; f++) {
      const X: number[][] = [], y: number[] = [];
      for (const [id, rows] of fitRows) if (fold.get(id) !== f) rows[h].X.forEach((x, i) => { X.push(cols.map((c) => x[c])); y.push(rows[h].y[i]); });
      if (spec.shuffle) { const r = rng(91 + f); for (let k = y.length - 1; k > 0; k--) { const j = Math.floor(r() * (k + 1)); [y[k], y[j]] = [y[j], y[k]]; } }
      const m: LogReg = fitLogReg(X, y, 1);
      if (spec.key === 'full' && f === 0) weights[h] = Object.fromEntries([['bias', m.w[0]], ...spec.cols.map((c, i) => [c, m.w[i + 1]])]);
      for (const p of people) {
        if (p.fold !== f) continue;
        const file = readJsonGz<StickFile>(cachePath('stick', `${p.id}.json.gz`));
        for (const [part, F] of [['roll', file[h].train], ['held', file[h].held]] as const) {
          const s = F.X.map((x) => predict(m, cols.map((c) => num(x[c]))));
          const mm = metrics(s, F.y), suf = spec.key === 'full' ? '' : `_${spec.key}`;
          for (const [k, v] of Object.entries(mm)) if ((k !== 'n' && k !== 'pos') || spec.key === 'full') stickOut[p.id][`${tag}_${part}_${k}${suf}`] = v;
          const pk = `${h}:${part}:${spec.key}`;
          (pooled[pk] ??= { p: [], y: [] }).p.push(...s); pooled[pk].y.push(...F.y);
        }
      }
    }
    const pr = pooled[`${h}:roll:${spec.key}`];
    log(`stick ${h} ${spec.key.padEnd(9)} pooled rolling AUC ${auc(pr.p, pr.y).toFixed(3)} AP ${averagePrecision(pr.p, pr.y).toFixed(3)}`);
  }
}
const pooledStick = Object.fromEntries(Object.entries(pooled).map(([k, s]) => { const rel = reliability(s.p, s.y); return [k, { auc: auc(s.p, s.y), ap: averagePrecision(s.p, s.y), brier: rel.brier, ece: rel.ece, n: s.y.length, pos: s.y.reduce((a, v) => a + v, 0) }]; }));

// ---------------------------------------------------------------- two-stage ranker
type Win = { start: number; held: boolean; loves: number; listed: number; popular20: number; popular50: number; X: (number | null)[][]; y: number[] };
const PROP = RANK2_FEATURES.indexOf('propList'), PROPX = RANK2_FEATURES.indexOf('propList_x_rrSum'), RR = RANK2_FEATURES.indexOf('rrSum');
const W8 = 8 * 7 * 86_400_000;
const winMeta = new Map<string, { start: number; held: boolean; loves: number; listed: number }[]>();
for (const p of people) winMeta.set(p.id, readJsonGz<Win[]>(cachePath('rank2', `${p.id}.json.gz`)).map(({ start, held, loves, listed }) => ({ start, held, loves, listed })));
const priorOf = (id: string) => { let n = 0, l = 0; for (const [k, ws] of winMeta) if (k !== id) for (const w of ws) if (!w.held) { n += w.loves; l += w.listed; } return n ? l / n : 0.4; };
const propOf = (id: string, start: number, prior: number) => {
  let n = 0, l = 0;
  for (const w of winMeta.get(id)!) if (!w.held && w.start + W8 <= start) { n += w.loves; l += w.listed; }
  return (l + PRIOR_K * prior) / (n + PRIOR_K);
};
const patch = (x: (number | null)[], prop: number) => { const z = x.map(num); z[PROP] = prop; z[PROPX] = prop * z[RR]; return z; };
const r2rows = new Map<string, { X: number[][]; y: number[] }>();
for (const p of people) {
  const ws = readJsonGz<Win[]>(cachePath('rank2', `${p.id}.json.gz`)), prior = priorOf(p.id), r = rng(hash(p.id) + 29), X: number[][] = [], y: number[] = [];
  for (const w of ws) {
    if (w.held) continue;
    const prop = propOf(p.id, w.start, prior), pos: number[] = [], neg: number[] = [];
    w.y.forEach((v, i) => (v ? pos : neg).push(i));
    for (const i of pos) { X.push(patch(w.X[i], prop)); y.push(1); }
    for (let k = 0; k < Math.min(NEG, neg.length); k++) { X.push(patch(w.X[neg[Math.floor(r() * neg.length)]], prop)); y.push(0); }
  }
  r2rows.set(p.id, { X, y });
}
const r2Out: Record<string, Record<string, number>> = Object.fromEntries(people.map((p) => [p.id, {}]));
const r2Pool: Record<string, number> = {};
let r2Weights: Record<string, number> = {};
for (let f = 0; f < FOLDS; f++) {
  const X: number[][] = [], y: number[] = [];
  for (const [id, o] of r2rows) if (fold.get(id) !== f) { X.push(...o.X); y.push(...o.y); }
  const m = fitLogReg(X, y, 1);
  if (f === 0) r2Weights = Object.fromEntries([['bias', m.w[0]], ...RANK2_FEATURES.map((c, i) => [c, m.w[i + 1]])]);
  for (const p of people) {
    if (p.fold !== f) continue;
    const ws = readJsonGz<Win[]>(cachePath('rank2', `${p.id}.json.gz`)), prior = priorOf(p.id);
    const acc: Record<string, Record<string, number>> = { roll: { n: 0 }, held: { n: 0 } };
    for (const w of ws) {
      if (!w.loves) continue;
      const prop = propOf(p.id, w.start, prior), xs = w.X.map((x) => patch(x, prop));
      const idx = xs.map((_, i) => i);
      const rank = (s: (i: number) => number) => [...idx].sort((a, b) => s(b) - s(a) || a - b);
      const lovesIdx = new Set(idx.filter((i) => w.y[i]).map(String));
      const byModel = rank((i) => predict(m, xs[i])).map(String), byLists = rank((i) => xs[i][4] * 1e3 + xs[i][6]).map(String);
      // recall counts every love in the window, also those stage one never proposed
      const rec = (rk: string[], K: number) => (lovesIdx.size ? (recallAt(rk, lovesIdx, K) * lovesIdx.size) / w.loves : 0);
      const a = acc[w.held ? 'held' : 'roll'];
      a.n += w.loves; a.cover = (a.cover ?? 0) + lovesIdx.size;
      for (const K of [20, 50]) {
        a[`model${K}`] = (a[`model${K}`] ?? 0) + rec(byModel, K) * w.loves;
        a[`lists${K}`] = (a[`lists${K}`] ?? 0) + rec(byLists, K) * w.loves;
        a[`popular${K}`] = (a[`popular${K}`] ?? 0) + (K === 20 ? w.popular20 : w.popular50) * w.loves;
      }
    }
    for (const [pre, a] of [['ts_roll', acc.roll], ['ts_fall', acc.held]] as const) {
      const o = r2Out[p.id];
      o[`${pre}N`] = a.n; o[`${pre}Cover`] = a.n ? a.cover / a.n : NaN;
      for (const k of ['model', 'lists', 'popular']) for (const K of [20, 50]) o[`${pre}${K}_${k}`] = a.n ? (a[`${k}${K}`] ?? 0) / a.n : NaN;
      for (const [k, v] of Object.entries(a)) r2Pool[`${pre}:${k}`] = (r2Pool[`${pre}:${k}`] ?? 0) + v;
    }
  }
  log(`ranker fold ${f} done`);
}
const P = (pre: string, k: string) => ((r2Pool[`${pre}:${k}`] ?? 0) / r2Pool[`${pre}:n`]).toFixed(3);
log(`ranker pooled rolling: cover ${P('ts_roll', 'cover')} r@50 model ${P('ts_roll', 'model50')} lists ${P('ts_roll', 'lists50')} popular ${P('ts_roll', 'popular50')}; held-out r@50 model ${P('ts_fall', 'model50')} popular ${P('ts_fall', 'popular50')}`);

writeFileSync(cachePath('models.json'), JSON.stringify({ at: new Date().toISOString(), stick: stickOut, pooledStick, stickWeights: weights, leak: { checked: leakChecked, failed: leakFailed }, rank2: r2Out, rank2Pool: r2Pool, rank2Weights: r2Weights }));
log('wrote bench/out/lb3/models.json');

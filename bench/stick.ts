// Bench for "will this new artist stick?" (engine/src/model/stick.ts). Every artist a person hears
// for the first time is a row; the label is whether it became a new love within 8 weeks. Two
// horizons: the end of the first session, and the end of the first 7 days (rows that had already
// met the definition by day 7 are left out of the week horizon, since they no longer need a guess).
//
// Leave one person out everywhere: the model, the taste map, the popularity counts and the prior for
// the person's own base rate all come from other people. Training rows are other people's new
// artists whose 8-week outcome ends inside their training period. Each person is scored on
//   rolling: every new artist in their own training period (outcome inside training),
//   held-out: every new artist first heard in the 8 weeks before the cutoff (outcome read in the
//             held-out weeks).
// Baselines are logistic regressions on one thing each: the person's base rate, the first-play skip,
// popularity, and (week horizon) the first week's play counts. Ablations drop one feature group.
// A shuffled-label run (labels permuted across all training rows) must score about 0.5. A second one
// permutes inside each training person, which keeps who sticks more overall and so shows how much
// of the score is differences between people.
// Writes bench/out/stick.json; `npm run report` folds it into REPORT.md.
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { loadPeople, Person } from './load.js';
import { Play, isSkip } from '../engine/src/parse/spotify.js';
import { listsOf, Lists } from '../engine/src/model/doors.js';
import { buildMap, TasteMap } from '../engine/src/model/map.js';
import { localClock } from '../engine/src/model/session.js';
import {
  newArtistRows, stickFeatures, horizonEnd, baseRateLogit, averagePrecision, featuresOf, noSkipFeaturesOf, STICK_GROUPS, Horizon, NewArtist, StickContext,
} from '../engine/src/model/stick.js';
import { fitLogReg, predict, auc } from '../engine/src/model/logreg.js';
import { fitGbt, predictGbt } from '../engine/src/model/gbt.js';
import { reliability, median } from '../engine/src/model/stats.js';
import { rng } from '../engine/src/model/linalg.js';

const QUICK = process.argv.includes('--quick'); // skip ablations and boosting
const DAY = 86_400_000, WEEK = 7 * DAY;
const log = (s: string) => process.stderr.write(s + '\n');
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '-');
const hash = (s: string) => [...s].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 7);

const LB_FILE = 'bench/data/lb/table.json';
const L: Lists | null = existsSync(LB_FILE) ? listsOf(JSON.parse(readFileSync(LB_FILE, 'utf8'))) : null;
if (!L) log('no ListenBrainz table: list features are zero');

const people = loadPeople();
const trainOf = (p: Person) => p.plays.slice(0, p.n);
let t0 = Date.now();
const maps = new Map<string, TasteMap>(people.map((p) => [p.name, buildMap(people.filter((q) => q !== p).map(trainOf))] as const));
log(`leave-one-out co-listening maps in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
const playedBy = new Map(people.map((p) => [p.name, new Set(trainOf(p).filter((x) => !isSkip(x)).map((x) => x.artist))] as const));
const othersPeople = (name: string) => {
  const m = new Map<string, number>();
  for (const [n, s] of playedBy) if (n !== name) for (const a of s) m.set(a, (m.get(a) ?? 0) + 1);
  return m;
};

// ---- rows per person
interface PRows { name: string; all: NewArtist[]; train: NewArtist[]; held: NewArtist[] }
const rowsOf: PRows[] = people.map((P) => {
  const all = newArtistRows(P.plays);
  const holdStart = P.cut - 8 * WEEK;
  return { name: P.name, all, train: all.filter((r) => r.t0 + 8 * WEEK <= P.cut), held: all.filter((r) => r.t0 >= holdStart && r.t0 < P.cut) };
});
const rateOf = (rs: NewArtist[]) => rs.reduce((s, r) => s + r.label, 0) / Math.max(1, rs.length);
const priorOf = (name: string) => rateOf(rowsOf.filter((r) => r.name !== name).flatMap((r) => r.train));

// ---- features per person and horizon (with the leakage check on a sample)
type Feat = { X: number[][]; y: number[]; rows: NewArtist[] };
const feats = new Map<string, Record<Horizon, { train: Feat; held: Feat }>>();
let leakChecked = 0, leakFailed = 0;
for (const P of people) {
  const R = rowsOf.find((r) => r.name === P.name)!;
  const clk = localClock(trainOf(P));
  const ctx: StickContext = { lists: L, map: maps.get(P.name)!, people: othersPeople(P.name), hour: (p: Play) => clk.hour(p) };
  const prior = priorOf(P.name), cache = new Map<number, Map<string, number>>();
  const r = rng(hash(P.name));
  const build = (rs: NewArtist[], h: Horizon): Feat => {
    const keep = h === 'week' ? rs.filter((x) => !x.fallenByWeek) : rs;
    const X = keep.map((x) => {
      const row = stickFeatures(P.plays, x.i, h, ctx, baseRateLogit(R.all, x.t0, prior), cache);
      if (r() < 0.01) { // the same row read from the history cut at the horizon must be identical
        const cut = P.plays.slice(0, horizonEnd(P.plays, x.i, h) + 1);
        const again = stickFeatures(cut, x.i, h, ctx, baseRateLogit(R.all, x.t0, prior));
        leakChecked++;
        if (again.some((v, k) => Math.abs(v - row[k]) > 1e-12)) leakFailed++;
      }
      return row;
    });
    return { X, y: keep.map((x) => x.label), rows: keep };
  };
  feats.set(P.name, { session: { train: build(R.train, 'session'), held: build(R.held, 'session') }, week: { train: build(R.train, 'week'), held: build(R.held, 'week') } });
  const f = feats.get(P.name)!;
  log(`${P.name.slice(0, 24).padEnd(24)} new artists: training ${f.session.train.y.length} (${f3(rateOf(R.train))} stick), held-out ${f.session.held.y.length} (${f.session.held.y.reduce((s, v) => s + v, 0)} stick); week rows ${f.week.train.y.length}`);
}
log(`leakage check: ${leakChecked} rows re-read from the history cut at the horizon, ${leakFailed} differ`);
if (leakFailed) throw new Error('features read past the horizon');

// ---- models
type Scorer = (x: number[]) => number;
function permute(a: number[], seed: number) { const r = rng(seed), b = [...a]; for (let k = b.length - 1; k > 0; k--) { const j = Math.floor(r() * (k + 1)); [b[k], b[j]] = [b[j], b[k]]; } return b; }
interface Spec { key: string; cols: string[] | 'all'; gbt?: boolean; shuffle?: 'all' | 'within' }
function fitFor(name: string, h: Horizon, spec: Spec): Scorer {
  const names = featuresOf(h), cols = spec.cols === 'all' ? names.map((_, k) => k) : spec.cols.map((c) => names.indexOf(c));
  const X: number[][] = [], y: number[] = [];
  for (const P of people) {
    if (P.name === name) continue;
    const f = feats.get(P.name)![h].train;
    let ys = f.y;
    if (spec.shuffle === 'within') ys = permute(f.y, hash(P.name) + 5);
    f.X.forEach((x, k) => { X.push(cols.map((c) => x[c])); y.push(ys[k]); });
  }
  if (spec.shuffle === 'all') y.splice(0, y.length, ...permute(y, hash(name) + 9));
  if (spec.gbt) { const m = fitGbt(X, y); return (x) => predictGbt(m, cols.map((c) => x[c])); }
  const m = fitLogReg(X, y, 1);
  return (x) => predict(m, cols.map((c) => x[c]));
}

const HORIZONS: Horizon[] = ['session', 'week'];
const specsFor = (h: Horizon): Spec[] => {
  const names = featuresOf(h);
  const s: Spec[] = [
    { key: 'full', cols: 'all' },
    { key: 'base', cols: ['baseLogit'] },
    { key: 'skip', cols: ['firstSkip'] },
    { key: 'pop', cols: ['logPeople', 'logListed'] },
    { key: 'shuffled', cols: 'all', shuffle: 'all' },
    { key: 'shuffledWithin', cols: 'all', shuffle: 'within' },
    { key: 'noSkip', cols: noSkipFeaturesOf(h) }, // only what a ListenBrainz history can fill (round 3)
  ];
  if (h === 'week') s.push({ key: 'wkCounts', cols: ['wkProper', 'wkDays'] });
  if (!QUICK) {
    s.push({ key: 'gbt', cols: 'all', gbt: true });
    for (const [g, cols] of Object.entries(STICK_GROUPS)) {
      const left = names.filter((n) => !cols.includes(n));
      if (left.length < names.length) s.push({ key: `drop:${g}`, cols: left });
    }
    for (const [g, cols] of Object.entries(STICK_GROUPS)) if (cols.every((c) => names.includes(c))) s.push({ key: `only:${g}`, cols });
  }
  return s;
};

interface Scored { p: number[]; y: number[]; rows: NewArtist[] }
const metrics = (s: Scored) => {
  const rel = reliability(s.p, s.y);
  const pos = s.y.reduce((a, v) => a + v, 0);
  const ok = pos >= 3 && s.y.length - pos >= 3;
  return { auc: ok ? auc(s.p, s.y) : NaN, ap: ok ? averagePrecision(s.p, s.y) : NaN, brier: rel.brier, ece: rel.ece, n: s.y.length, pos };
};

const out: Record<string, number>[] = people.map(() => ({}));
const pooled: Record<string, { p: number[]; y: number[] }> = {};
const examples: { name: string; r: NewArtist; p: number; x: number[] }[] = [];
// held-out session-horizon scores of the full model, for the prediction game (bench/game.ts)
const heldScores: Record<string, { p: number[]; y: number[]; logPeople: number[]; logListed: number[]; sessProper: number[]; sessPlays: number[] }> = {};
for (const h of HORIZONS) {
  const tag = h === 'session' ? 'st_s' : 'st_w';
  for (const spec of specsFor(h)) {
    t0 = Date.now();
    people.forEach((P, k) => {
      const f = feats.get(P.name)![h], score = fitFor(P.name, h, spec);
      for (const [part, F] of [['roll', f.train], ['held', f.held]] as const) {
        const s: Scored = { p: F.X.map(score), y: F.y, rows: F.rows };
        const m = metrics(s);
        for (const [mk, v] of Object.entries(m)) if (mk === 'n' || mk === 'pos' ? spec.key === 'full' : true) out[k][`${tag}_${part}_${mk}${spec.key === 'full' ? '' : `_${spec.key}`}`] = v;
        const pk = `${h}:${part}:${spec.key}`;
        (pooled[pk] ??= { p: [], y: [] }).p.push(...s.p); pooled[pk].y.push(...s.y);
        if (spec.key === 'full' && part === 'held' && h === 'session') {
          s.rows.forEach((r, j) => examples.push({ name: P.name, r, p: s.p[j], x: F.X[j] }));
          const F0 = featuresOf('session'), col = (c: string) => F.X.map((x) => x[F0.indexOf(c)]);
          heldScores[P.name] = { p: s.p, y: s.y, logPeople: col('logPeople'), logListed: col('logListed'), sessProper: col('sessProper'), sessPlays: col('sessPlays') };
        }
      }
    });
    const pr = pooled[`${h}:roll:${spec.key}`];
    log(`${h.padEnd(7)} ${spec.key.padEnd(34)} pooled rolling AUC ${f3(auc(pr.p, pr.y))} AP ${f3(averagePrecision(pr.p, pr.y))}  median per-person AUC ${f3(median(out.map((o) => o[`${tag}_roll_auc${spec.key === 'full' ? '' : `_${spec.key}`}`])))}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
}

// ---- print
const pooledRow = (h: Horizon, part: string, key: string) => {
  const s = pooled[`${h}:${part}:${key}`];
  if (!s) return null;
  const rel = reliability(s.p, s.y);
  return { auc: auc(s.p, s.y), ap: averagePrecision(s.p, s.y), brier: rel.brier, ece: rel.ece, n: s.y.length, pos: s.y.reduce((a, v) => a + v, 0) };
};
console.log(`\nWILL IT STICK over ${people.length} people`);
for (const h of HORIZONS) for (const part of ['roll', 'held']) {
  const full = pooledRow(h, part, 'full')!;
  console.log(`\n${h} horizon, ${part === 'roll' ? 'rolling (training period)' : 'held-out'}: ${full.n} new artists, ${full.pos} stuck (base rate ${f3(full.pos / full.n)})`);
  for (const spec of specsFor(h)) {
    const r = pooledRow(h, part, spec.key)!;
    const tag = h === 'session' ? 'st_s' : 'st_w', suf = spec.key === 'full' ? '' : `_${spec.key}`;
    console.log(`  ${spec.key.padEnd(34)} pooled AUC ${f3(r.auc)} AP ${f3(r.ap)} Brier ${f3(r.brier)} ECE ${f3(r.ece)} | median per person AUC ${f3(median(out.map((o) => o[`${tag}_${part}_auc${suf}`])))} AP ${f3(median(out.map((o) => o[`${tag}_${part}_ap${suf}`])))}`);
  }
}
const wAll = fitLogReg(people.flatMap((P) => feats.get(P.name)!.session.train.X), people.flatMap((P) => feats.get(P.name)!.session.train.y), 1).w;
const weights = Object.fromEntries([['bias', wAll[0]], ...featuresOf('session').map((f, i) => [f, wAll[i + 1]])]);
console.log(`\nsession-horizon weights fitted on everyone, standardised (reading only): ${Object.entries(weights).map(([k, v]) => `${k} ${(v as number).toFixed(2)}`).join(', ')}`);

// ---- examples written as checkable statements (held-out weeks, session horizon)
const fmtDate = (t: number) => new Date(t).toLocaleString('en-GB', { month: 'long', year: 'numeric', timeZone: 'UTC' });
const F = featuresOf('session'), at = (x: number[], f: string) => x[F.indexOf(f)];
function statement(e: (typeof examples)[number]) {
  const P = people.find((p) => p.name === e.name)!, plays = P.plays, first = plays[e.r.i];
  const end = horizonEnd(plays, e.r.i, 'session');
  const own = plays.slice(e.r.i, end + 1).filter((p) => p.artist === e.r.artist);
  const prev = e.r.i > 0 && plays[e.r.i - 1].artist !== e.r.artist && first.t - (plays[e.r.i - 1].t + plays[e.r.i - 1].ms) <= 30 * 60_000 ? plays[e.r.i - 1].artist : null;
  const how = at(e.x, 'picked') ? 'you picked it yourself' : at(e.x, 'cameOn') ? (prev ? `it came on after ${prev}` : 'it came on by itself') : 'it started some other way';
  const facts = [`${own.length} play${own.length > 1 ? 's' : ''} of it that session, ${own.filter((p) => !isSkip(p)).length} played past 30 seconds`, how];
  if (at(e.x, 'prevLists')) facts.push(`${prev} lists it on ListenBrainz`);
  else if (at(e.x, 'favLists')) facts.push('one of your recent favourites lists it on ListenBrainz');
  const outcome = e.r.label ? `It stuck: 10 proper plays on 3 days by day ${Math.ceil(e.r.daysTo10!)}.` : 'It did not stick within 8 weeks.';
  return `First heard ${e.r.artist} in ${fmtDate(e.r.t0)} (${facts.join('; ')}). After that session the model gave it ${Math.round(100 * e.p)}% to stick. ${outcome}`;
}
const byP = [...examples].sort((a, b) => b.p - a.p);
const sessPlays = (e: (typeof examples)[number]) => Math.round(Math.expm1(at(e.x, 'sessPlays')));
const pick = [
  ...byP.filter((e) => e.r.label && sessPlays(e) <= 4).slice(0, 2),
  ...byP.filter((e) => !e.r.label).slice(0, 1),
  ...byP.filter((e) => e.r.label).slice(-1),
  ...byP.filter((e) => !e.r.label && e.p < 0.02).slice(0, 1),
];
const exampleLines = pick.map(statement);
console.log('\nexamples (held-out weeks, after the first session):');
for (const s of exampleLines) console.log(`  - ${s}`);

mkdirSync('bench/out', { recursive: true });
const pooledOut: Record<string, ReturnType<typeof pooledRow>> = {};
for (const h of HORIZONS) for (const part of ['roll', 'held']) for (const spec of specsFor(h)) pooledOut[`${h}:${part}:${spec.key}`] = pooledRow(h, part, spec.key);
writeFileSync('bench/out/stick.json', JSON.stringify({
  at: new Date().toISOString(), quick: QUICK, names: people.map((p) => p.name), rows: out, pooled: pooledOut,
  specs: Object.fromEntries(HORIZONS.map((h) => [h, specsFor(h).map((s) => s.key)])), weights, leak: { checked: leakChecked, failed: leakFailed }, examples: exampleLines,
}));
writeFileSync('bench/out/stick_held.json', JSON.stringify({ at: new Date().toISOString(), people: heldScores }));
log('wrote bench/out/stick.json');

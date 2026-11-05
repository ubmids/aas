// Bench for the two-stage new-loves ranker (engine/src/model/rank2.ts). Same windows as task 6 of
// run.ts: rolling 8-week windows inside each person's training period, plus the held-out 8 weeks.
// Leave one person out everywhere: the ranker is fitted on other people's windows, the taste map
// and the popularity counts come from other people's training weeks. Recall counts every new love
// in the window, so loves that never became candidates are misses.
// Baselines: popular (unheard artists by how many other people played them, then plays) and
// lists (ListenBrainz evidence alone: favourite-weighted list scores, ties by popularity).
// Writes bench/out/rank2.json; `npm run report` folds it into REPORT.md.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { loadPeople, DAY } from './load.js';
import { Play } from '../engine/src/parse/spotify.js';
import { isSkip } from '../engine/src/parse/spotify.js';
import { buildLbMap, LbTable } from '../engine/src/model/lbmap.js';
import { knownBefore, findFalls, FALL_WEEKS } from '../engine/src/model/falls.js';
import { favourites, candidates, recallAt, RANK2_FEATURES, ListIndex, Candidate } from '../engine/src/model/rank2.js';
import { fitLogReg, predict } from '../engine/src/model/logreg.js';
import { listenCounts } from '../engine/src/model/taste.js';
import { rng } from '../engine/src/model/linalg.js';

const WEEK = 7 * DAY, NEG = 200, PRIOR_K = 5;
const log = (s: string) => process.stderr.write(s + '\n');
const hash = (s: string) => [...s].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 7);
const LB: LbTable = JSON.parse(readFileSync('bench/data/lb/table.json', 'utf8'));

// list lookups by name
const listMap = new Map<string, Map<string, [number, number]>>();
for (const [k, list] of Object.entries(LB.lists)) {
  if (!list.length) continue;
  const top = list[0][1] || 1, m = new Map<string, [number, number]>();
  list.forEach(([n, s], r) => { const a = LB.nodes[n]; if (!m.has(a)) m.set(a, [r + 1, s / top]); });
  listMap.set(LB.nodes[Number(k)], m);
}
const lists: ListIndex = { get: (a) => listMap.get(a) };

const people = loadPeople();
const trainOf = (p: (typeof people)[number]) => p.plays.slice(0, p.n);
function seedsOf(ps: typeof people) {
  const out = new Set<string>();
  for (const p of ps) { const c = new Map<string, number>(); for (const x of trainOf(p)) if (!isSkip(x)) c.set(x.artist, (c.get(x.artist) ?? 0) + 1); for (const [a, v] of c) if (v >= 3) out.add(a); }
  return out;
}
const popPlays = new Map(people.map((p) => [p.name, listenCounts(trainOf(p))] as const));
function othersPop(name: string) {
  const ppl = new Map<string, number>(), plays = new Map<string, number>();
  for (const [n, c] of popPlays) if (n !== name) for (const [a, v] of c) { ppl.set(a, (ppl.get(a) ?? 0) + 1); plays.set(a, (plays.get(a) ?? 0) + v); }
  return { people: ppl, plays };
}

let t0 = Date.now();
const maps = new Map(people.map((p) => [p.name, buildLbMap(LB, seedsOf(people.filter((q) => q !== p)))] as const));
log(`leave-one-out ListenBrainz maps in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

// ---- windows: per person, rolling windows (oldest first) and the held-out one
interface Win { start: number; known: Set<string>; loves: Set<string>; favs: ReturnType<typeof favourites>; held: boolean; listed: number }
const wins = new Map<string, Win[]>();
for (const P of people) {
  const train = trainOf(P), first = train[0].t, ws: Win[] = [];
  for (let k = 1; k <= 12; k++) {
    const start = P.cut - k * FALL_WEEKS * WEEK;
    if (start - first < 26 * WEEK) break;
    const known = knownBefore(train, start);
    ws.push({ start, known, loves: new Set(findFalls(train, start, known)), favs: favourites(train, start), held: false, listed: 0 });
  }
  const known = knownBefore(P.plays, P.cut);
  ws.push({ start: P.cut, known, loves: new Set(findFalls(P.plays, P.cut, known)), favs: favourites(train, P.cut), held: true, listed: 0 });
  ws.sort((a, b) => a.start - b.start);
  // how many of each window's loves a favourite listed (read at that window's start)
  for (const w of ws) for (const a of w.loves) if (w.favs.some((f) => lists.get(f.artist)?.has(a))) w.listed++;
  wins.set(P.name, ws);
}
/** The person's share of earlier loves that a favourite listed, shrunk to `prior`; only windows that ended by `start`. */
const propOf = (ws: Win[], start: number, prior: number) => {
  let n = 0, l = 0;
  for (const w of ws) if (!w.held && w.start + FALL_WEEKS * WEEK <= start) { n += w.loves.size; l += w.listed; }
  return (l + PRIOR_K * prior) / (n + PRIOR_K);
};
const priorOf = (name: string) => {
  let n = 0, l = 0;
  for (const [nm, ws] of wins) if (nm !== name) for (const w of ws) if (!w.held) { n += w.loves.size; l += w.listed; }
  return n ? l / n : 0.4;
};

// ---- candidates and features per window (each person's own map and popularity, leave one out)
const cands = new Map<string, Candidate[][]>();
for (const P of people) {
  const m = maps.get(P.name)!, pop = othersPop(P.name), prior = priorOf(P.name), ws = wins.get(P.name)!;
  cands.set(P.name, ws.map((w) => candidates(w.favs, w.known, lists, m, pop, propOf(ws, w.start, prior))));
  log(`${P.name.slice(0, 24).padEnd(24)} ${ws.length} windows, median ${[...cands.get(P.name)!.map((c) => c.length)].sort((a, b) => a - b)[ws.length >> 1]} candidates`);
}

// ---- ranker rows from rolling windows only
const rowsOf = new Map<string, { X: number[][]; y: number[] }>();
for (const P of people) {
  const r = rng(hash(P.name)), X: number[][] = [], y: number[] = [];
  wins.get(P.name)!.forEach((w, i) => {
    if (w.held) return;
    const cs = cands.get(P.name)![i];
    const pos = cs.filter((c) => w.loves.has(c.artist)), neg = cs.filter((c) => !w.loves.has(c.artist));
    for (const c of pos) { X.push(c.x); y.push(1); }
    for (let j = 0; j < Math.min(NEG, neg.length); j++) { X.push(neg[Math.floor(r() * neg.length)].x); y.push(0); }
  });
  rowsOf.set(P.name, { X, y });
}

// ---- score
const out: Record<string, number>[] = [], names: string[] = [];
const pooled: Record<string, number> = {};
const all = [...rowsOf.values()];
const wAll = fitLogReg(all.flatMap((o) => o.X), all.flatMap((o) => o.y), 1).w;
let candCover = 0, loveN = 0;
for (const P of people) {
  const others = people.filter((q) => q !== P).map((q) => rowsOf.get(q.name)!);
  const model = fitLogReg(others.flatMap((o) => o.X), others.flatMap((o) => o.y), 1);
  const pop = othersPop(P.name), ws = wins.get(P.name)!;
  const popRank = [...pop.people.keys()].sort((a, b) => pop.people.get(b)! - pop.people.get(a)! || pop.plays.get(b)! - pop.plays.get(a)! || (a < b ? -1 : 1));
  const row: Record<string, number> = {};
  const acc = { roll: { n: 0 } as Record<string, number>, held: { n: 0 } as Record<string, number> };
  ws.forEach((w, i) => {
    if (!w.loves.size) return;
    const cs = cands.get(P.name)![i];
    const byScore = (f: (c: Candidate) => number) => [...cs].map((c) => [c.artist, f(c)] as const).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map(([a]) => a);
    const ranked: Record<string, string[]> = {
      model: byScore((c) => predict(model, c.x)),
      lists: byScore((c) => c.x[4] * 1e3 + c.x[6]), // list evidence alone, ties by popularity
      popular: popRank.filter((a) => !w.known.has(a)).slice(0, 50),
    };
    const a = w.held ? acc.held : acc.roll;
    a.n += w.loves.size;
    for (const [k, rk] of Object.entries(ranked)) for (const K of [20, 50]) a[`${k}${K}`] = (a[`${k}${K}`] ?? 0) + recallAt(rk, w.loves, K) * w.loves.size;
    a.cover = (a.cover ?? 0) + cs.filter((c) => w.loves.has(c.artist)).length;
    if (!w.held) { candCover += cs.filter((c) => w.loves.has(c.artist)).length; loveN += w.loves.size; }
  });
  for (const [pre, a] of [['ts_roll', acc.roll], ['ts_fall', acc.held]] as const) {
    row[`${pre}N`] = a.n; row[`${pre}Cover`] = a.n ? a.cover / a.n : NaN;
    for (const k of ['model', 'lists', 'popular']) for (const K of [20, 50]) row[`${pre}${K}_${k}`] = a.n ? (a[`${k}${K}`] ?? 0) / a.n : NaN;
    for (const [k, v] of Object.entries(a)) pooled[`${pre}:${k}`] = (pooled[`${pre}:${k}`] ?? 0) + v;
  }
  out.push(row); names.push(P.name);
  log(`${P.name.slice(0, 24).padEnd(24)} rolling ${row.ts_rollN} loves (cover ${row.ts_rollCover.toFixed(2)}) r@50 model ${row.ts_roll50_model.toFixed(3)} lists ${row.ts_roll50_lists.toFixed(3)} popular ${row.ts_roll50_popular.toFixed(3)} | held-out ${row.ts_fallN} loves r@50 model ${(row.ts_fall50_model ?? NaN).toFixed(3)} popular ${(row.ts_fall50_popular ?? NaN).toFixed(3)}`);
}
const pr = (pre: string, k: string) => (pooled[`${pre}:${k}`] ?? 0) / pooled[`${pre}:n`];
console.log(`\nTWO-STAGE RANKER over ${people.length} people`);
console.log(`candidates cover ${(candCover / loveN).toFixed(3)} of rolling-window loves (${loveN})`);
for (const pre of ['ts_roll', 'ts_fall']) console.log(`${pre === 'ts_roll' ? 'rolling ' : 'held-out'} pooled (${pooled[`${pre}:n`]} loves): ` + [20, 50].map((K) => `recall@${K} model ${pr(pre, `model${K}`).toFixed(3)} lists ${pr(pre, `lists${K}`).toFixed(3)} popular ${pr(pre, `popular${K}`).toFixed(3)}`).join(' | '));
console.log(`weights fitted on everyone (reading only): bias ${wAll[0].toFixed(2)} ` + RANK2_FEATURES.map((f, i) => `${f} ${wAll[i + 1].toFixed(2)}`).join(', '));
mkdirSync('bench/out', { recursive: true });
writeFileSync('bench/out/rank2.json', JSON.stringify({ at: new Date().toISOString(), names, rows: out, pooled, weights: Object.fromEntries([['bias', wAll[0]], ...RANK2_FEATURES.map((f, i) => [f, wAll[i + 1]])]) }));
log('wrote bench/out/rank2.json');

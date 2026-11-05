// How many selves? BIC alone nearly always picks the cap of 7. This compares a stronger BIC penalty
// and a share floor, chosen per person on validation (the last 8 weeks of training, never the
// held-out weeks), and re-scores task 7: predict a held-out session's self from its surroundings.
// `npx tsx bench/selves_k.ts`
import { loadPeople, HOLDOUT_WEEKS, DAY, Person } from './load.js';
import { buildMap } from '../engine/src/model/map.js';
import { sessionRows, fitSelves, contextEncoder, SessionRow, SelvesOptions, Selves } from '../engine/src/model/selves.js';
import { fitSoftmax, probs } from '../engine/src/model/softmax.js';
import { logLik, Gmm } from '../engine/src/model/gmm.js';

const WEEK = 7 * DAY;
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '  -  ');
const median = (a: number[]) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const people = loadPeople();
const map = buildMap(people.map((p) => p.plays.slice(0, p.n)));

const CONFIGS: [string, SelvesOptions][] = [
  ['bic', {}],
  ...[2, 4, 8, 16].flatMap((pen) => [[`pen${pen}`, { penalty: pen }], [`pen${pen}+floor`, { penalty: pen, floor: 0.05 }]] as [string, SelvesOptions][]),
  ['bic+floor', { floor: 0.05 }],
];

/** Task 7 on (fit -> score) rows: accuracy, majority, log-loss, prior log-loss, number of selves. */
function task7(selves: Selves, fitRows: SessionRow[], scoreRows: SessionRow[], trainPlays: Person['plays']) {
  const ctx = contextEncoder(trainPlays), live = selves.live, idx = new Map(live.map((c, i) => [c, i]));
  const fr = fitRows.slice(-8000), ytr = fr.map((x) => idx.get(selves.label(x))!), yte = scoreRows.map((x) => idx.get(selves.label(x))!);
  const freq = live.map((_, i) => (ytr.filter((y) => y === i).length + 1) / (ytr.length + live.length));
  const maj = freq.indexOf(Math.max(...freq));
  const sm = fitSoftmax(fr.map((x) => ctx.encode(x.s)), ytr, live.length);
  const pr = scoreRows.map((x) => probs(sm, ctx.encode(x.s)));
  const n = yte.length || NaN;
  return {
    k: live.length,
    acc: pr.filter((p, i) => p.indexOf(Math.max(...p)) === yte[i]).length / n,
    maj: yte.filter((y) => y === maj).length / n,
    nll: -pr.reduce((s, p, i) => s + Math.log(Math.max(1e-9, p[yte[i]])), 0) / n,
    prior: -yte.reduce((s, y) => s + Math.log(freq[y]), 0) / n,
  };
}

type Res = ReturnType<typeof task7>;
const heldOut = new Map<string, Res[]>(CONFIGS.map(([k]) => [k, []]));
const chosen: Res[] = [], chosenLL: Res[] = [], picks: string[] = [], picksLL: string[] = [];
for (const P of people) {
  const train = P.plays.slice(0, P.n), test = P.plays.slice(P.n), vcut = P.cut - HOLDOUT_WEEKS * WEEK;
  const rowsTrain = sessionRows(train, map), rowsTest = sessionRows(test, map);
  const rowsFit = rowsTrain.filter((r) => r.s.start < vcut), rowsVal = rowsTrain.filter((r) => r.s.start >= vcut);
  const fitPlays = train.filter((p) => p.t < vcut);
  // validation: per config, context gain (prior - nll) and the mixture's own likelihood of later sessions
  let best = { key: 'bic', gain: -Infinity }, bestLL = { key: 'bic', ll: -Infinity };
  const val: string[] = [];
  const cacheFit = new Map<number, Gmm>(), cacheTrain = new Map<number, Gmm>();
  if (rowsFit.length > 200 && rowsVal.length > 30) for (const [key, o] of CONFIGS) {
    const s = fitSelves(rowsFit, { ...o, cache: cacheFit }), v = task7(s, rowsFit, rowsVal, fitPlays);
    const ll = logLik(s.g, rowsVal.map(s.vector)) / rowsVal.length;
    const ok = v.k >= 3 && v.k <= 6;
    if (ok && v.prior - v.nll > best.gain + 1e-9) best = { key, gain: v.prior - v.nll };
    if (ok && ll > bestLL.ll + 1e-9) bestLL = { key, ll };
    val.push(`${key}:k${v.k}`);
  }
  const results = new Map<string, Res>();
  for (const [key, o] of CONFIGS) { const r = task7(fitSelves(rowsTrain, { ...o, cache: cacheTrain }), rowsTrain, rowsTest, train); results.set(key, r); heldOut.get(key)!.push(r); }
  chosen.push(results.get(best.key)!); picks.push(best.key);
  chosenLL.push(results.get(bestLL.key)!); picksLL.push(bestLL.key);
  const b = results.get('bic')!, c = results.get(best.key)!;
  console.log(`${P.name.slice(0, 24).padEnd(24)} bic k${b.k} acc ${f3(b.acc)}/${f3(b.maj)} | chosen ${best.key} k${c.k} acc ${f3(c.acc)}/${f3(c.maj)} gain ${f3(c.prior - c.nll)} vs ${f3(b.prior - b.nll)} | by LL ${bestLL.key}`);
}
const line = (label: string, rs: Res[]) => {
  const w = rs.filter((r) => r.acc > r.maj).length, wn = rs.filter((r) => r.nll < r.prior).length;
  return `  ${label.padEnd(22)} selves ${median(rs.map((r) => r.k))} | acc ${f3(median(rs.map((r) => r.acc)))} vs majority ${f3(median(rs.map((r) => r.maj)))} (wins ${w}/${rs.length}) lift ${f3(median(rs.map((r) => r.acc - r.maj)))} | log-loss ${f3(median(rs.map((r) => r.nll)))} vs prior ${f3(median(rs.map((r) => r.prior)))} (wins ${wn}/${rs.length}) gain ${f3(median(rs.map((r) => r.prior - r.nll)))} nats`;
};
console.log(`\nTASK 7 on held-out weeks, median over ${people.length} people`);
for (const [key] of CONFIGS) console.log(line(key, heldOut.get(key)!));
console.log(line('chosen on val (gain)', chosen) + `\n    picks: ${picks.join(' ')}`);
console.log(line('chosen on val (LL)', chosenLL) + `\n    picks: ${picksLL.join(' ')}`);

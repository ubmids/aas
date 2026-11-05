// The prediction game, as pre-registered in docs/GAME.md. Three games, each scored per person on
// held-out weeks against a strong per-person baseline:
//   1 call 10 skips at the start of a block of 10 plays (13 Spotify people)
//   2 of three new artists (one stuck, two did not), pick the one that sticks (13 Spotify, 304 ListenBrainz)
//   3 name next week's top 5 artists (13 Spotify, 304 ListenBrainz)
// Needs bench/out/stick_held.json (written by `npm run stick`) and the round 3 cache under
// bench/out/lb3 (`npm run lb3`). Writes bench/out/game.json and bench/out/game.md.
// Run: npm run game   (--only=1,3 to play some games; the summary needs all three)
import { readdirSync, readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { loadPeople, Person } from './load.js';
import { Play, isSkip } from '../engine/src/parse/spotify.js';
import { localClock } from '../engine/src/model/session.js';
import { fitLogReg, predict } from '../engine/src/model/logreg.js';
import { rng } from '../engine/src/model/linalg.js';
import { bootstrap, signTest, mean } from '../engine/src/model/stats.js';
import { featuresOf, noSkipFeaturesOf } from '../engine/src/model/stick.js';
import {
  blocks, frozenSkipFeatures, hitsAt, hitsTopK, bestThreshold, tripleAccuracy, weekTop, topByCount, topByDecay, precisionAt, wilson, holm, ARTIST_RATE, BLOCK,
} from '../engine/src/model/game.js';
import { cachePath, readJsonGz, loadPerson, FOLDS, hash as lbHash } from './lb3/common.js';

const DAY = 86_400_000, WEEK = 7 * DAY, HOLD = 8, MAX_TRAIN = 60_000;
const log = (s: string) => process.stderr.write(`${new Date().toISOString().slice(11, 19)} ${s}\n`);
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '-');
const f2 = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '-');
const pf = (p: number) => (p < 0.001 ? '<0.001' : p.toFixed(3));
const ONLY = (process.argv.find((a) => a.startsWith('--only='))?.slice(7) ?? '1,2,3').split(',').map(Number);
const quantile = (a: number[], q: number) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); if (!s.length) return NaN; const i = q * (s.length - 1), lo = Math.floor(i); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo); };

/** One person's line in a game: model and baseline scores (higher is better), plus extras. */
interface Row { id: string; model: number; base: number; n: number; extra?: Record<string, number> }
interface Result { key: string; game: number; data: string; label: string; unit: string; baseName: string; rows: Row[]; primary: boolean; eligible: number; total: number }
const results: Result[] = [];

// ======================================================================== game 1: call 10 skips
let spotify: Person[] | null = null;
const spotifyPeople = () => (spotify ??= loadPeople());
if (ONLY.includes(1)) {
  const rows: Row[] = [], topk: Row[] = [];
  const people = spotifyPeople();
  let total = 0;
  for (const P of people) {
    total++;
    const { plays, n, cut } = P, train = plays.slice(0, n);
    const bs = blocks(plays), X = frozenSkipFeatures(plays, bs, localClock(train).hour), y: number[] = plays.map((p) => (isSkip(p) ? 1 : 0));
    const vcut = cut - HOLD * WEEK, nV = plays.findIndex((p) => p.t >= vcut);
    const fit = (a: number, b: number) => { const lo = Math.max(0, b - MAX_TRAIN); return fitLogReg(X.slice(lo, b), y.slice(lo, b), 1); };
    const full = (from: number, to: number) => bs.filter((b) => b.len === BLOCK && b.start >= from && b.start + BLOCK <= to);
    // thresholds on the validation slice, model fitted before it
    const mV = fit(0, nV), vRounds = full(nV, n);
    const tModel = bestThreshold(vRounds.map((b) => ({ s: X.slice(b.start, b.start + BLOCK).map((x) => predict(mV, x)), y: y.slice(b.start, b.start + BLOCK) })));
    const tArtist = bestThreshold(vRounds.map((b) => ({ s: X.slice(b.start, b.start + BLOCK).map((x) => x[ARTIST_RATE]), y: y.slice(b.start, b.start + BLOCK) })));
    const m = fit(0, n), rate = y.slice(0, n).reduce((s, v) => s + v, 0) / n, constCall = rate >= 0.5 ? 1 : 0;
    const rounds = full(n, plays.length).map((b) => {
      const yy = y.slice(b.start, b.start + BLOCK), s = X.slice(b.start, b.start + BLOCK).map((x) => predict(m, x)), a = X.slice(b.start, b.start + BLOCK).map((x) => x[ARTIST_RATE]);
      return {
        session: b.session, skips: yy.reduce((q, v) => q + v, 0),
        model: hitsAt(s, yy, tModel), artist: hitsAt(a, yy, tArtist), constant: yy.filter((v) => v === constCall).length,
        modelK: hitsTopK(s, yy), artistK: hitsTopK(a, yy),
      };
    });
    log(`game 1 ${P.name.slice(0, 24).padEnd(24)} ${rounds.length} rounds, thresholds model ${tModel} artist ${tArtist}, training skip rate ${f3(rate)}`);
    if (rounds.length < 10) continue;
    const mm = mean(rounds.map((r) => r.model)), ma = mean(rounds.map((r) => r.artist)), mc = mean(rounds.map((r) => r.constant));
    const useArtist = ma >= mc, key = useArtist ? 'artist' : 'constant';
    // per-person interval for the difference: bootstrap over sessions
    const bySession = [...new Set(rounds.map((r) => r.session))].map((s) => rounds.filter((r) => r.session === s));
    const ci = bootstrap(bySession, (ss) => { const rr = ss.flat(); return mean(rr.map((r) => r.model - (r as Record<string, number>)[key])); }, 2000, lbHash(P.name));
    rows.push({ id: P.name, model: mm, base: Math.max(ma, mc), n: rounds.length, extra: { artist: ma, constant: mc, baseIsArtist: useArtist ? 1 : 0, skipRate: mean(rounds.map((r) => r.skips)) / BLOCK, ciLo: ci[0], ciHi: ci[1], tModel, tArtist } });
    topk.push({ id: P.name, model: mean(rounds.map((r) => r.modelK)), base: mean(rounds.map((r) => r.artistK)), n: rounds.length });
  }
  results.push({ key: 'g1', game: 1, data: 'Spotify', label: 'call 10 skips: model vs the better of own skip rate and most-skipped artists', unit: 'hits out of 10', baseName: 'best baseline', rows, primary: true, eligible: rows.length, total });
  results.push({ key: 'g1k', game: 1, data: 'Spotify', label: 'call 10 skips, told how many: model top k vs artist skip rate top k', unit: 'hits out of 10', baseName: 'artist top k', rows: topk, primary: false, eligible: topk.length, total });
}

// ======================================================================== game 2: which of 3 new artists sticks
type Held = { id: string; p: number[]; y: number[]; logPeople: number[]; logListed: number[]; sessProper: number[]; sessPlays: number[] };
function tripleRows(people: Held[]) {
  const vsPop: Row[] = [], vsChance: Row[] = [], vsFirst: Row[] = [];
  for (const q of people) {
    const pos = q.y.map((v, i) => (v ? i : -1)).filter((i) => i >= 0), neg = q.y.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
    if (pos.length < 2 || neg.length < 2) continue;
    const acc = (s: number[]) => tripleAccuracy(pos.map((i) => s[i]), neg.map((i) => s[i]));
    const pop = q.logPeople.map((v, i) => v + 1e-3 * q.logListed[i]);
    const am = acc(q.p), ap = acc(pop);
    vsPop.push({ id: q.id, model: am, base: ap, n: pos.length * ((neg.length * (neg.length - 1)) / 2), extra: { stuck: pos.length, notStuck: neg.length } });
    vsChance.push({ id: q.id, model: am, base: 1 / 3, n: vsPop[vsPop.length - 1].n });
    // exploratory, added after the pre-registered run: the artist played most in its first session
    vsFirst.push({ id: q.id, model: am, base: acc(q.sessProper.map((v, i) => v + 1e-3 * q.sessPlays[i])), n: vsPop[vsPop.length - 1].n });
  }
  return { vsPop, vsChance, vsFirst };
}
if (ONLY.includes(2)) {
  // Spotify: the full stick model's held-out scores from bench/stick.ts (leave one person out)
  if (!existsSync('bench/out/stick_held.json')) throw new Error('run `npm run stick` first (writes bench/out/stick_held.json)');
  const sh = JSON.parse(readFileSync('bench/out/stick_held.json', 'utf8')).people as Record<string, Omit<Held, 'id'>>;
  const sp = tripleRows(Object.entries(sh).map(([id, v]) => ({ id, ...v })));
  results.push({ key: 'g2s', game: 2, data: 'Spotify', label: 'which of 3 new artists sticks: model vs popularity', unit: 'accuracy', baseName: 'popularity', rows: sp.vsPop, primary: true, eligible: sp.vsPop.length, total: Object.keys(sh).length });
  results.push({ key: 'g2sc', game: 2, data: 'Spotify', label: 'which of 3 new artists sticks: model vs chance', unit: 'accuracy', baseName: 'chance', rows: sp.vsChance, primary: false, eligible: sp.vsChance.length, total: Object.keys(sh).length });
  results.push({ key: 'g2sf', game: 2, data: 'Spotify', label: 'which of 3 new artists sticks: model vs most played in its first session (exploratory, added after the run)', unit: 'accuracy', baseName: 'first session', rows: sp.vsFirst, primary: false, eligible: sp.vsFirst.length, total: Object.keys(sh).length });

  // ListenBrainz: the full model of bench/lb3/models.ts (columns a history without skips can fill), fitted on the other four folds
  const lbPeople = readdirSync(cachePath('person')).filter((f) => f.endsWith('.json')).map((f) => JSON.parse(readFileSync(cachePath('person', f), 'utf8'))).filter((p) => !p.skipped) as { id: string; fold: number }[];
  type Part = { X: (number | null)[][]; y: number[] };
  const num = (v: number | null) => (v === null ? 0 : v);
  const names = featuresOf('session'), cols = noSkipFeaturesOf('session').map((c) => names.indexOf(c));
  const lp = names.indexOf('logPeople'), ll = names.indexOf('logListed'), sp1 = names.indexOf('sessProper'), sp2 = names.indexOf('sessPlays');
  const fitRows = new Map<string, { X: number[][]; y: number[] }>(), held = new Map<string, Part>();
  for (const p of lbPeople) {
    const f = readJsonGz<{ session: { train: Part; held: Part } }>(cachePath('stick', `${p.id}.json.gz`));
    const r = rng(lbHash(p.id) + 17); // the same seeded 800-row subsample as models.ts (session is drawn first there)
    const idx = f.session.train.y.map((_, i) => i).map((i) => [r(), i] as const).sort((a, b) => a[0] - b[0]).slice(0, 800).map(([, i]) => i);
    fitRows.set(p.id, { X: idx.map((i) => f.session.train.X[i].map(num)), y: idx.map((i) => f.session.train.y[i]) });
    held.set(p.id, f.session.held);
  }
  const scored: Held[] = [];
  for (let f = 0; f < FOLDS; f++) {
    const X: number[][] = [], y: number[] = [];
    for (const p of lbPeople) if (p.fold !== f) { const o = fitRows.get(p.id)!; o.X.forEach((x, i) => { X.push(cols.map((c) => x[c])); y.push(o.y[i]); }); }
    const m = fitLogReg(X, y, 1);
    for (const p of lbPeople) {
      if (p.fold !== f) continue;
      const H = held.get(p.id)!, xs = H.X.map((x) => x.map(num));
      scored.push({ id: p.id, p: xs.map((x) => predict(m, cols.map((c) => x[c]))), y: H.y, logPeople: xs.map((x) => x[lp]), logListed: xs.map((x) => x[ll]), sessProper: xs.map((x) => x[sp1]), sessPlays: xs.map((x) => x[sp2]) });
    }
    log(`game 2 ListenBrainz fold ${f} fitted on ${y.length} rows`);
  }
  const lb = tripleRows(scored);
  results.push({ key: 'g2l', game: 2, data: 'ListenBrainz', label: 'which of 3 new artists sticks: model vs popularity', unit: 'accuracy', baseName: 'popularity', rows: lb.vsPop, primary: true, eligible: lb.vsPop.length, total: lbPeople.length });
  results.push({ key: 'g2lc', game: 2, data: 'ListenBrainz', label: 'which of 3 new artists sticks: model vs chance', unit: 'accuracy', baseName: 'chance', rows: lb.vsChance, primary: false, eligible: lb.vsChance.length, total: lbPeople.length });
  results.push({ key: 'g2lf', game: 2, data: 'ListenBrainz', label: 'which of 3 new artists sticks: model vs most played in its first session (exploratory, added after the run)', unit: 'accuracy', baseName: 'first session', rows: lb.vsFirst, primary: false, eligible: lb.vsFirst.length, total: lbPeople.length });
}

// ======================================================================== game 3: next week's top 5
const HALF_LIVES = [3, 7, 14, 30, 60, 120, 365, 3650];
function weekGame(id: string, plays: Play[], cut: number): Row | null {
  const roundsFrom = (start: number) => Array.from({ length: HOLD }, (_, k) => start + k * WEEK)
    .map((t) => ({ t, truth: weekTop(plays, t, t + WEEK) })).filter((r): r is { t: number; truth: Set<string> } => r.truth !== null);
  // half-life tuned on the 8 weeks before the cutoff (only plays before each round are read)
  const val = roundsFrom(cut - HOLD * WEEK);
  let hl = 30, best = -1;
  if (val.length) for (const h of HALF_LIVES) {
    const s = mean(val.map((r) => precisionAt(topByDecay(plays, r.t, h), r.truth)));
    if (s > best + 1e-12) { best = s; hl = h; }
  }
  const held = roundsFrom(cut);
  if (held.length < 4) return null;
  const model = mean(held.map((r) => precisionAt(topByDecay(plays, r.t, hl), r.truth)));
  const base = mean(held.map((r) => precisionAt(topByCount(plays, r.t - 28 * DAY, r.t), r.truth)));
  return { id, model, base, n: held.length, extra: { hl } };
}
if (ONLY.includes(3)) {
  const sRows: Row[] = [];
  const people = spotifyPeople();
  for (const P of people) { const r = weekGame(P.name, P.plays, P.cut); if (r) sRows.push(r); }
  results.push({ key: 'g3s', game: 3, data: 'Spotify', label: "next week's top 5: decayed score vs last 4 weeks' top 5", unit: 'precision at 5', baseName: 'last 4 weeks', rows: sRows, primary: true, eligible: sRows.length, total: people.length });
  log(`game 3 Spotify: ${sRows.length} people`);
  spotify = null;
  const lbIds = readdirSync(cachePath('person')).filter((f) => f.endsWith('.json')).map((f) => JSON.parse(readFileSync(cachePath('person', f), 'utf8'))).filter((p) => !p.skipped).map((p) => p.id as string);
  const lRows: Row[] = [];
  let k = 0;
  for (const id of lbIds) {
    const P = loadPerson(id);
    if (P) { const r = weekGame(id, P.plays, P.cut); if (r) lRows.push(r); }
    if (++k % 50 === 0) log(`game 3 ListenBrainz ${k}/${lbIds.length}`);
  }
  results.push({ key: 'g3l', game: 3, data: 'ListenBrainz', label: "next week's top 5: decayed score vs last 4 weeks' top 5", unit: 'precision at 5', baseName: 'last 4 weeks', rows: lRows, primary: true, eligible: lRows.length, total: lbIds.length });
}

// ======================================================================== summary
const EPS = 1e-12;
const summary = results.map((R) => {
  const d = R.rows.map((r) => r.model - r.base);
  const wins = d.filter((x) => x > EPS).length, losses = d.filter((x) => x < -EPS).length, ties = d.length - wins - losses;
  const st = signTest(d.map((x) => (Math.abs(x) <= EPS ? 0 : x)));
  const share = d.length ? wins / d.length : NaN, [wLo, wHi] = wilson(wins, d.length);
  return {
    key: R.key, game: R.game, data: R.data, label: R.label, unit: R.unit, baseName: R.baseName, primary: R.primary, eligible: R.eligible, total: R.total,
    modelMean: mean(R.rows.map((r) => r.model)), baseMean: mean(R.rows.map((r) => r.base)), diffMean: mean(d), diffCi: bootstrap(d, mean, 4000, 3),
    diffQ: [0, 0.25, 0.5, 0.75, 1].map((q) => quantile(d, q)), wins, losses, ties, share, wilson: [wLo, wHi], p: st.p, pHolm: NaN, pass: false,
  };
});
const prim = summary.filter((s) => s.primary);
if (ONLY.length === 3) {
  const adj = holm(prim.map((s) => s.p));
  prim.forEach((s, i) => { s.pHolm = adj[i]; s.pass = s.share >= 0.7 && s.wilson[0] > 0.5 && s.pHolm < 0.05 && s.wins > s.losses; });
}

const lines: string[] = [];
for (const s of summary) {
  lines.push(`${s.primary ? 'PRIMARY  ' : 'secondary'} game ${s.game} ${s.data}: ${s.label}`);
  lines.push(`  ${s.eligible} of ${s.total} people eligible; mean per person ${s.unit}: model ${f3(s.modelMean)} vs ${s.baseName} ${f3(s.baseMean)}; difference ${f3(s.diffMean)} [${f3(s.diffCi[0])}, ${f3(s.diffCi[1])}]`);
  lines.push(`  difference min/q1/median/q3/max ${s.diffQ.map(f3).join(' / ')}`);
  lines.push(`  model wins for ${s.wins} of ${s.eligible} (${f3(s.share)}, Wilson [${f3(s.wilson[0])}, ${f3(s.wilson[1])}]), loses ${s.losses}, ties ${s.ties}; sign p ${pf(s.p)}${s.primary ? `, Holm ${pf(s.pHolm)}; ${s.pass ? 'PASSES' : 'misses'} the product-grade bar` : ''}`);
}
console.log(lines.join('\n'));

// per-person tables (people as P01..., in the order they were scored)
const md: string[] = [];
for (const R of results) {
  const s = summary.find((x) => x.key === R.key)!;
  md.push(`### Game ${R.game}, ${R.data}: ${R.label}`, '');
  md.push(`${s.eligible} of ${s.total} people eligible. Mean per person (${R.unit}): model ${f3(s.modelMean)}, ${R.baseName} ${f3(s.baseMean)}, difference ${f3(s.diffMean)} [${f3(s.diffCi[0])}, ${f3(s.diffCi[1])}]. Per-person difference min ${f3(s.diffQ[0])}, quartiles ${f3(s.diffQ[1])} / ${f3(s.diffQ[2])} / ${f3(s.diffQ[3])}, max ${f3(s.diffQ[4])}. The model wins for ${s.wins} of ${s.eligible} (share ${f3(s.share)}, Wilson 95% [${f3(s.wilson[0])}, ${f3(s.wilson[1])}]), loses for ${s.losses}, ties ${s.ties}. Sign test p ${pf(s.p)}${R.primary ? `, Holm ${pf(s.pHolm)}. ${s.pass ? 'Passes' : 'Misses'} the product-grade bar.` : ' (secondary, unadjusted).'}`, '');
  if (R.rows.length <= 20) {
    const ex = R.rows[0]?.extra ? Object.keys(R.rows[0].extra).filter((k) => !['ciLo', 'ciHi', 'tModel', 'tArtist', 'baseIsArtist'].includes(k)) : [];
    const ci = R.key === 'g1';
    md.push(`| person | rounds | model | ${R.baseName} | difference${ci ? ' [95% over sessions]' : ''} | ${ex.join(' | ')}${ci ? ' | baseline used' : ''} |`);
    md.push(`|---|---|---|---|---|${ex.map(() => '---|').join('')}${ci ? '---|' : ''}`);
    R.rows.forEach((r, i) => md.push(`| P${String(i + 1).padStart(2, '0')} | ${r.n} | ${f3(r.model)} | ${f3(r.base)} | ${f3(r.model - r.base)}${ci ? ` [${f2(r.extra!.ciLo)}, ${f2(r.extra!.ciHi)}]` : ''} | ${ex.map((k) => f3(r.extra![k])).join(' | ')}${ci ? ` | ${r.extra!.baseIsArtist ? 'artist rate' : 'own rate'}` : ''} |`));
    md.push('');
  } else {
    // a histogram of differences for the large set
    const edges = [-1, -0.2, -0.1, -0.05, -EPS, EPS, 0.05, 0.1, 0.2, 1.0001];
    const lab = ['below -0.2', '-0.2 to -0.1', '-0.1 to -0.05', '-0.05 to 0', 'tie', '0 to 0.05', '0.05 to 0.1', '0.1 to 0.2', 'above 0.2'];
    const d = R.rows.map((r) => r.model - r.base);
    md.push('| difference | people |', '|---|---|');
    lab.forEach((l, k) => md.push(`| ${l} | ${d.filter((x) => (k === 4 ? Math.abs(x) <= EPS : x >= edges[k] && x < edges[k + 1] && Math.abs(x) > EPS)).length} |`));
    md.push('');
  }
}
mkdirSync('bench/out', { recursive: true });
writeFileSync('bench/out/game.md', md.join('\n'));
writeFileSync('bench/out/game.json', JSON.stringify({ at: new Date().toISOString(), summary, results }));
log('wrote bench/out/game.json and bench/out/game.md');

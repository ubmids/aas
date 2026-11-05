// Round 3, step 4: the statistics. Reads bench/out/lb3/person/*.json and models.json, writes
// bench/out/lb3.json and bench/out/REPORT_LB.md (folded into REPORT.md by bench/report.ts).
// Same reading as the Spotify report: medians over people with 95% percentile bootstrap intervals,
// paired differences signed so positive means the model is better, exact two-sided sign tests, and
// Holm across every sign test in this round (the paired comparisons and the door comparisons).
// Traits: split-half stability with bootstrap intervals, then the factor structure (normal scores,
// Horn's parallel analysis, varimax loadings with bootstrap intervals).
// Run: npx tsx bench/lb3/analyse.ts
import { readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { cachePath, log } from './common.js';
import { bootstrap, signTest, median, pearson, spearman, normalScores, parallelAnalysis, pcaLoadings, corrMatrix } from '../../engine/src/model/stats.js';
import { TRAITS } from '../../engine/src/model/traits.js';
import { rng } from '../../engine/src/model/linalg.js';

const ALPHA = 0.05;
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '-');
const ci = ([lo, hi]: [number, number]) => `[${f3(lo)}, ${f3(hi)}]`;
const pf = (p: number) => (p < 0.001 ? '<0.001' : p.toFixed(3));
const people = readdirSync(cachePath('person')).filter((f) => f.endsWith('.json')).map((f) => JSON.parse(readFileSync(cachePath('person', f), 'utf8'))).filter((p) => !p.skipped);
const models = existsSync(cachePath('models.json')) ? JSON.parse(readFileSync(cachePath('models.json'), 'utf8')) : null;
const nAll = readdirSync(cachePath('person')).filter((f) => f.endsWith('.json')).length;
log(`${people.length} people with a held-out split (of ${nAll})`);

const rows: Record<string, number>[] = people.map((p) => ({ ...p.tasks, ...(models?.stick[p.id] ?? {}), ...(models?.rank2[p.id] ?? {}) }));
for (const r of rows) for (const k of Object.keys(r)) if (r[k] === null) r[k] = NaN;

// ---------------------------------------------------------------- paired comparisons
interface Cmp { section: string; label: string; a: string; b: string | number; lower?: boolean }
const C: Cmp[] = [
  { section: '2 next 8 weeks\' artists (NDCG@10)', label: 'decay vs all-time', a: 'artists', b: 'artists_allTime' },
  { section: '2 next 8 weeks\' artists (NDCG@10)', label: 'decay vs last 4 weeks', a: 'artists', b: 'artists_last4w' },
  { section: '4 when you listen (bits per play, UTC hour of the week)', label: 'tuned clock vs uniform', a: 'clock', b: 0 },
  { section: '4 when you listen (bits per play, UTC hour of the week)', label: 'tuned clock vs all-time clock', a: 'clock', b: 'clock_allTime' },
  { section: '6 new loves, rolling windows (recall@50)', label: 'two-stage ranker vs popular', a: 'ts_roll50_model', b: 'ts_roll50_popular' },
  { section: '6 new loves, rolling windows (recall@50)', label: 'two-stage ranker vs plain lists', a: 'ts_roll50_model', b: 'ts_roll50_lists' },
  { section: '6 new loves, rolling windows (recall@50)', label: 'plain lists vs popular', a: 'ts_roll50_lists', b: 'ts_roll50_popular' },
  { section: '6 new loves, rolling windows (recall@20)', label: 'two-stage ranker vs popular', a: 'ts_roll20_model', b: 'ts_roll20_popular' },
  { section: '6 new loves, held-out 8 weeks (recall@50)', label: 'two-stage ranker vs popular', a: 'ts_fall50_model', b: 'ts_fall50_popular' },
  { section: '8 eras (NDCG@10)', label: 'era-aware forecast vs decay', a: 'eraForecast', b: 'artists' },
  { section: '8 eras (NDCG@10)', label: 'current era only vs decay', a: 'eraOnly', b: 'artists' },
  { section: '9 will it stick, after the first session (AUC, training period)', label: 'model vs own base rate', a: 'st_s_roll_auc', b: 'st_s_roll_auc_base' },
  { section: '9 will it stick, after the first session (AUC, training period)', label: 'model vs popularity', a: 'st_s_roll_auc', b: 'st_s_roll_auc_pop' },
  { section: '9 will it stick, after the first session (AUC, training period)', label: 'model vs the model without ListenBrainz list features', a: 'st_s_roll_auc', b: 'st_s_roll_auc_noLists' },
  { section: '9 will it stick, after the first session (average precision, training period)', label: 'model vs popularity', a: 'st_s_roll_ap', b: 'st_s_roll_ap_pop' },
  { section: '9 will it stick, after the first session (Brier, training period)', label: 'model vs own base rate (lower is better)', a: 'st_s_roll_brier', b: 'st_s_roll_brier_base', lower: true },
  { section: '9 will it stick, after the first session (AUC, held-out 8 weeks)', label: 'model vs own base rate', a: 'st_s_held_auc', b: 'st_s_held_auc_base' },
  { section: '9 will it stick, after the first session (AUC, held-out 8 weeks)', label: 'model vs popularity', a: 'st_s_held_auc', b: 'st_s_held_auc_pop' },
  { section: '10 will it stick, after 7 days (AUC, training period)', label: 'model vs first-week play counts', a: 'st_w_roll_auc', b: 'st_w_roll_auc_wkCounts' },
  { section: '10 will it stick, after 7 days (AUC, training period)', label: 'model vs popularity', a: 'st_w_roll_auc', b: 'st_w_roll_auc_pop' },
  { section: '10 will it stick, after 7 days (AUC, held-out 8 weeks)', label: 'model vs first-week play counts', a: 'st_w_held_auc', b: 'st_w_held_auc_wkCounts' },
];
const have = (k: string) => rows.some((r) => Number.isFinite(r[k]));
type Done = Cmp & { n: number; ma: number; ia: [number, number]; mb: number; ib: [number, number]; md: number; id: [number, number]; mean: number; im: [number, number]; wins: number; losses: number; ties: number; p: number; holm: number; verdict: string };
const cmps: Done[] = C.filter((c) => have(c.a) && (typeof c.b === 'number' || have(c.b))).map((c, i) => {
  const units = rows.filter((r) => Number.isFinite(r[c.a]) && (typeof c.b === 'number' || Number.isFinite(r[c.b])));
  const bv = (r: Record<string, number>) => (typeof c.b === 'number' ? c.b : r[c.b]);
  const diff = (r: Record<string, number>) => (c.lower ? bv(r) - r[c.a] : r[c.a] - bv(r));
  return {
    ...c, n: units.length,
    ma: median(units.map((r) => r[c.a])), ia: bootstrap(units, (u) => median(u.map((r) => r[c.a])), 2000, 11 + i),
    mb: median(units.map(bv)), ib: typeof c.b === 'number' ? [c.b, c.b] : bootstrap(units, (u) => median(u.map(bv)), 2000, 31 + i),
    md: median(units.map(diff)), id: bootstrap(units, (u) => median(u.map(diff)), 2000, 51 + i),
    mean: units.reduce((t, r) => t + diff(r), 0) / units.length, im: bootstrap(units, (u) => u.reduce((t, r) => t + diff(r), 0) / u.length, 2000, 71 + i),
    ...signTest(units.map(diff)), holm: NaN, verdict: '',
  };
});

// ---------------------------------------------------------------- doors
interface Tally { fallYes: number; fallN: number; ctlYes: number; ctlN: number }
const DOOR_FACTS: [string, string, (p: any) => Tally | undefined][] = [
  ['came on straight after an artist that lists it', 'prevLists', (p) => p.doors.facets.prevLists],
  ['a recent favourite lists it', 'favLists', (p) => p.doors.facets.favLists],
  ['a recent favourite lists it in its top 25', 'favListsTop25', (p) => p.doors.facets.favListsTop25],
  ['the artist just before was a recent favourite', 'prevIsFav', (p) => p.doors.facets.prevIsFav],
  ['a recent favourite lists it (vs never-played artists)', 'unheard', (p) => p.doors.unheard],
];
const poolT = (xs: Tally[]) => xs.reduce((a, t) => ({ fy: a.fy + t.fallYes, fn: a.fn + t.fallN, cy: a.cy + t.ctlYes, cn: a.cn + t.ctlN }), { fy: 0, fn: 0, cy: 0, cn: 0 });
const ratioOf = (xs: Tally[]) => { const s = poolT(xs); return s.cy > 0 && s.fn > 0 ? (s.fy / s.fn) / (s.cy / s.cn) : NaN; };
function doorCmp(label: string, get: (p: any) => Tally | undefined, ps: any[], seed: number) {
  const ts = ps.map(get).filter((t): t is Tally => !!t && t.fallN > 0);
  const s = poolT(ts), elig = ts.filter((t) => t.fallN >= 3), st = signTest(elig.map((t) => t.fallYes / t.fallN - t.ctlYes / t.ctlN));
  return { label, n: ts.length, eligible: elig.length, fall: s.fy / s.fn, ctl: s.cy / s.cn, ratio: ratioOf(ts), iv: bootstrap(ts, ratioOf, 2000, seed), ...st, holm: NaN, verdict: '' };
}
const doorsAll = people.filter((p) => p.doors);
const doorRows = DOOR_FACTS.map(([label, , get], i) => doorCmp(label, get, doorsAll, 301 + i));
// who shaped the lists: ListenBrainz computed its similar-artist lists from its users' sessions,
// these people included. If that leaked, heavy listeners (more sessions in the lists) would show a
// stronger list door than light ones. Compare the lightest and heaviest thirds by listens.
const byVol = [...doorsAll].sort((a, b) => a.plays - b.plays), third = Math.floor(byVol.length / 3);
const volCheck = [['lightest third', byVol.slice(0, third)], ['heaviest third', byVol.slice(-third)]].map(([lab, ps], i) => doorCmp(lab as string, (p) => p.doors.unheard, ps as any[], 351 + i));

// ---------------------------------------------------------------- Holm over the round's sign tests
{
  const fam = [...cmps.map((c) => ({ c, p: c.p })), ...doorRows.map((d) => ({ c: d, p: d.p }))].sort((x, y) => x.p - y.p);
  let run = 0;
  fam.forEach((x, k) => { run = Math.max(run, Math.min(1, (fam.length - k) * x.p)); x.c.holm = run; });
}
for (const c of cmps) {
  // a sign test only counts for the model when the model is the one winning
  const meanOk = c.im[0] > 0, signOk = c.p < ALPHA && c.wins > c.losses, worse = c.im[1] < 0 && c.losses > c.wins;
  c.verdict = c.holm < ALPHA && meanOk && signOk ? 'survives' : signOk && meanOk ? 'survives before multiple-test adjustment' : worse ? 'worse than baseline' : meanOk || signOk ? 'weak' : 'no difference';
}
for (const d of doorRows) {
  // the round 2 bar: a door counts when the ratio is at least 1.5, its interval clears 1 and most people agree
  const more = d.ratio >= 1.5 && d.iv[0] > 1 && d.wins > d.eligible / 2, less = d.ratio <= 1 / 1.5 && d.iv[1] < 1 && d.losses > d.eligible / 2;
  const small = !more && d.iv[0] > 1 && d.wins > d.losses && d.holm < ALPHA;
  d.verdict = more ? (d.holm < ALPHA ? 'more common for new loves, survives' : 'more common for new loves, not after adjustment') : less ? (d.holm < ALPHA ? 'less common for new loves, survives' : 'less common for new loves, not after adjustment') : small ? 'more common for new loves, but under the 1.5 bar' : 'no clear difference';
}

// ---------------------------------------------------------------- traits: stability
const KEYS = TRAITS.map((t) => t.key);
const stab = TRAITS.map((t, i) => {
  const pairs = people.map((p) => [p.traits.a[t.key], p.traits.b[t.key]]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b) && a !== null && b !== null) as [number, number][];
  if (pairs.length < 10) return { key: t.key, group: t.group, n: pairs.length, r: NaN, rho: NaN, ir: [NaN, NaN] as [number, number], irho: [NaN, NaN] as [number, number], reading: 'not measurable here' };
  const r = pearson(pairs.map((p) => p[0]), pairs.map((p) => p[1])), rho = spearman(pairs.map((p) => p[0]), pairs.map((p) => p[1]));
  const ir = bootstrap(pairs, (u) => pearson(u.map((p) => p[0]), u.map((p) => p[1])), 2000, 91 + i), irho = bootstrap(pairs, (u) => spearman(u.map((p) => p[0]), u.map((p) => p[1])), 2000, 131 + i);
  const reading = ir[0] >= 0.5 && irho[0] >= 0.5 ? 'stable, interval above 0.5' : ir[0] > 0 && irho[0] > 0 ? 'above zero, not shown to reach 0.5' : 'not shown to be stable';
  return { key: t.key, group: t.group, n: pairs.length, r, rho, ir, irho, reading };
});

// ---------------------------------------------------------------- traits: factor structure
const FA_KEYS = stab.filter((s) => s.n >= 100).map((s) => s.key);
const faPeople = people.filter((p) => FA_KEYS.every((k) => Number.isFinite(p.traits.full[k]) && p.traits.full[k] !== null));
const raw = faPeople.map((p) => FA_KEYS.map((k) => p.traits.full[k] as number));
const Z = (() => { const cols = FA_KEYS.map((_, j) => normalScores(raw.map((r) => r[j]))); return raw.map((_, i) => cols.map((c) => c[i])); })();
const R = corrMatrix(Z), pa = parallelAnalysis(Z, 1000, 5), K = Math.max(1, pa.keep);
const load = pcaLoadings(Z, K);
// bootstrap the loadings: each resample's components are matched to the reference by the largest
// absolute congruence, and flipped to agree in sign
const congr = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0) / Math.sqrt(a.reduce((s, v) => s + v * v, 0) * b.reduce((s, v) => s + v * v, 0));
const col = (L: number[][], c: number) => L.map((r) => r[c]);
const bootL: number[][][] = FA_KEYS.map(() => Array.from({ length: K }, () => []));
const rr = rng(77);
let bootKeep: number[] = [];
for (let b = 0; b < 500; b++) {
  const s = Array.from({ length: Z.length }, () => Z[Math.floor(rr() * Z.length)]);
  const L = pcaLoadings(s, K), used = new Set<number>();
  for (let c = 0; c < K; c++) {
    let best = -1, bc = -1;
    for (let d = 0; d < K; d++) if (!used.has(d)) { const g = Math.abs(congr(col(load, c), col(L, d))); if (g > best) { best = g; bc = d; } }
    used.add(bc);
    const sign = Math.sign(congr(col(load, c), col(L, bc))) || 1;
    FA_KEYS.forEach((_, i) => bootL[i][c].push(sign * L[i][bc]));
  }
  if (b < 200) bootKeep.push(parallelAnalysis(s, 100, 1000 + b).keep);
}
const pct = (a: number[], q: number) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(q * (s.length - 1))]; };
const keepDist = Object.fromEntries([...new Set(bootKeep)].sort().map((k) => [k, bootKeep.filter((x) => x === k).length / bootKeep.length]));

// ---------------------------------------------------------------- write
const out = { at: new Date().toISOString(), people: people.length, of: nAll, cmps, doors: doorRows, volCheck, stability: stab, factor: { keys: FA_KEYS, n: faPeople.length, corr: R, parallel: pa, loadings: load, loadingCI: bootL.map((r) => r.map((xs) => [pct(xs, 0.025), pct(xs, 0.975)])), keepDist }, models: models ? { pooledStick: models.pooledStick, stickWeights: models.stickWeights, leak: models.leak, rank2Pool: models.rank2Pool, rank2Weights: models.rank2Weights } : null };
writeFileSync('bench/out/lb3.json', JSON.stringify(out));

const L: string[] = [];
L.push('## Round 3: 343 ListenBrainz listeners', '');
const span = people.map((p) => p.days / 365.25);
L.push(`${people.length} of the ${nAll} people kept in bench/lbusers have at least 1,000 listens before their last 8 weeks and 200 in them, so they get the same time split as the Spotify bench. Median history ${f3(median(span))} years, median ${Math.round(median(people.map((p) => p.plays)))} listens. ListenBrainz histories have no play lengths, skips, start reasons, shuffle, device or country, so finish-or-skip, the selves, the game against matched decoys and the local clock cannot be run here; the clock below uses UTC hours. Everything learned (taste map, stick model, ranker) is fitted on the other four of five folds of people. Holm runs over all ${cmps.length + doorRows.length} sign tests of this round.`, '');
L.push('One caveat runs through every ListenBrainz list feature here: ListenBrainz computes its similar-artist lists from its users\' listening sessions, and these people are ListenBrainz users. Their own sessions are a small part of the data behind the lists, but not zero, so list features can carry a trace of the very listening they are used to explain. The stick model is reported with and without them, and the door section checks whether heavy listeners (more sessions in the lists) show a stronger list door than light ones.', '');
L.push('### Paired comparisons', '');
L.push('| comparison | n | model median [95%] | baseline median [95%] | median difference [95%] | mean difference [95%] | wins/losses/ties | sign p | Holm p | verdict |', '|---|---|---|---|---|---|---|---|---|---|');
let sec = '';
for (const c of cmps) {
  if (c.section !== sec) { sec = c.section; L.push(`| **${sec}** | | | | | | | | | |`); }
  L.push(`| ${c.label} | ${c.n} | ${f3(c.ma)} ${ci(c.ia)} | ${typeof c.b === 'number' ? f3(c.b) : `${f3(c.mb)} ${ci(c.ib)}`} | ${f3(c.md)} ${ci(c.id)} | ${f3(c.mean)} ${ci(c.im)} | ${c.wins}/${c.losses}/${c.ties} | ${pf(c.p)} | ${pf(c.holm)} | ${c.verdict} |`);
}
if (models) {
  const P = models.pooledStick as Record<string, { auc: number; ap: number; brier: number; ece: number; n: number; pos: number }>;
  L.push('', '### Will a new artist stick, pooled', '');
  L.push(`Rows: every artist a person first heard after their first 12 weeks, at most 4,000 per person in the training period (a seeded sample) and all of them in the held-out 8 weeks. The leakage re-read checked ${models.leak.checked} sampled rows against the history cut at the horizon; ${models.leak.failed} differed. The model uses the columns this data can fill: the ListenBrainz list facts, favourites in the session, map nearness, popularity, the first session's plays, tracks, repeats and returns, its length, the own base rate, and after 7 days the week's plays, days, tracks and whether it came back after the first session.`, '');
  L.push('| horizon | rows | model | pooled AUC | pooled AP | pooled Brier | pooled ECE | stuck / rows |', '|---|---|---|---|---|---|---|---|');
  const lab: Record<string, string> = { full: 'model', noLists: 'model without ListenBrainz lists', base: 'own base rate', pop: 'popularity', wkCounts: 'first-week play counts', shuffled: 'labels shuffled (null)' };
  for (const h of ['session', 'week']) for (const part of ['roll', 'held']) for (const k of Object.keys(lab)) {
    const r = P[`${h}:${part}:${k}`];
    if (!r) continue;
    L.push(`| ${h === 'session' ? 'first session' : 'first 7 days'} | ${part === 'roll' ? 'training period' : 'held-out 8 weeks'} | ${lab[k]} | ${f3(r.auc)} | ${f3(r.ap)} | ${f3(r.brier)} | ${f3(r.ece)} | ${r.pos} / ${r.n} |`);
  }
  L.push('', `Session-horizon weights (fold 0, standardised, reading only): ${Object.entries(models.stickWeights.session as Record<string, number>).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')}.`, '');
  const RP = models.rank2Pool as Record<string, number>, q = (pre: string, k: string) => f3((RP[`${pre}:${k}`] ?? 0) / RP[`${pre}:n`]);
  L.push('### New loves: the two-stage ranker, pooled', '');
  L.push('| windows | new loves | candidate coverage | ranker @20 | lists @20 | popular @20 | ranker @50 | lists @50 | popular @50 |', '|---|---|---|---|---|---|---|---|---|');
  for (const [lab2, pre] of [['rolling, inside training', 'ts_roll'], ['held-out 8 weeks', 'ts_fall']]) L.push(`| ${lab2} | ${RP[`${pre}:n`]} | ${q(pre, 'cover')} | ${q(pre, 'model20')} | ${q(pre, 'lists20')} | ${q(pre, 'popular20')} | ${q(pre, 'model50')} | ${q(pre, 'lists50')} | ${q(pre, 'popular50')} |`);
  L.push('');
}
L.push('### Doors: list facts at the first play', '');
L.push('Start reasons do not exist here, so only the list facts are read: whether the artist played just before (same session) lists the new artist, and whether a recent favourite does. New loves are compared with up to 5 artists the same person tried and dropped (at most 2 listens ever), first heard nearest in time, and the favourite fact also with never-played ListenBrainz artists named on as many lists.', '');
L.push('| fact | people | new loves | controls | ratio [95%] | wins/losses/ties | sign p | Holm p | verdict |', '|---|---|---|---|---|---|---|---|---|');
for (const d of doorRows) L.push(`| ${d.label} | ${d.n} | ${f3(d.fall)} | ${f3(d.ctl)} | ${d.ratio.toFixed(2)} ${ci(d.iv)} | ${d.wins}/${d.losses}/${d.ties} | ${pf(d.p)} | ${pf(d.holm)} | ${d.verdict} |`);
L.push('', 'The list-leak check, a recent favourite lists it vs never-played artists, by listening volume:', '', '| people | n | new loves | controls | ratio [95%] | wins/losses |', '|---|---|---|---|---|---|');
for (const d of volCheck) L.push(`| ${d.label} | ${d.n} | ${f3(d.fall)} | ${f3(d.ctl)} | ${d.ratio.toFixed(2)} ${ci(d.iv)} | ${d.wins}/${d.losses} |`);
const g = doorsAll.flatMap((p) => p.doors.growth).filter((x: number | null) => x !== null && Number.isFinite(x)) as number[];
L.push('', `${doorsAll.reduce((s, p) => s + p.doors.falls, 0)} new loves and ${doorsAll.reduce((s, p) => s + p.doors.dropped, 0)} tried-and-dropped artists. Median days from first listen to the 10th: ${f3(median(g))}.`, '');
L.push('### Traits: stability', '');
L.push('Each trait on the two halves of every person\'s training weeks, correlated across people, with bootstrap intervals over people. Traits that need skips, play lengths or a time zone cannot be measured here.', '');
L.push('| group | trait | n | Pearson r [95%] | Spearman rho [95%] | reading |', '|---|---|---|---|---|---|');
for (const s of stab) L.push(`| ${s.group} | ${s.key} | ${s.n} | ${f3(s.r)} ${ci(s.ir)} | ${f3(s.rho)} ${ci(s.irho)} | ${s.reading} |`);
L.push('', '### Traits: how many dials', '');
L.push(`${faPeople.length} people with every measurable trait. Each trait is turned into normal scores, the correlation matrix is decomposed, and Horn's parallel analysis (1,000 column shuffles, 95th percentile) decides how many components carry more than noise. Loadings are varimax-rotated, with 95% intervals from 500 bootstrap resamples of people (components matched by congruence).`, '');
L.push('| component | observed eigenvalue | 95th percentile under shuffling | kept |', '|---|---|---|---|');
pa.observed.forEach((v, i) => L.push(`| ${i + 1} | ${f3(v)} | ${f3(pa.p95[i])} | ${i < pa.keep ? 'yes' : 'no'} |`));
L.push('', `Number of components kept in 200 bootstrap resamples: ${Object.entries(keepDist).map(([k, v]) => `${k}: ${Math.round(100 * (v as number))}%`).join(', ')}.`, '');
L.push(`| trait | ${Array.from({ length: K }, (_, c) => `component ${c + 1} [95%]`).join(' | ')} |`, `|---|${Array.from({ length: K }, () => '---|').join('')}`);
FA_KEYS.forEach((k, i) => L.push(`| ${k} | ${load[i].map((v, c) => `${f3(v)} ${ci([pct(bootL[i][c], 0.025), pct(bootL[i][c], 0.975)])}`).join(' | ')} |`));
L.push('', 'Correlations between traits (normal scores):', '', `| | ${FA_KEYS.join(' | ')} |`, `|---|${FA_KEYS.map(() => '---|').join('')}`);
FA_KEYS.forEach((k, i) => L.push(`| ${k} | ${R[i].map(f3).join(' | ')} |`));
L.push('', '### What survives in round 3', '');
const surv = cmps.filter((c) => c.verdict === 'survives'), pre = cmps.filter((c) => c.verdict.startsWith('survives before')), weak = cmps.filter((c) => c.verdict === 'weak'), none = cmps.filter((c) => c.verdict === 'no difference' || c.verdict === 'worse than baseline');
const nm = (c: Done) => `${c.section}: ${c.label}`;
L.push(`Survive the adjustment: ${surv.map(nm).join('; ') || 'none'}.`, '');
L.push(`Significant only before adjustment: ${pre.map(nm).join('; ') || 'none'}.`, '');
L.push(`Weak (one test passes, the other does not): ${weak.map(nm).join('; ') || 'none'}.`, '');
L.push(`No difference or worse: ${none.map((c) => `${nm(c)} (${c.verdict})`).join('; ') || 'none'}.`, '');
L.push(`Doors: ${doorRows.map((d) => `${d.label}: ${d.verdict}`).join('; ')}.`, '');
writeFileSync('bench/out/REPORT_LB.md', L.join('\n') + '\n');
log(`wrote bench/out/REPORT_LB.md and bench/out/lb3.json (${cmps.length} comparisons, ${surv.length} survive)`);
void KEYS;

// The bench report: every headline number with a 95% bootstrap interval over people, paired
// comparisons against each baseline (median difference with its interval, exact sign test, Holm
// adjustment across the family), skip-model calibration, the doors ratios and trait stability.
// Reads bench/out/{bench,doors,traits}.json (written by `npm run bench`, `doors`, `traits`) and
// writes bench/out/REPORT.md. People appear as P01..P13, never by name.
// `npm run report` runs all three and then this; `npm run report:only` only rereads the outputs.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { bootstrap, signTest, reliability, median, pearson, spearman, Reliability } from '../engine/src/model/stats.js';

const OUT = 'bench/out';
const read = (f: string) => (existsSync(`${OUT}/${f}`) ? JSON.parse(readFileSync(`${OUT}/${f}`, 'utf8')) : null);
const bench = read('bench.json'), doors = read('doors.json'), traits = read('traits.json');
if (!bench) { console.error('no bench/out/bench.json: run `npm run bench` first'); process.exit(1); }
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '-');
const ci = ([lo, hi]: [number, number]) => `[${f3(lo)}, ${f3(hi)}]`;
const pf = (p: number) => (p < 0.001 ? '<0.001' : p.toFixed(3));
const rank2 = read('rank2.json'), stick = read('stick.json');
const rows: Record<string, number>[] = bench.rows.map((r: Record<string, number>, i: number) => {
  const j = rank2 ? rank2.names.indexOf(bench.names[i]) : -1, k = stick ? stick.names.indexOf(bench.names[i]) : -1;
  return { ...r, ...(j >= 0 ? rank2.rows[j] : {}), ...(k >= 0 ? stick.rows[k] : {}) };
});
const ALPHA = 0.05;

// ---------------------------------------------------------------- paired comparisons
interface Cmp { section: string; label: string; a: string; b: string | number; lower?: boolean }
const C: Cmp[] = [
  { section: '1 finish or skip (AUC)', label: 'skip model vs artist skip rate', a: 'skip', b: 'skip_b' },
  { section: '2 next 8 weeks\' artists (NDCG@10)', label: 'decay vs all-time', a: 'artists', b: 'artists_allTime' },
  { section: '2 next 8 weeks\' artists (NDCG@10)', label: 'decay vs last 4 weeks', a: 'artists', b: 'artists_last4w' },
  { section: '3 the game (share correct)', label: 'old scorer vs chance', a: 'game', b: 0.2 },
  { section: '3 the game (share correct)', label: 'co-listening map vs old scorer', a: 'gameMap', b: 'game' },
  { section: '3 the game (share correct)', label: 'blend map vs old scorer', a: 'gameBl', b: 'game' },
  { section: '3 the game (share correct)', label: 'new-artist rounds, map vs chance', a: 'gameMapNew', b: 0.2 },
  { section: '3 the game (share correct)', label: 'new-artist rounds, blend vs chance', a: 'gameBlNew', b: 0.2 },
  { section: '4 when you listen (bits per play)', label: 'local clock vs uniform', a: 'clock', b: 0 },
  { section: '4 when you listen (bits per play)', label: 'tuned clock vs all-time clock', a: 'clock', b: 'clock_allTime' },
  { section: '4 when you listen (bits per play)', label: 'local hours vs UTC hours', a: 'clock', b: 'clockUtc' },
  { section: '5 new or familiar (AUC)', label: 'novelty model vs chance', a: 'novel', b: 0.5 },
  { section: '6 new loves, rolling windows (recall@50)', label: 'v0 ranker vs popular', a: 'roll50_model', b: 'roll50_popular' },
  { section: '6 new loves, rolling windows (recall@50)', label: 'blend ranker vs popular', a: 'bl_roll50_model', b: 'bl_roll50_popular' },
  { section: '6 new loves, rolling windows (recall@50)', label: 'plain lists vs popular', a: 'bl_roll50_lists', b: 'bl_roll50_popular' },
  { section: '6 new loves, rolling windows (recall@50)', label: 'two-stage ranker vs popular', a: 'ts_roll50_model', b: 'ts_roll50_popular' },
  { section: '6 new loves, rolling windows (recall@50)', label: 'two-stage ranker vs plain lists', a: 'ts_roll50_model', b: 'ts_roll50_lists' },
  { section: '6 new loves, rolling windows (recall@20)', label: 'two-stage ranker vs popular', a: 'ts_roll20_model', b: 'ts_roll20_popular' },
  { section: '6 new loves, held-out 8 weeks (recall@50)', label: 'two-stage ranker vs popular', a: 'ts_fall50_model', b: 'ts_fall50_popular' },
  { section: '7 selves from context', label: 'accuracy vs majority self', a: 'self', b: 'self_majority' },
  { section: '7 selves from context', label: 'log-loss vs self frequencies (lower is better)', a: 'selfNll', b: 'selfNll_prior', lower: true },
  { section: '8 eras (NDCG@10)', label: 'era-aware forecast vs decay', a: 'eraForecast', b: 'artists' },
  { section: '8 eras (NDCG@10)', label: 'current era only vs decay', a: 'eraOnly', b: 'artists' },
  { section: '9 will it stick, after the first session (AUC, training period)', label: 'model vs own base rate', a: 'st_s_roll_auc', b: 'st_s_roll_auc_base' },
  { section: '9 will it stick, after the first session (AUC, training period)', label: 'model vs first-play skip', a: 'st_s_roll_auc', b: 'st_s_roll_auc_skip' },
  { section: '9 will it stick, after the first session (AUC, training period)', label: 'model vs popularity', a: 'st_s_roll_auc', b: 'st_s_roll_auc_pop' },
  { section: '9 will it stick, after the first session (AUC, training period)', label: 'model vs the same model without skip, start, shuffle and hour features', a: 'st_s_roll_auc', b: 'st_s_roll_auc_noSkip' },
  { section: '9 will it stick, after the first session (AUC, training period)', label: 'model without skip, start, shuffle and hour features vs own base rate', a: 'st_s_roll_auc_noSkip', b: 'st_s_roll_auc_base' },
  { section: '9 will it stick, after the first session (average precision, training period)', label: 'model vs popularity', a: 'st_s_roll_ap', b: 'st_s_roll_ap_pop' },
  { section: '9 will it stick, after the first session (Brier, training period)', label: 'model vs own base rate (lower is better)', a: 'st_s_roll_brier', b: 'st_s_roll_brier_base', lower: true },
  { section: '9 will it stick, after the first session (AUC, held-out 8 weeks)', label: 'model vs popularity', a: 'st_s_held_auc', b: 'st_s_held_auc_pop' },
  { section: '10 will it stick, after 7 days (AUC, training period)', label: 'model vs first-week play counts', a: 'st_w_roll_auc', b: 'st_w_roll_auc_wkCounts' },
  { section: '10 will it stick, after 7 days (AUC, training period)', label: 'model vs popularity', a: 'st_w_roll_auc', b: 'st_w_roll_auc_pop' },
  { section: '10 will it stick, after 7 days (AUC, training period)', label: 'model vs the same model without skip, start, shuffle and hour features', a: 'st_w_roll_auc', b: 'st_w_roll_auc_noSkip' },
  { section: '10 will it stick, after 7 days (AUC, held-out 8 weeks)', label: 'model vs first-week play counts', a: 'st_w_held_auc', b: 'st_w_held_auc_wkCounts' },
];
const have = (k: string) => rows.some((r) => Number.isFinite(r[k]));
const cmps = C.filter((c) => have(c.a) && (typeof c.b === 'number' || have(c.b))).map((c, i) => {
  const units = rows.filter((r) => Number.isFinite(r[c.a]) && (typeof c.b === 'number' || Number.isFinite(r[c.b])));
  const bv = (r: Record<string, number>) => (typeof c.b === 'number' ? c.b : r[c.b]);
  const diff = (r: Record<string, number>) => (c.lower ? bv(r) - r[c.a] : r[c.a] - bv(r)); // positive = model better
  const s = signTest(units.map(diff));
  return {
    ...c, n: units.length,
    ma: median(units.map((r) => r[c.a])), ia: bootstrap(units, (u) => median(u.map((r) => r[c.a])), 4000, 11 + i),
    mb: median(units.map(bv)), ib: typeof c.b === 'number' ? ([c.b, c.b] as [number, number]) : bootstrap(units, (u) => median(u.map(bv)), 4000, 31 + i),
    md: median(units.map(diff)), id: bootstrap(units, (u) => median(u.map(diff)), 4000, 51 + i),
    mean: units.reduce((t, r) => t + diff(r), 0) / units.length, im: bootstrap(units, (u) => u.reduce((t, r) => t + diff(r), 0) / u.length, 4000, 71 + i),
    ...s, holm: NaN, verdict: '',
  };
});
// Holm step-down over the whole family of sign tests
{
  const order = [...cmps].sort((x, y) => x.p - y.p);
  let run = 0;
  order.forEach((c, k) => { run = Math.max(run, Math.min(1, (order.length - k) * c.p)); c.holm = run; });
}
for (const c of cmps) {
  // a sign test only counts for the model when the model is the one winning
  const meanOk = c.im[0] > 0, signOk = c.p < ALPHA && c.wins > c.losses, worse = c.im[1] < 0 && c.losses > c.wins;
  c.verdict = c.holm < ALPHA && meanOk && signOk ? 'survives' : signOk && meanOk ? 'survives before multiple-test adjustment' : worse ? 'worse than baseline' : meanOk || signOk ? 'weak' : 'no difference';
}

// ---------------------------------------------------------------- calibration
const cal = bench.skipCal as { p: number[]; b: number[]; y: number[] }[];
const pooled = (k: 'p' | 'b') => reliability(cal.flatMap((c) => c[k]), cal.flatMap((c) => c.y));
const calM = pooled('p'), calB = pooled('b');
const per = cal.map((c) => ({ m: reliability(c.p, c.y), b: reliability(c.b, c.y), base: c.y.reduce((s, v) => s + v, 0) / c.y.length, y: c.y }));
// Brier skill against the person's own constant skip rate from the held-out weeks would leak; use
// the training-free reference "predict the artist rate" (the baseline) and report both Briers.
const brierSign = signTest(per.map((x) => x.b.brier - x.m.brier));
const eceSign = signTest(per.map((x) => x.b.ece - x.m.ece));
const relTable = (r: Reliability) => r.bins.filter((b) => b.n).map((b) => `| ${b.lo.toFixed(1)} to ${b.hi.toFixed(1)} | ${b.n} | ${f3(b.meanP)} | ${f3(b.rate)} |`).join('\n');

// ---------------------------------------------------------------- doors
interface Tally { fallYes: number; fallN: number; ctlYes: number; ctlN: number }
function doorRow(label: string, get: (p: any) => Tally | undefined, seed: number) {
  const ts: Tally[] = doors.people.map(get).filter((t: Tally | undefined): t is Tally => !!t && t.fallN > 0);
  const ratio = (xs: Tally[]) => { const s = xs.reduce((a, t) => ({ fy: a.fy + t.fallYes, fn: a.fn + t.fallN, cy: a.cy + t.ctlYes, cn: a.cn + t.ctlN }), { fy: 0, fn: 0, cy: 0, cn: 0 }); return s.cy > 0 ? (s.fy / s.fn) / (s.cy / s.cn) : NaN; };
  const pool = ts.reduce((a, t) => ({ fy: a.fy + t.fallYes, fn: a.fn + t.fallN, cy: a.cy + t.ctlYes, cn: a.cn + t.ctlN }), { fy: 0, fn: 0, cy: 0, cn: 0 });
  const elig = ts.filter((t) => t.fallN >= 3), s = signTest(elig.map((t) => t.fallYes / t.fallN - t.ctlYes / t.ctlN));
  const r = ratio(ts), iv = bootstrap(ts, ratio, 4000, seed);
  const verdict = r >= 1.5 && iv[0] > 1 && s.wins > elig.length / 2 ? 'more common for new loves' : r <= 1 / 1.5 && iv[1] < 1 && s.losses > elig.length / 2 ? 'less common for new loves' : 'no clear difference';
  return `| ${label} | ${f3(pool.fy / pool.fn)} | ${f3(pool.cy / pool.cn)} | ${r.toFixed(2)} ${ci(iv).replace(/(\d\.\d\d)\d/g, '$1')} | ${s.wins}/${s.losses}/${s.ties} | ${pf(s.p)} | ${verdict} |`;
}

// ---------------------------------------------------------------- traits
function traitRow(t: { key: string; group: string; status: string }, i: number) {
  const pairs = (traits.people as any[]).map((p) => [p.a[t.key], p.b[t.key]]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b)) as [number, number][];
  const r = pearson(pairs.map((p) => p[0]), pairs.map((p) => p[1])), rho = spearman(pairs.map((p) => p[0]), pairs.map((p) => p[1]));
  const ir = bootstrap(pairs, (u) => pearson(u.map((p) => p[0]), u.map((p) => p[1])), 4000, 91 + i);
  const irho = bootstrap(pairs, (u) => spearman(u.map((p) => p[0]), u.map((p) => p[1])), 4000, 131 + i);
  const holds = ir[0] >= 0.5 && irho[0] >= 0.5 ? 'stable, interval above 0.5' : ir[0] > 0 && irho[0] > 0 ? 'above zero, not shown to reach 0.5' : 'not shown to be stable';
  return `| ${t.group} | ${t.key} | ${pairs.length} | ${f3(r)} ${ci(ir)} | ${f3(rho)} ${ci(irho)} | ${t.status} | ${holds} |`;
}

// ---------------------------------------------------------------- write
const L: string[] = [];
L.push('# AAS bench report', '');
L.push(`Generated ${new Date().toISOString()} from bench run ${bench.at}${bench.lb ? ' (with the ListenBrainz map)' : ' (quick run, no ListenBrainz map)'}. ${rows.length} people, each split by time: trained on everything before their last 8 weeks, scored on those weeks. Doors and traits use training weeks only.`, '');
L.push('How to read it. Each line pairs a model with a baseline over the same people. Medians come with a 95% percentile bootstrap interval over people (4,000 resamples). The difference is model minus baseline, signed so that positive means the model is better. The sign test is exact and two-sided, with ties dropped. Holm adjusts the sign-test p-values across every comparison in the table. A result survives when the Holm-adjusted p is below 0.05 and the bootstrap interval for the mean difference sits above zero. With 13 people the sign test cannot reach 0.05 unless at least 10 of 13 people agree (10 to 3 gives p = 0.092; 11 to 2 gives p = 0.022), so a small consistent effect can fail here and still be real.', '');
let sec = '';
L.push('## Paired comparisons', '');
L.push('| comparison | n | model median [95%] | baseline median [95%] | median difference [95%] | mean difference [95%] | wins/losses/ties | sign p | Holm p | verdict |', '|---|---|---|---|---|---|---|---|---|---|');
for (const c of cmps) {
  if (c.section !== sec) { sec = c.section; L.push(`| **${sec}** | | | | | | | | | |`); }
  L.push(`| ${c.label} | ${c.n} | ${f3(c.ma)} ${ci(c.ia)} | ${typeof c.b === 'number' ? f3(c.b) : `${f3(c.mb)} ${ci(c.ib)}`} | ${f3(c.md)} ${ci(c.id)} | ${f3(c.mean)} ${ci(c.im)} | ${c.wins}/${c.losses}/${c.ties} | ${pf(c.p)} | ${pf(c.holm)} | ${c.verdict} |`);
}
L.push('', '## Skip model calibration', '');
L.push(`Held-out plays pooled over everyone (${calM.n}). Expected calibration error (ECE) is the play-weighted mean gap between predicted and observed skip rate over 10 equal-width bins. Brier is the mean squared error of the probability.`, '');
L.push('| | Brier | ECE |', '|---|---|---|');
L.push(`| skip model | ${f3(calM.brier)} | ${f3(calM.ece)} |`, `| artist skip rate (baseline) | ${f3(calB.brier)} | ${f3(calB.ece)} |`);
L.push('', `Per person: median Brier ${f3(median(per.map((x) => x.m.brier)))} ${ci(bootstrap(per, (u) => median(u.map((x) => x.m.brier)), 4000, 201))} against ${f3(median(per.map((x) => x.b.brier)))} for the baseline; the model has the lower Brier for ${brierSign.wins} of ${per.length} people (sign p ${pf(brierSign.p)}). Median ECE ${f3(median(per.map((x) => x.m.ece)))} ${ci(bootstrap(per, (u) => median(u.map((x) => x.m.ece)), 4000, 202))} against ${f3(median(per.map((x) => x.b.ece)))}; lower for the model in ${eceSign.wins} of ${per.length} (sign p ${pf(eceSign.p)}).`, '');
L.push('Reliability, skip model (pooled):', '', '| predicted | plays | mean predicted | observed skip rate |', '|---|---|---|---|', relTable(calM), '');
L.push('Reliability, artist skip rate (pooled):', '', '| predicted | plays | mean predicted | observed skip rate |', '|---|---|---|---|', relTable(calB), '');
if (doors) {
  L.push('## Doors: how new loves came in', '');
  L.push('Each new love (an artist played properly 10+ times on 3+ days within 8 weeks of first hearing it) is compared with up to 5 artists the same person tried and dropped, first heard nearest in time. The list fact is also compared with never-played ListenBrainz artists named on as many lists. The ratio is the pooled rate for new loves over the pooled rate for controls, with a bootstrap interval over people. Wins count people (with 3+ new loves) whose new-love rate is higher; the sign test is over them.', '');
  L.push('| door | new loves | controls | ratio [95%] | wins/losses/ties | sign p | verdict |', '|---|---|---|---|---|---|---|');
  const KINDS = ['picked', 'came on after a similar artist', 'next to a favourite', 'came on by itself', 'out of nowhere'];
  KINDS.forEach((k, i) => L.push(doorRow(k, (p) => p.kinds[k], 301 + i)));
  const facetKeys = Object.keys(doors.people[0]?.facets ?? {});
  L.push('', 'Single facts at the first play (an artist can have several):', '', '| fact | new loves | controls | ratio [95%] | wins/losses/ties | sign p | verdict |', '|---|---|---|---|---|---|---|');
  facetKeys.forEach((k, i) => L.push(doorRow(k, (p) => p.facets[k], 401 + i)));
  L.push(doorRow('a recent favourite lists it (vs never-played)', (p) => p.unheard, 499));
  const g = (doors.people as any[]).flatMap((p) => p.growth).filter(Number.isFinite);
  L.push('', `${(doors.people as any[]).reduce((s, p) => s + p.falls, 0)} new loves and ${(doors.people as any[]).reduce((s, p) => s + p.dropped, 0)} tried-and-dropped artists. Median days from first play to the 10th proper play: ${f3(median(g))}.`, '');
}
if (traits) {
  L.push('## Trait stability', '');
  L.push('Each trait is measured on the two halves of every person\'s training time and correlated across people (test-retest). Intervals are a bootstrap over people. The bench rule for showing a trait (both correlations at least 0.5 as point estimates) is in the status column; the last column asks the stricter question of whether the interval itself clears 0.5.', '');
  L.push('| group | trait | n | Pearson r [95%] | Spearman rho [95%] | bench status | interval reading |', '|---|---|---|---|---|---|---|');
  (traits.traits as any[]).forEach((t, i) => L.push(traitRow(t, i)));
  L.push('');
}
if (rank2) {
  const P = rank2.pooled as Record<string, number>, q = (pre: string, k: string) => f3((P[`${pre}:${k}`] ?? 0) / P[`${pre}:n`]);
  L.push('## New loves: the two-stage ranker, pooled', '');
  L.push('Recall pools every new love across people and windows (each love counts once). Coverage is the share of new loves that stage one proposes at all; a love outside the candidates is a miss for the ranker and the lists baseline, while popular ranks every unheard artist other people played.', '');
  L.push('| windows | new loves | candidate coverage | ranker @20 | lists @20 | popular @20 | ranker @50 | lists @50 | popular @50 |', '|---|---|---|---|---|---|---|---|---|');
  for (const [lab, pre] of [['rolling, inside training', 'ts_roll'], ['held-out 8 weeks', 'ts_fall']]) L.push(`| ${lab} | ${P[`${pre}:n`]} | ${q(pre, 'cover')} | ${q(pre, 'model20')} | ${q(pre, 'lists20')} | ${q(pre, 'popular20')} | ${q(pre, 'model50')} | ${q(pre, 'lists50')} | ${q(pre, 'popular50')} |`);
  L.push('', `Ranker weights fitted on everyone, standardised features (reading only): ${Object.entries(rank2.weights as Record<string, number>).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')}.`, '');
}
if (stick) {
  const P = stick.pooled as Record<string, { auc: number; ap: number; brier: number; ece: number; n: number; pos: number } | null>;
  const med = (k: string) => { const u = rows.filter((r) => Number.isFinite(r[k])); return `${f3(median(u.map((r) => r[k])))} ${ci(bootstrap(u, (x) => median(x.map((r) => r[k])), 4000, 601))}`; };
  L.push('## Will a new artist stick', '');
  L.push(`Every artist a person hears for the first time (after their first 12 weeks) is one row; it sticks when it becomes a new love (10+ proper plays on 3+ days within 8 weeks). The model is a logistic regression trained on other people only and reads nothing after the horizon: the end of the first session, or the end of the first 7 days (rows already past the bar by day 7 are left out there). The leakage check re-read ${stick.leak.checked} sampled rows from the history cut at the horizon; ${stick.leak.failed} differed. Per-person numbers are the primary reading. Pooled numbers rank people against each other as well, so a pooled AUC can sit below 0.5 under shuffled labels.`, '');
  L.push('| horizon | rows | new artists | stuck | model | median AUC per person [95%] | median average precision [95%] | median Brier [95%] | median ECE [95%] | pooled AUC | pooled AP |', '|---|---|---|---|---|---|---|---|---|---|---|');
  const labels: Record<string, string> = { full: 'model', noSkip: 'model without skip, start, shuffle and hour features', base: 'own base rate', skip: 'first-play skip', pop: 'popularity', wkCounts: 'first-week play counts', gbt: 'boosted trees (check)', shuffled: 'labels shuffled (null)', shuffledWithin: 'labels shuffled inside each person' };
  for (const [h, tag] of [['session', 'st_s'], ['week', 'st_w']]) for (const [part, pl] of [['roll', 'training period'], ['held', 'held-out 8 weeks']]) {
    const full = P[`${h}:${part}:full`]!;
    for (const key of Object.keys(labels)) {
      const r = P[`${h}:${part}:${key}`];
      if (!r) continue;
      const suf = key === 'full' ? '' : `_${key}`;
      L.push(`| ${h === 'session' ? 'first session' : 'first 7 days'} | ${pl} | ${full.n} | ${full.pos} | ${labels[key]} | ${med(`${tag}_${part}_auc${suf}`)} | ${med(`${tag}_${part}_ap${suf}`)} | ${med(`${tag}_${part}_brier${suf}`)} | ${med(`${tag}_${part}_ece${suf}`)} | ${f3(r.auc)} | ${f3(r.ap)} |`);
    }
  }
  L.push('', 'Feature groups, training period. "Without" drops the group from the model; "alone" is a model on that group only. AUC is the median per person; the change is the median of each person\'s difference from the full model, with its interval.', '');
  L.push('| horizon | group | AUC without | change without [95%] | AUC alone |', '|---|---|---|---|---|');
  for (const [h, tag] of [['session', 'st_s'], ['week', 'st_w']]) for (const key of (stick.specs[h] as string[]).filter((k) => k.startsWith('drop:'))) {
    const g = key.slice(5), dk = `${tag}_roll_auc_${key}`, ok = `${tag}_roll_auc_only:${g}`;
    const u = rows.filter((r) => Number.isFinite(r[dk]) && Number.isFinite(r[`${tag}_roll_auc`]));
    const d = (x: Record<string, number>[]) => median(x.map((r) => r[dk] - r[`${tag}_roll_auc`]));
    L.push(`| ${h === 'session' ? 'first session' : 'first 7 days'} | ${g} | ${f3(median(u.map((r) => r[dk])))} | ${f3(d(u))} ${ci(bootstrap(u, d, 4000, 701))} | ${f3(median(rows.map((r) => r[ok]).filter(Number.isFinite)))} |`);
  }
  L.push('', `Session-horizon weights fitted on everyone, standardised (reading only): ${Object.entries(stick.weights as Record<string, number>).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')}.`, '');
  L.push('Examples from the held-out weeks, scored after the first session, written so the person can check them against their own history:', '', ...(stick.examples as string[]).map((e) => `- ${e}`), '');
}
L.push('## What survives', '');
const surv = cmps.filter((c) => c.verdict === 'survives'), pre = cmps.filter((c) => c.verdict.startsWith('survives before')), weak = cmps.filter((c) => c.verdict === 'weak'), none = cmps.filter((c) => c.verdict === 'no difference' || c.verdict === 'worse than baseline');
L.push(`Survive the adjustment: ${surv.map((c) => `${c.section}: ${c.label}`).join('; ') || 'none'}.`, '');
L.push(`Significant only before adjustment: ${pre.map((c) => `${c.section}: ${c.label}`).join('; ') || 'none'}.`, '');
L.push(`Weak (one test passes, the other does not): ${weak.map((c) => `${c.section}: ${c.label}`).join('; ') || 'none'}.`, '');
L.push(`No difference or worse: ${none.map((c) => `${c.section}: ${c.label}`).join('; ') || 'none'}.`, '');
// round 3, the ListenBrainz-user bench (bench/lb3/analyse.ts), when it has been run
if (existsSync(`${OUT}/REPORT_LB.md`)) L.push('', readFileSync(`${OUT}/REPORT_LB.md`, 'utf8').trimEnd());
writeFileSync(`${OUT}/REPORT.md`, L.join('\n') + '\n');
console.log(`wrote ${OUT}/REPORT.md (${cmps.length} comparisons, ${surv.length} survive)`);

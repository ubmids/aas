// Measured listening traits: values per person, percentiles against the other people, and a
// test-retest check. Each trait is measured on the two halves of a person's training time
// separately (the second half may only use the first to know which artists were already
// familiar); a trait is stable when the halves agree across people (r >= 0.5, Pearson and Spearman).
// `npm run traits` (`--cards` prints one person's card lines).
import { writeFileSync, mkdirSync } from 'node:fs';
import { loadPeople } from './load.js';
import { buildMap } from '../engine/src/model/map.js';
import { measureTraits, percentile, traitLine, TRAITS, TraitKey, TraitValue } from '../engine/src/model/traits.js';

const CARDS = process.argv.includes('--cards');
const STABLE_R = 0.5;
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '  -  ');
const people = loadPeople();
const map = buildMap(people.map((p) => p.plays.slice(0, p.n)));

const byKey = (ts: TraitValue[]) => new Map(ts.map((t) => [t.key, t]));
const full = new Map<string, Map<TraitKey, TraitValue>>(), halfA = new Map<string, Map<TraitKey, TraitValue>>(), halfB = new Map<string, Map<TraitKey, TraitValue>>();
for (const P of people) {
  const train = P.plays.slice(0, P.n), mid = (train[0].t + train[train.length - 1].t) / 2;
  const a = train.filter((p) => p.t < mid), b = train.filter((p) => p.t >= mid);
  full.set(P.name, byKey(measureTraits(train, map)));
  halfA.set(P.name, byKey(measureTraits(a, map)));
  halfB.set(P.name, byKey(measureTraits(b, map, a)));
  process.stderr.write(`${P.name.slice(0, 24).padEnd(24)} ${train.length} plays, halves ${a.length} / ${b.length}\n`);
}

function pearson(x: number[], y: number[]) {
  const n = x.length, mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}
const ranks = (x: number[]) => { const o = x.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]), r = Array(x.length); o.forEach(([, i], k) => (r[i] = k)); return r; };
const spearman = (x: number[], y: number[]) => pearson(ranks(x), ranks(y));

console.log(`\nTRAITS over ${people.length} people (training weeks only)`);
console.log('test-retest: the same trait on the two halves of each person\'s training time');
const stable = new Map<TraitKey, boolean>();
for (const d of TRAITS) {
  const pairs = people.map((p) => [halfA.get(p.name)!.get(d.key)?.value ?? NaN, halfB.get(p.name)!.get(d.key)?.value ?? NaN]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
  const x = pairs.map((p) => p[0]), y = pairs.map((p) => p[1]);
  const r = pairs.length >= 5 ? pearson(x, y) : NaN, rho = pairs.length >= 5 ? spearman(x, y) : NaN;
  const vals = people.map((p) => full.get(p.name)!.get(d.key)?.value ?? NaN).filter(Number.isFinite).sort((a, b) => a - b);
  const ok = r >= STABLE_R && rho >= STABLE_R, half = r >= STABLE_R || rho >= STABLE_R;
  stable.set(d.key, ok);
  const verdict = ok ? 'stable' : half ? 'flagged' : 'dropped';
  if (verdict !== d.status) console.log(`  NOTE: ${d.key} measures ${verdict} but TRAITS says ${d.status}`);
  console.log(`  ${d.group.padEnd(16)} ${d.key.padEnd(12)} n ${String(pairs.length).padEnd(3)} r ${f3(r)}  rho ${f3(rho)}  ${verdict.padEnd(7)} | bench range ${f3(vals[0])} to ${f3(vals[vals.length - 1])}, median ${f3(vals[vals.length >> 1])}`);
}

// how much the traits overlap: correlations between traits across people (full training)
console.log('overlap between traits (Spearman across people, |rho| >= 0.6 shown):');
for (let i = 0; i < TRAITS.length; i++) for (let j = i + 1; j < TRAITS.length; j++) {
  const pairs = people.map((p) => [full.get(p.name)!.get(TRAITS[i].key)?.value ?? NaN, full.get(p.name)!.get(TRAITS[j].key)?.value ?? NaN]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
  if (pairs.length < 5) continue;
  const rho = spearman(pairs.map((p) => p[0]), pairs.map((p) => p[1]));
  if (Math.abs(rho) >= 0.6) console.log(`  ${TRAITS[i].key} ~ ${TRAITS[j].key}: ${f3(rho)}`);
}

console.log('values and percentiles (against the other people):');
for (const P of people) {
  const mine = full.get(P.name)!;
  const cells = TRAITS.map((d) => {
    const v = mine.get(d.key)?.value ?? NaN, others = people.filter((q) => q !== P).map((q) => full.get(q.name)!.get(d.key)?.value ?? NaN);
    return `${d.key} ${f3(v)} (p${Number.isFinite(percentile(v, others)) ? Math.round(100 * percentile(v, others)) : '-'})`;
  });
  console.log(`  ${P.name.slice(0, 20).padEnd(20)} ${cells.join('  ')}`);
}

if (CARDS) {
  const P = people[0];
  console.log(`\ncards for one person (${P.name}):`);
  for (const d of TRAITS) {
    const t = full.get(P.name)!.get(d.key);
    if (!t || !Number.isFinite(t.value)) continue;
    const others = people.filter((q) => q !== P).map((q) => full.get(q.name)!.get(d.key)?.value ?? NaN);
    const line = traitLine(t, percentile(t.value, others));
    console.log(line ? `  - ${line}` : `  (${d.key} not shown: ${d.status})`);
  }
}

mkdirSync('bench/out', { recursive: true });
const vals = (m: Map<string, Map<TraitKey, TraitValue>>, name: string) => Object.fromEntries(TRAITS.map((d) => [d.key, m.get(name)!.get(d.key)?.value ?? null]));
writeFileSync('bench/out/traits.json', JSON.stringify({ at: new Date().toISOString(), traits: TRAITS.map((d) => ({ key: d.key, group: d.group, status: d.status })), people: people.map((p) => ({ name: p.name, full: vals(full, p.name), a: vals(halfA, p.name), b: vals(halfB, p.name) })) }));
process.stderr.write('wrote bench/out/traits.json\n');

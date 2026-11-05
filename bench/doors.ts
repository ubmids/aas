// Bench for the look-back "where your new artists came from". It is not a forecast, so there is no
// held-out split: the question is whether a door is real. Every fall is compared with
//   (a) artists the same person tried and dropped (<= 2 proper plays ever), first heard nearest in time,
//   (b) for the list facts, ListenBrainz artists the person never played, matched on how many lists name them.
// A door counts as evidence when it is clearly more common for falls: pooled rate ratio >= 1.5, the
// 95% bootstrap interval over people above 1, and higher for falls in most people who have both.
// `npm run doors` prints the table; `--story` also prints each person's summary and a few stories.
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { loadPeople } from './load.js';
import { listsOf, newArtists, readDoor, matchDropped, FACETS, doorKind, DoorKind, Door, recentFavs, summarise, story, Lists } from '../engine/src/model/doors.js';
import { rng } from '../engine/src/model/linalg.js';

const STORY = process.argv.includes('--story');
const LB_FILE = 'bench/data/lb/table.json';
const L: Lists | null = existsSync(LB_FILE) ? listsOf(JSON.parse(readFileSync(LB_FILE, 'utf8'))) : null;
if (!L) process.stderr.write('no ListenBrainz table: list facts are skipped\n');
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '  -  ');
const hash = (s: string) => [...s].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 7);

interface Tally { fallYes: number; fallN: number; ctlYes: number; ctlN: number } // ctl counts are per-fall averages
const people = loadPeople();
type PersonOut = { name: string; facets: Record<string, Tally>; kinds: Record<string, Tally>; unheard: Tally; falls: Door[]; dropped: Door[]; matched: Map<Door, Door[]>; ranksFall: number[]; ranksCtl: number[] };
const outs: PersonOut[] = [];
const KINDS: DoorKind[] = ['picked', 'came on after a similar artist', 'next to a favourite', 'came on by itself', 'out of nowhere'];
const blank = (): Tally => ({ fallYes: 0, fallN: 0, ctlYes: 0, ctlN: 0 });

// popularity buckets of ListenBrainz nodes for (b): log2 of how many lists name the artist
const buckets = new Map<number, number[]>();
if (L) L.listedBy.forEach((c, i) => { if (c) { const b = Math.round(2 * Math.log2(c)); (buckets.get(b) ?? buckets.set(b, []).get(b)!).push(i); } });

for (const P of people) {
  const plays = P.plays, everPlayed = new Set(plays.map((p) => p.artist));
  const doors = newArtists(plays).map(({ i, kind, daysTo10 }) => readDoor(plays, i, kind, L, daysTo10));
  const falls = doors.filter((d) => d.kind === 'fall'), dropped = doors.filter((d) => d.kind === 'dropped');
  const matched = matchDropped(doors);
  const o: PersonOut = { name: P.name, facets: {}, kinds: {}, unheard: blank(), falls, dropped, matched, ranksFall: [], ranksCtl: [] };
  const tally = (bag: Record<string, Tally>, key: string, f: (d: Door) => boolean | null) => {
    const t = (bag[key] ??= blank());
    for (const [fd, ctl] of matched) {
      const v = f(fd);
      if (v === null) continue;
      const cv = ctl.map(f).filter((x): x is boolean => x !== null);
      if (!cv.length) continue;
      t.fallN++; if (v) t.fallYes++;
      t.ctlN++; t.ctlYes += cv.filter(Boolean).length / cv.length;
    }
  };
  for (const [k, f] of Object.entries(FACETS)) tally(o.facets, k, f);
  for (const k of KINDS) tally(o.kinds, k, (d) => (d.mode === 'unknown' ? null : doorKind(d) === k));
  // (b) never-played ListenBrainz artists of matched list count, read at the fall's first play
  if (L) {
    const r = rng(hash(P.name));
    for (const fd of falls) {
      const ix = L.ix.get(fd.artist);
      if (ix === undefined || !L.listedBy[ix]) continue;
      const pool = (buckets.get(Math.round(2 * Math.log2(L.listedBy[ix]))) ?? []).filter((j) => !everPlayed.has(L.nodes[j]));
      if (pool.length < 5) continue;
      const i = plays.findIndex((p) => p.t === fd.first && p.artist === fd.artist);
      const favs = [...recentFavs(plays, i).keys()];
      const bestRank = (a: string) => favs.reduce<number | null>((b, f) => { const x = L.rank(f, a); return x !== null && (b === null || x < b) ? x : b; }, null);
      let yes = 0;
      for (let k = 0; k < 5; k++) { const c = L.nodes[pool[Math.floor(r() * pool.length)]], br = bestRank(c); if (br !== null) { yes++; o.ranksCtl.push(br); } }
      o.unheard.fallN++; if (fd.favRank !== null) { o.unheard.fallYes++; o.ranksFall.push(fd.favRank); }
      o.unheard.ctlN++; o.unheard.ctlYes += yes / 5;
    }
  }
  outs.push(o);
  process.stderr.write(`${P.name.slice(0, 24).padEnd(24)} falls ${falls.length} (${matched.size} matched), dropped ${dropped.length}\n`);
}

/** Pooled rates, per-person wins and a bootstrap interval over people for the rate ratio. */
function compare(get: (o: PersonOut) => Tally | undefined) {
  const ts = outs.map(get).filter((t): t is Tally => !!t && t.fallN > 0);
  const pool = (xs: Tally[]) => { const s = xs.reduce((a, t) => ({ fy: a.fy + t.fallYes, fn: a.fn + t.fallN, cy: a.cy + t.ctlYes, cn: a.cn + t.ctlN }), { fy: 0, fn: 0, cy: 0, cn: 0 }); return { f: s.fy / s.fn, c: s.cy / s.cn, n: s.fn }; };
  const all = pool(ts), ratio = all.f / all.c;
  const r = rng(99), rs: number[] = [];
  for (let b = 0; b < 2000; b++) { const s = pool(Array.from({ length: ts.length }, () => ts[Math.floor(r() * ts.length)])); rs.push(s.c > 0 ? s.f / s.c : Infinity); }
  rs.sort((a, b) => a - b);
  const lo = rs[Math.floor(0.025 * rs.length)], hi = rs[Math.floor(0.975 * rs.length)];
  const eligible = ts.filter((t) => t.fallN >= 3), wins = eligible.filter((t) => t.fallYes / t.fallN > t.ctlYes / t.ctlN).length;
  const evidence = ratio >= 1.5 && lo > 1 && wins > eligible.length / 2;
  const below = ratio <= 1 / 1.5 && hi < 1 && eligible.length - wins > eligible.length / 2;
  return { ...all, ratio, lo, hi, wins: `${wins}/${eligible.length}`, verdict: evidence ? 'MORE for falls' : below ? 'LESS for falls' : 'no difference', evidence };
}
const row = (label: string, c: ReturnType<typeof compare>) =>
  `  ${label.padEnd(32)} falls ${f3(c.f)}  controls ${f3(c.c)}  ratio ${c.ratio.toFixed(2)} [${c.lo.toFixed(2)}, ${c.hi.toFixed(2)}]  wins ${c.wins.padEnd(5)} n ${String(c.n).padEnd(4)} ${c.verdict}`;

const nFalls = outs.reduce((s, o) => s + o.falls.length, 0), nMatched = outs.reduce((s, o) => s + o.matched.size, 0);
console.log(`\nDOORS over ${outs.length} people: ${nFalls} falls (${nMatched} with dropped controls), ${outs.reduce((s, o) => s + o.dropped.length, 0)} tried-and-dropped artists`);
console.log('facets, falls vs tried-and-dropped (same person, 5 nearest first plays):');
for (const k of Object.keys(FACETS)) console.log(row(k, compare((o) => o.facets[k])));
console.log('door kinds (one per artist), falls vs tried-and-dropped:');
const evidence = new Set<DoorKind>();
for (const k of KINDS) { const c = compare((o) => o.kinds[k]); if (c.evidence) evidence.add(k); console.log(row(k, c)); }
if (L) {
  console.log('list facts, falls vs never-played ListenBrainz artists of matched list count:');
  console.log(row('a recent favourite lists it', compare((o) => o.unheard)));
  const med = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
  console.log(`  median best rank when listed: falls ${med(outs.flatMap((o) => o.ranksFall))}, never-played ${med(outs.flatMap((o) => o.ranksCtl))}`);
}
const growth = outs.flatMap((o) => o.falls.map((d) => d.daysTo10!)).sort((a, b) => a - b);
console.log(`growth: days from first play to the 10th proper play, median ${growth[growth.length >> 1]?.toFixed(1)}, quartiles ${growth[growth.length >> 2]?.toFixed(1)} to ${growth[(3 * growth.length) >> 2]?.toFixed(1)}`);
console.log(`door kinds that count as evidence: ${[...evidence].join(', ') || 'none'}`);
mkdirSync('bench/out', { recursive: true });
writeFileSync('bench/out/doors.json', JSON.stringify({ at: new Date().toISOString(), people: outs.map((o) => ({ name: o.name, falls: o.falls.length, dropped: o.dropped.length, facets: o.facets, kinds: o.kinds, unheard: o.unheard, growth: o.falls.map((d) => d.daysTo10) })) }));
process.stderr.write('wrote bench/out/doors.json\n');

if (STORY) for (const o of outs) {
  const s = summarise(o.falls, evidence, o.dropped);
  console.log(`\n${o.name}\n  ${s.line}\n  kinds: ${s.kinds.map(([k, n]) => `${k} ${n}`).join(', ')}`);
  for (const d of [...o.falls].sort((a, b) => b.first - a.first).slice(0, 3)) console.log(`  - ${story(d)}`);
}

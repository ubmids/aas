// Builds the ListenBrainz artist table for the bench (bench/data/lb/, gitignored).
// 1. names -> MusicBrainz artist ids: labs acr-lookup on each artist's most played tracks
//    (100 pairs per request, exact credit match after ListenBrainz's own normalisation), then a
//    MusicBrainz artist search (1 request a second) for the misses with 10+ proper plays.
// 2. similar artists (session based, top 100) for every seed (an artist with 3+ proper plays in
//    anyone's training weeks), then for every artist on those lists (--one-hop skips this).
//    One request per artist, 4 at a time, every raw response cached.
// 3. table.json: nodes (bench names where known, else the ListenBrainz name) and every list.
// Run: npx tsx bench/scripts/lb_build.ts   (resumable; rerun picks up from the cache)
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { loadPeople } from '../load.js';
import { isSkip } from '../../engine/src/parse/spotify.js';

const DIR = 'bench/data/lb', UA = 'aas-bench/0.1', LABS = 'https://labs.api.listenbrainz.org';
export const ALGORITHM = 'session_based_days_7500_session_300_contribution_5_threshold_10_limit_100_filter_True_skip_30';
mkdirSync(DIR, { recursive: true });
// the list cache is kept gzipped between runs (143 MB plain); unpack it to resume
if (existsSync(`${DIR}/similar.jsonl.gz`) && !existsSync(`${DIR}/similar.jsonl`)) writeFileSync(`${DIR}/similar.jsonl`, gunzipSync(readFileSync(`${DIR}/similar.jsonl.gz`)));
const log = (s: string) => process.stderr.write(s + '\n');
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function get(url: string, body?: unknown): Promise<unknown> {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url, body === undefined ? { headers: { 'user-agent': UA } }
        : { method: 'POST', headers: { 'user-agent': UA, 'content-type': 'application/json' }, body: JSON.stringify(body) });
      if (res.status === 429 || res.status >= 500) throw new Error(`HTTP ${res.status}`);
      if (res.status === 400 || res.status === 404) return [];
      return await res.json();
    } catch (e) {
      if (attempt >= 6) throw e;
      await sleep(2000 * 2 ** attempt);
    }
  }
}

// ---- artists, their top tracks, and the seeds
const people = loadPeople();
const tracks = new Map<string, Map<string, number>>(), seeds = new Set<string>();
for (const p of people) {
  const proper = new Map<string, number>();
  p.plays.forEach((x, i) => {
    if (isSkip(x)) return;
    const m = tracks.get(x.artist) ?? tracks.set(x.artist, new Map()).get(x.artist)!;
    m.set(x.track, (m.get(x.track) ?? 0) + 1);
    if (i < p.n) proper.set(x.artist, (proper.get(x.artist) ?? 0) + 1);
  });
  for (const [a, c] of proper) if (c >= 3) seeds.add(a);
}
log(`${tracks.size} artists with a proper play, ${seeds.size} seeds`);

// ---- 1. names -> artist ids
const norm = (s: string) => s.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
const acrFile = `${DIR}/acr.jsonl`, asked = new Set<string>();
if (existsSync(acrFile)) for (const l of readFileSync(acrFile, 'utf8').split('\n')) if (l) asked.add(JSON.parse(l).q.join('\u0000'));
// Spotify titles carry suffixes MusicBrainz titles do not: " - 2005 Remaster", "(feat. X)", "[Live]"
const clean = (t: string) => t.replace(/\s+-\s+.*$/, '').replace(/\s*[([](feat|ft|with|from|prod)\b[^)\]]*[)\]]/gi, '').replace(/\s*\[[^\]]*\]/g, '').trim();
const queries: [string, string][] = [];
for (const [a, m] of tracks) {
  const top = [...m].sort((x, y) => y[1] - x[1]).slice(0, 6).map(([t]) => t);
  const qs = [...new Set([...top.slice(0, 3), ...top.map(clean).filter(Boolean)])];
  for (const t of qs) if (!asked.has(a + '\u0000' + t)) queries.push([a, t]);
}
log(`acr-lookup: ${queries.length} pairs to ask`);
for (let i = 0; i < queries.length; i += 100) {
  const batch = queries.slice(i, i + 100);
  const res = (await get(`${LABS}/acr-lookup/json`, batch.map(([a, t]) => ({ artist_credit_name: a, recording_name: t })))) as Record<string, unknown>[];
  const byIx = new Map(res.map((r) => [Number(r.index), r]));
  appendFileSync(acrFile, batch.map((q, j) => JSON.stringify({ q, r: byIx.get(j) ?? null })).join('\n') + '\n');
  if ((i / 100) % 20 === 0) log(`  ${i + batch.length}/${queries.length}`);
}
const votes = new Map<string, Map<string, number>>();
for (const l of readFileSync(acrFile, 'utf8').split('\n')) {
  if (!l) continue;
  const { q, r } = JSON.parse(l) as { q: [string, string]; r: { artist_credit_name: string; artist_mbids: string[] } | null };
  if (!r || r.artist_mbids.length !== 1) continue;
  // an exact credit match counts fully; a different spelling (often another script, Hitorie and
  // ヒトリエ) counts only as a weak vote and needs two tracks agreeing
  const w = norm(r.artist_credit_name) === norm(q[0]) ? 2 : 1;
  const v = votes.get(q[0]) ?? votes.set(q[0], new Map()).get(q[0])!;
  v.set(r.artist_mbids[0], (v.get(r.artist_mbids[0]) ?? 0) + w);
}
const mbidOf = new Map<string, string>();
for (const [a, v] of votes) { const [m, w] = [...v].sort((x, y) => y[1] - x[1])[0]; if (w >= 2) mbidOf.set(a, m); }
// 1b. artists the track lookup missed, with 10+ proper plays (enough to ever be a fall): MusicBrainz
// artist search, 1 request a second, accepted only for one clear exact-name match
const searchFile = `${DIR}/mb_search.jsonl`, searched = new Set<string>();
if (existsSync(searchFile)) for (const l of readFileSync(searchFile, 'utf8').split('\n')) if (l) searched.add(JSON.parse(l).a);
const total = (a: string) => [...(tracks.get(a)?.values() ?? [])].reduce((s, v) => s + v, 0);
const toSearch = [...tracks.keys()].filter((a) => !mbidOf.has(a) && !searched.has(a) && total(a) >= 10);
function applySearch() {
  if (!existsSync(searchFile)) return;
  for (const l of readFileSync(searchFile, 'utf8').split('\n')) {
    if (!l) continue;
    const { a, r } = JSON.parse(l) as { a: string; r: [string, string, number, string[]][] };
    if (mbidOf.has(a)) continue;
    const exact = r.filter(([, n, sc, al]) => sc >= 90 && (norm(n) === norm(a) || al.some((x) => norm(x) === norm(a))));
    if (exact.length === 1) mbidOf.set(a, exact[0][0]);
  }
}
applySearch();
log(`mapped ${mbidOf.size}/${tracks.size} artists, ${[...seeds].filter((a) => mbidOf.has(a)).length}/${seeds.size} seeds`);

async function mbSearch() {
  log(`musicbrainz search: ${toSearch.length} artists to ask`);
  for (const [j, a] of toSearch.entries()) {
    const q = encodeURIComponent(a.replace(/[":()\[\]{}^~*?\\/!+-]/g, ' '));
    const res = (await get(`https://musicbrainz.org/ws/2/artist?query=${q}&fmt=json&limit=5`).catch(() => null)) as { artists?: { id: string; name: string; score: number; 'sort-name'?: string; aliases?: { name: string }[] }[] } | null;
    // the sort name "Kitani, Tatsuya" read forwards is how Spotify often spells a Japanese artist
    const forwards = (x: string) => x.split(', ').reverse().join(' ');
    if (!res) continue; // still busy after the retries: not recorded, asked again next run
    appendFileSync(searchFile, JSON.stringify({ a, r: (res.artists ?? []).map((x) => [x.id, x.name, x.score, [...(x.aliases ?? []).map((y) => y.name), ...(x['sort-name'] ? [forwards(x['sort-name'])] : [])]]) }) + '\n');
    if (j % 200 === 0) log(`  ${j}/${toSearch.length}`);
    await sleep(1100);
  }
}

// ---- 2. similar artists for every mapped seed, then for every artist the seeds list (one hop out,
// so an artist's place never depends on whether someone in the bench played it)
const simFile = `${DIR}/similar.jsonl`, lists = new Map<string, [string, string, number][]>();
const readLists = () => { if (existsSync(simFile)) for (const l of readFileSync(simFile, 'utf8').split('\n')) if (l) { const { mbid, list } = JSON.parse(l); lists.set(mbid, list); } };
async function fetchLists(want: string[]) {
  const todo = [...new Set(want)].filter((m) => !lists.has(m));
  log(`similar-artists: ${todo.length} to fetch`);
  let k = 0;
  const worker = async () => {
    while (k < todo.length) {
      const mbid = todo[k++];
      const res = (await get(`${LABS}/similar-artists/json?artist_mbids=${mbid}&algorithm=${ALGORITHM}`)) as { artist_mbid: string; name: string; score: number }[];
      const list = res.map((x) => [x.artist_mbid, x.name, x.score] as [string, string, number]);
      lists.set(mbid, list);
      appendFileSync(simFile, JSON.stringify({ mbid, list }) + '\n');
      if (k % 500 === 0) log(`  ${k}/${todo.length}`);
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
}
readLists();
const seedIds = [...seeds].map((a) => mbidOf.get(a)).filter((m): m is string => !!m);
const lbFetch = async () => {
  await fetchLists(seedIds);
  if (!process.argv.includes('--one-hop')) await fetchLists(seedIds.flatMap((m) => (lists.get(m) ?? []).map(([x]) => x)));
};
// the search is throttled hard, so it runs alongside; artists it finds get their lists on the next run
await Promise.all([lbFetch(), process.argv.includes('--no-search') ? Promise.resolve() : mbSearch()]);
applySearch();
log(`after search: mapped ${mbidOf.size}/${tracks.size} artists; ${[...seeds].filter((a) => { const m = mbidOf.get(a); return m && !lists.has(m); }).length} seeds still without a list (rerun to fetch)`);

// ---- 3. the table: nodes (bench spelling where known) and every fetched list, by node index
const nameOf = new Map<string, string>();
for (const [a, m] of mbidOf) { const cur = nameOf.get(m); if (!cur || total(a) > total(cur)) nameOf.set(m, a); }
const lbName = new Map<string, string>();
for (const list of lists.values()) for (const [m, n] of list) lbName.set(m, n);
const nodes: string[] = [], nodeIx = new Map<string, number>();
const node = (mbid: string) => {
  let i = nodeIx.get(mbid);
  if (i === undefined) { i = nodes.length; nodeIx.set(mbid, i); nodes.push(nameOf.get(mbid) ?? `lb:${lbName.get(mbid) ?? mbid}`); }
  return i;
};
const outLists: Record<string, [number, number][]> = {};
for (const [m, list] of lists) if (list.length) outLists[node(m)] = list.map(([x, , sc]) => [node(x), sc] as [number, number]);
const out = { algorithm: ALGORITHM, nodes, lists: outLists, mapped: Object.fromEntries(mbidOf) };
writeFileSync(`${DIR}/table.json`, JSON.stringify(out));
log(`table: ${nodes.length} nodes, ${Object.keys(outLists).length} lists`);

// Fetches ListenBrainz similar-artist lists (the same algorithm as bench/scripts/lb_build.ts) for
// the artists the ListenBrainz-user bench needs: each person's 200 most played artists in their
// training weeks (everything before their last 8 weeks), by MusicBrainz id, where the Spotify
// bench's cache (bench/data/lb/similar.jsonl.gz) has no list yet. Raw lists are appended to
// bench/data/lbusers/similar_extra.jsonl (gitignored) in the same {mbid, list} shape.
// Resumable: rerun to pick up where it stopped. Every request has a timeout and a capped number of
// retries, so a network drop slows the run down but cannot hang it.
// Run: npx tsx bench/lbusers/lists.ts [concurrency]
import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { lbUserIds, loadLbUser } from './load.js';

const UA = 'aas-bench/0.1', LABS = 'https://labs.api.listenbrainz.org';
const ALGORITHM = 'session_based_days_7500_session_300_contribution_5_threshold_10_limit_100_filter_True_skip_30';
const OUT = 'bench/data/lbusers/similar_extra.jsonl', BASE = 'bench/data/lb/similar.jsonl.gz';
const TOP = 200, HOLD = 8 * 7 * 86_400_000, CONC = Number(process.argv[2] ?? 4);
const log = (s: string) => process.stderr.write(`${new Date().toISOString()} ${s}\n`);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const have = new Set<string>();
for (const l of gunzipSync(readFileSync(BASE)).toString().split('\n')) if (l) have.add(JSON.parse(l).mbid);
if (existsSync(OUT)) for (const l of readFileSync(OUT, 'utf8').split('\n')) if (l) { try { have.add(JSON.parse(l).mbid); } catch { /* a line cut by a crash */ } }

const want = new Map<string, number>();
for (const id of lbUserIds()) {
  const ps = loadLbUser(id), cut = ps[ps.length - 1].t - HOLD, c = new Map<string, number>();
  for (const p of ps) if (p.t < cut && p.artistMbids?.length === 1) c.set(p.artistMbids[0], (c.get(p.artistMbids[0]) ?? 0) + 1);
  for (const [m] of [...c].sort((a, b) => b[1] - a[1]).slice(0, TOP)) if (!have.has(m)) want.set(m, (want.get(m) ?? 0) + 1);
}
const todo = [...want.keys()].sort((a, b) => want.get(b)! - want.get(a)! || (a < b ? -1 : 1));
log(`${have.size} lists cached, ${todo.length} to fetch`);

async function get(mbid: string): Promise<[string, string, number][] | null> {
  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      const res = await fetch(`${LABS}/similar-artists/json?artist_mbids=${mbid}&algorithm=${ALGORITHM}`, { headers: { 'user-agent': UA }, signal: AbortSignal.timeout(30_000) });
      if (res.status === 400 || res.status === 404) return [];
      if (res.status === 429 || res.status >= 500) throw new Error(`HTTP ${res.status}`);
      const body = (await res.json()) as { artist_mbid: string; name: string; score: number }[];
      return Array.isArray(body) ? body.map((x) => [x.artist_mbid, x.name, x.score]) : [];
    } catch {
      await sleep(2000 * 2 ** attempt);
    }
  }
  return null; // gave up; the next run tries again
}

let k = 0, done = 0, failed = 0;
async function worker() {
  while (k < todo.length) {
    const mbid = todo[k++], list = await get(mbid);
    if (list === null) { failed++; continue; }
    appendFileSync(OUT, JSON.stringify({ mbid, list }) + '\n');
    if (++done % 500 === 0) log(`${done}/${todo.length} fetched, ${failed} gave up`);
  }
}
await Promise.all(Array.from({ length: CONC }, worker));
log(`finished: ${done} fetched, ${failed} gave up (rerun to retry them)`);

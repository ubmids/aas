// Step 2: download each qualifying user's most recent listens, newest first, 1,000 per page.
// Raw pages are trimmed (cover art and streaming links dropped) and cached gzipped under
// bench/data/lbusers/raw/<pseudonym>/. Resumable per user: state in raw/<pseudonym>/state.json.
// Usage: npx tsx bench/lbusers/fetch.ts [users=360] [cap=60000] [workers=3]
//        LB_SET=fresh npx tsx bench/lbusers/fetch.ts 400 60000 3   (the confirmation set)
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync, gzipSync } from 'node:zlib';
import { DATA, RAW, RAW_MAIN, SET, get, pseud, readJson, writeJson } from './lib.js';

const nUsers = Number(process.argv[2] ?? 360);
const CAP = Number(process.argv[3] ?? 60_000);   // most recent listens per user
const WORKERS = Number(process.argv[4] ?? 3);
const DISK_CAP = 2e9;                            // stop starting new users past ~2 GB of raw pages
const MIN_SPAN = 730 * 86400;                    // heavy listeners: go past CAP until 2 years are covered...
const HARD_CAP = 150_000;                        // ...but never past this many listens

const cands = readJson<Record<string, any>>(join(DATA, 'candidates.json'), {});
// deterministic order: pseudonym, so the sample does not follow the graph walk
// LB_SET=fresh: only users the first download never started (no folder under raw/)
const users = Object.values(cands).filter((c) => c.ok).map((c) => c.user as string)
  .filter((u) => SET !== 'fresh' || !existsSync(join(RAW_MAIN, pseud(u))))
  .sort((a, b) => (pseud(a) < pseud(b) ? -1 : 1)).slice(0, nUsers);

function trim(l: any) {
  const m = l.track_metadata ?? {}, a = m.additional_info ?? {}, mm = m.mbid_mapping ?? null;
  return {
    ts: l.listened_at, ins: l.inserted_at, msid: l.recording_msid,
    artist: m.artist_name, track: m.track_name, release: m.release_name ?? null,
    client: a.submission_client ?? null, service: a.music_service ?? a.music_service_name ?? null,
    from: a.listening_from ?? null, player: a.media_player ?? null,
    dur: a.duration_ms ?? (a.duration ? a.duration * 1000 : null), spotify: a.spotify_id ?? null,
    artistMbids: mm?.artist_mbids ?? a.artist_mbids ?? null,
    recMbid: mm?.recording_mbid ?? a.recording_mbid ?? null,
    relMbid: mm?.release_mbid ?? a.release_mbid ?? null,
  };
}

function diskUsed(): number {
  let total = 0;
  for (const d of readdirSync(RAW)) {
    const p = join(RAW, d);
    if (statSync(p).isDirectory()) for (const f of readdirSync(p)) total += statSync(join(p, f)).size;
  }
  return total;
}

async function fetchUser(user: string) {
  const id = pseud(user), dir = join(RAW, id);
  mkdirSync(dir, { recursive: true });
  const stFile = join(dir, 'state.json');
  const st = readJson(stFile, { got: 0, maxTs: null as number | null, done: false, pages: 0, newest: null as number | null, ended: false });
  if (st.newest == null && st.pages > 0) { // states written before `newest` existed
    const first = JSON.parse(gunzipSync(readFileSync(join(dir, 'p0000.json.gz'))).toString());
    st.newest = Math.max(...first.map((l: any) => l.ts));
  }
  // a user stopped by CAP whose listens cover under 2 years gets extended (resumes from maxTs)
  if (st.done && !st.ended && st.got >= CAP && st.got < HARD_CAP && st.newest && st.maxTs && st.newest - st.maxTs < MIN_SPAN) st.done = false;
  if (st.done) return st;
  while (!st.done) {
    const q = st.maxTs ? `&max_ts=${st.maxTs}` : '';
    const r = await get(`/user/${encodeURIComponent(user)}/listens?count=1000${q}`);
    const ls: any[] = r?.payload?.listens ?? [];
    if (!ls.length) { st.done = true; st.ended = true; break; }
    st.newest ??= Math.max(...ls.map((l) => l.listened_at));
    writeFileSync(join(dir, `p${String(st.pages).padStart(4, '0')}.json.gz`), gzipSync(JSON.stringify(ls.map(trim))));
    st.pages++; st.got += ls.length;
    // max_ts is exclusive; step to the oldest timestamp + 1 so listens sharing it are not lost
    // (the overlap is removed when normalising, by timestamp + msid)
    const oldest = Math.min(...ls.map((l) => l.listened_at));
    const next = ls.length < 1000 ? null : oldest + 1;
    const covered = st.newest - oldest >= MIN_SPAN;
    if (next === null || next === st.maxTs) { st.done = true; st.ended = true; }
    else if (st.got >= HARD_CAP || (st.got >= CAP && covered)) st.done = true;
    else st.maxTs = next === st.maxTs ? oldest : next;
    if (st.maxTs !== null && st.maxTs === oldest + 1 && ls.every((l) => l.listened_at === oldest)) st.maxTs = oldest; // a whole page on one second
    writeJson(stFile, st);
  }
  writeJson(stFile, st);
  return st;
}

let i = 0, done = 0, stoppedForDisk = false;
async function worker() {
  while (i < users.length) {
    if (diskUsed() > DISK_CAP) { stoppedForDisk = true; return; }
    const u = users[i++];
    try {
      await fetchUser(u);
      done++;
      if (done % 10 === 0) console.log(`${new Date().toISOString()} ${done}/${users.length} users, ${(diskUsed() / 1e6).toFixed(0)} MB`);
    } catch (e) {
      // never log the username: error messages carry the request path
      const msg = (e as Error).message.split(encodeURIComponent(u)).join('<user>').split(u).join('<user>');
      console.log(`user ${pseud(u)} failed: ${msg}`);
    }
  }
}
await Promise.all(Array.from({ length: WORKERS }, worker));
console.log(`finished ${done} users${stoppedForDisk ? ' (stopped at the disk cap)' : ''}, ${(diskUsed() / 1e6).toFixed(0)} MB`);

// Step 3: turn each user's raw pages into one clean, time-ordered play list, drop duplicate
// scrobbles and impossible timestamps, and exclude accounts that look imported-in-bulk or automated.
// Writes bench/data/lbusers/plays/<pseudonym>.json.gz and bench/data/lbusers/summary.json
// (with LB_SET=fresh: plays_fresh/ and summary_fresh.json).
// Usage: npx tsx bench/lbusers/normalise.ts
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync, gzipSync } from 'node:zlib';
import { PLAYS, RAW, SUMMARY, readJson, writeJson } from './lib.js';
import type { LbPlay } from './load.js';

const DAY = 86400;
const EARLIEST = Date.UTC(2002, 0, 1) / 1000;  // before Audioscrobbler: clock or import errors
const DUP_WINDOW = 30;                          // same track again within 30 s = one listen sent twice
export const RULES = {
  minListens: 5_000,        // after cleaning
  minSpanDays: 365,         // first to last kept listen
  minActiveWeeks: 26,       // weeks with at least one listen
  maxSharedTs: 0.10,        // share of listens stamped on a second shared with 3+ others (bulk import)
  maxFastGaps: 0.25,        // share of gaps under 30 s between different tracks (faster than listening)
  maxHeavyDays: 0.02,       // share of active days with 500+ listens
  maxTopTrack: 0.30,        // one track over 30% of all listens (loops, sleep sounds, bots)
  maxRegularGap: 0.30,      // one exact gap length over 30% of gaps under an hour (scripted pacing)
  // added 5 Nov 2026 (statements round 2) after an account playing one artist only slipped through:
  maxTopArtist: 0.75,       // one artist over 75% of all listens
  minArtists: 50,           // fewer than 50 different artists in the whole history
};

type Raw = { ts: number; ins: number | null; msid: string; artist: string; track: string; release: string | null;
  client: string | null; service: string | null; from: string | null; player: string | null; dur: number | null;
  spotify: string | null; artistMbids: string[] | null; recMbid: string | null; relMbid: string | null };

const spotifyUri = (s: string | null) => {
  if (!s) return null;
  const m = s.match(/track[/:]([A-Za-z0-9]{22})/);
  return m ? `spotify:track:${m[1]}` : null;
};

function clean(id: string) {
  const dir = join(RAW, id);
  const pages = readdirSync(dir).filter((f) => f.endsWith('.json.gz')).sort();
  const seen = new Set<string>();
  let raw: Raw[] = [];
  for (const p of pages) for (const l of JSON.parse(gunzipSync(readFileSync(join(dir, p))).toString()) as Raw[]) {
    const key = l.ts + '|' + l.msid; // pages overlap by one second on purpose
    if (!seen.has(key)) { seen.add(key); raw.push(l); }
  }
  const n0 = raw.length;
  const nowS = Date.now() / 1000;
  raw = raw.filter((l) => l.artist && l.track && l.ts >= EARLIEST && l.ts <= nowS + DAY);
  const badTs = n0 - raw.length - 0;
  raw.sort((a, b) => a.ts - b.ts);

  // duplicate scrobbles: the same track again within DUP_WINDOW seconds (two clients, retries)
  const plays: Raw[] = [];
  let dups = 0;
  const lastAt = new Map<string, number>();
  for (const l of raw) {
    const k = (l.artist + '\u0000' + l.track).toLowerCase();
    const prev = lastAt.get(k);
    if (prev !== undefined && l.ts - prev < DUP_WINDOW) { dups++; continue; }
    lastAt.set(k, l.ts);
    plays.push(l);
  }

  // signals
  const tsCount = new Map<number, number>();
  for (const l of plays) tsCount.set(l.ts, (tsCount.get(l.ts) ?? 0) + 1);
  const sharedTs = plays.filter((l) => tsCount.get(l.ts)! >= 4).length / Math.max(1, plays.length);
  let fast = 0, gaps = 0;
  const gapHist = new Map<number, number>();
  for (let i = 1; i < plays.length; i++) {
    const g = plays[i].ts - plays[i - 1].ts;
    if (g <= 0) continue;
    gaps++;
    if (g < 30 && plays[i].track !== plays[i - 1].track) fast++;
    if (g < 3600) gapHist.set(g, (gapHist.get(g) ?? 0) + 1);
  }
  const shortGaps = [...gapHist.values()].reduce((a, b) => a + b, 0);
  const regularGap = shortGaps ? Math.max(...gapHist.values()) / shortGaps : 0;
  const perDay = new Map<number, number>();
  const weeks = new Set<number>();
  for (const l of plays) {
    const d = Math.floor(l.ts / DAY);
    perDay.set(d, (perDay.get(d) ?? 0) + 1);
    weeks.add(Math.floor(l.ts / (7 * DAY)));
  }
  const heavyDays = [...perDay.values()].filter((c) => c >= 500).length / Math.max(1, perDay.size);
  const trackCount = new Map<string, number>();
  for (const l of plays) { const k = (l.artist + '\u0000' + l.track).toLowerCase(); trackCount.set(k, (trackCount.get(k) ?? 0) + 1); }
  const topTrack = Math.max(0, ...trackCount.values()) / Math.max(1, plays.length);
  const artistCount = new Map<string, number>(); // keyed as the statements engine keys artists
  for (const l of plays) { const k = l.artistMbids?.length === 1 ? l.artistMbids[0] : 'n:' + l.artist.toLowerCase(); artistCount.set(k, (artistCount.get(k) ?? 0) + 1); }
  const topArtist = Math.max(0, ...artistCount.values()) / Math.max(1, plays.length);
  const span = plays.length ? (plays[plays.length - 1].ts - plays[0].ts) / DAY : 0;
  const imported = plays.filter((l) => l.ins !== null && l.ins - l.ts > 7 * DAY).length / Math.max(1, plays.length);
  const withMbid = plays.filter((l) => l.artistMbids?.length).length / Math.max(1, plays.length);
  const withDur = plays.filter((l) => l.dur).length / Math.max(1, plays.length);
  const clients = new Map<string, number>();
  for (const l of plays) { const c = (l.client ?? 'unknown').toLowerCase(); clients.set(c, (clients.get(c) ?? 0) + 1); }

  const why: string[] = [];
  if (sharedTs > RULES.maxSharedTs) why.push('bulk-import timestamps');
  if (gaps && fast / gaps > RULES.maxFastGaps) why.push('listens faster than listening');
  if (heavyDays > RULES.maxHeavyDays) why.push('500+ listens a day');
  if (topTrack > RULES.maxTopTrack) why.push('one track dominates');
  if (regularGap > RULES.maxRegularGap) why.push('scripted pacing');
  if (topArtist > RULES.maxTopArtist || artistCount.size < RULES.minArtists) why.push('one artist only');
  if (plays.length < RULES.minListens) why.push('too few listens after cleaning');
  if (span < RULES.minSpanDays) why.push('under 12 months kept');
  if (weeks.size < RULES.minActiveWeeks) why.push('too few active weeks');

  const out: LbPlay[] = plays.map((l) => ({
    t: l.ts * 1000, durMs: l.dur ?? null, track: l.track, artist: l.artist, album: l.release,
    uri: spotifyUri(l.spotify), artistMbids: l.artistMbids?.length ? l.artistMbids : null, recMbid: l.recMbid,
    client: l.client, imported: l.ins !== null && l.ins - l.ts > 7 * DAY,
  }));
  const stats = {
    id, fetched: n0, kept: plays.length, badTs, dups, spanDays: Math.round(span), activeWeeks: weeks.size,
    first: plays[0]?.ts ?? null, last: plays[plays.length - 1]?.ts ?? null,
    sharedTs: +sharedTs.toFixed(4), fastGaps: gaps ? +(fast / gaps).toFixed(4) : 0, heavyDays: +heavyDays.toFixed(4),
    topTrack: +topTrack.toFixed(4), topArtist: +topArtist.toFixed(4), artists: artistCount.size, regularGap: +regularGap.toFixed(4), imported: +imported.toFixed(3),
    withMbid: +withMbid.toFixed(3), withDur: +withDur.toFixed(3),
    topClients: [...clients].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([c, n]) => [c, +(n / plays.length).toFixed(3)]),
    excluded: why.length ? why : null,
  };
  return { out, stats };
}

const ids = readdirSync(RAW).filter((d) => existsSync(join(RAW, d, 'state.json')) && readJson<any>(join(RAW, d, 'state.json'), {}).done);
const summary: any[] = [];
for (const id of ids) {
  const { out, stats } = clean(id);
  summary.push(stats);
  const f = join(PLAYS, id + '.json.gz');
  if (!stats.excluded) writeFileSync(f, gzipSync(JSON.stringify(out)));
  else if (existsSync(f)) writeFileSync(f, ''); // keep the bench consistent with the rules
}
writeJson(SUMMARY, summary);
const kept = summary.filter((s) => !s.excluded);
const reasons = new Map<string, number>();
for (const s of summary) for (const w of s.excluded ?? []) reasons.set(w, (reasons.get(w) ?? 0) + 1);
const med = (xs: number[]) => { const a = [...xs].sort((x, y) => x - y); return a.length ? a[Math.floor(a.length / 2)] : 0; };
console.log(JSON.stringify({
  users: summary.length, kept: kept.length, listens: kept.reduce((a, s) => a + s.kept, 0),
  medianListens: med(kept.map((s) => s.kept)), medianSpanDays: med(kept.map((s) => s.spanDays)),
  earliest: new Date(Math.min(...kept.map((s) => s.first)) * 1000).toISOString().slice(0, 10),
  latest: new Date(Math.max(...kept.map((s) => s.last)) * 1000).toISOString().slice(0, 10),
  dupsRemoved: summary.reduce((a, s) => a + s.dups, 0), badTsRemoved: summary.reduce((a, s) => a + s.badTs, 0),
  medianImported: med(kept.map((s) => s.imported)), medianWithMbid: med(kept.map((s) => s.withMbid)),
  medianWithDur: med(kept.map((s) => s.withDur)),
  exclusions: Object.fromEntries(reasons),
}, null, 1));

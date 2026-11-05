// Shared pieces for round 3: the ListenBrainz-user bench (bench/lbusers, 343 people). Each person
// is split by time like the Spotify bench: training is everything before their last 8 weeks, the
// held-out weeks are scored. People are placed in 5 folds by a hash of their pseudonym; every
// learned thing that scores a person (taste map, stick model, ranker) is fitted on the other four
// folds only, so a person never shapes what scores them. This is leave-one-person-out in spirit
// with 5 fits instead of 343.
//
// Artists are keyed by MusicBrainz id where the listen carries exactly one, else by "n:" plus the
// lower-cased credit, so the ListenBrainz similar-artist lists (keyed by id) can be read directly.
// Every step caches its output under bench/out/lb3/ (gitignored) and skips work already cached,
// so a crash or a network drop costs only the person in progress.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync, gzipSync } from 'node:zlib';
import { join } from 'node:path';
import { lbUserIds, loadLbUser, LbPlay } from '../lbusers/load.js';
import type { Play } from '../../engine/src/parse/spotify.js';
import { TasteMap } from '../../engine/src/model/map.js';
import { LbTable } from '../../engine/src/model/lbmap.js';

export const DAY = 86_400_000, WEEK = 7 * DAY, HOLDOUT_WEEKS = 8, FOLDS = 5;
export const CACHE = 'bench/out/lb3';
export const log = (s: string) => process.stderr.write(`${new Date().toISOString().slice(11, 19)} ${s}\n`);
export const hash = (s: string) => [...s].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 7);
export const foldOf = (id: string) => hash(id + ':fold') % FOLDS;
export const ids = () => lbUserIds();

export const artistKey = (p: LbPlay) => (p.artistMbids?.length === 1 ? p.artistMbids[0] : `n:${p.artist.toLowerCase()}`);

export interface LbPerson { id: string; plays: Play[]; n: number; cut: number; names: Map<string, string> }

const pool = new Map<string, string>();
const intern = (s: string) => pool.get(s) ?? (pool.set(s, s), s);

/** One person as engine plays (artist = key), split at their last 8 weeks. Null when too short. */
export function loadPerson(id: string): LbPerson | null {
  const raw = loadLbUser(id), names = new Map<string, string>();
  const plays: Play[] = raw.map((p) => {
    const k = intern(artistKey(p));
    if (!names.has(k)) names.set(k, p.artist);
    const ms = p.durMs && p.durMs > 30_000 ? p.durMs : 180_000;
    return { t: p.t, ms, track: p.track, artist: k, album: null, uri: p.recMbid ?? null, reasonStart: null, reasonEnd: null, shuffle: null, skipped: null, platform: null, country: null };
  });
  if (plays.length < 2000) return null;
  const cut = plays[plays.length - 1].t - HOLDOUT_WEEKS * WEEK;
  const n = plays.findIndex((p) => p.t >= cut);
  if (n < 1000 || plays.length - n < 200) return null;
  return { id, plays, n, cut, names };
}

export const utcHour = (p: Pick<Play, 't'>) => new Date(p.t).getUTCHours();

// ---- cache helpers
export function cachePath(...parts: string[]) { const p = join(CACHE, ...parts); mkdirSync(join(p, '..'), { recursive: true }); return p; }
export const has = (...parts: string[]) => existsSync(join(CACHE, ...parts));
export function writeJsonGz(path: string, v: unknown) { writeFileSync(path + '.tmp', gzipSync(JSON.stringify(v))); writeFileSync(path, readFileSync(path + '.tmp')); }
export const readJsonGz = <T>(path: string): T => JSON.parse(gunzipSync(readFileSync(path)).toString());

export function saveMap(path: string, m: TasteMap) {
  writeFileSync(path + '.bin', Buffer.from(new Float32Array(m.vec).buffer));
  writeFileSync(path + '.json', JSON.stringify({ artists: m.artists, dim: m.dim }));
}
export function loadMap(path: string): TasteMap {
  const { artists, dim } = JSON.parse(readFileSync(path + '.json', 'utf8')) as { artists: string[]; dim: number };
  const b = readFileSync(path + '.bin');
  const vec = Float64Array.from(new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4));
  return { artists, index: new Map(artists.map((a, i) => [a, i])), dim, vec };
}

// ---- ListenBrainz lists keyed by MusicBrainz id (the Spotify bench's cache plus this bench's extra fetch)
let table: LbTable | null = null;
export function lbTable(): LbTable {
  if (table) return table;
  const nodes: string[] = [], ix = new Map<string, number>(), lists: Record<string, [number, number][]> = {};
  const node = (m: string) => { let i = ix.get(m); if (i === undefined) { i = nodes.length; ix.set(m, i); nodes.push(m); } return i; };
  const add = (text: string) => {
    for (const l of text.split('\n')) {
      if (!l) continue;
      let o: { mbid: string; list: [string, string, number][] };
      try { o = JSON.parse(l); } catch { continue; }
      if (!o.list.length) continue;
      lists[node(o.mbid)] = o.list.map(([m, , s]) => [node(m), s] as [number, number]);
    }
  };
  add(gunzipSync(readFileSync('bench/data/lb/similar.jsonl.gz')).toString());
  if (existsSync('bench/data/lbusers/similar_extra.jsonl')) add(readFileSync('bench/data/lbusers/similar_extra.jsonl', 'utf8'));
  return (table = { nodes, lists });
}

/** Popularity and base rates over everyone's training weeks, written by prep.ts. */
export interface Pop { people: Record<string, number>; plays: Record<string, number>; rows: Record<string, [number, number]>; loves: Record<string, [number, number]> }

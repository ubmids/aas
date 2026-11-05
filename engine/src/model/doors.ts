// Where your new artists came from: a look back, not a forecast. For every artist a person fell
// for (first heard, then 10+ proper plays on 3+ days within 8 weeks) we read the door it came in
// through, with receipts: how its first play started (Spotify reason_start), the artist just before
// it in the same session, whether that artist or one of the person's recent favourites lists it on
// ListenBrainz (and at what rank), and how fast it grew.
//
// The same facts are read for controls, so a door only counts as an explanation when it is clearly
// more common for falls than for artists the person tried and dropped (same person, nearby dates).
// Facts are always shown as receipts; "came through" claims are only made for doors that pass.
import { Play, isSkip } from '../parse/spotify.js';
import { SESSION_GAP } from './session.js';
import { LbTable } from './lbmap.js';

const DAY = 86_400_000, WEEK = 7 * DAY;
export const FALL_PLAYS = 10, FALL_DAYS = 3, FALL_WINDOW = 8 * WEEK;
/** A first play needs this much history before it, so "new" and "recent favourites" mean something. */
export const MIN_HISTORY = 12 * WEEK;
/** Tried and dropped: at most this many proper plays ever. */
export const DROPPED_MAX = 2;
const FAVS = 30;

const PICKED = new Set(['clickrow', 'playbtn']);
const PASSIVE = new Set(['trackdone', 'fwdbtn']);
export type Mode = 'picked' | 'passive' | 'other' | 'unknown';
export const modeOf = (r: string | null): Mode => (!r ? 'unknown' : PICKED.has(r) ? 'picked' : PASSIVE.has(r) ? 'passive' : 'other');

export interface Door {
  artist: string;
  kind: 'fall' | 'dropped';
  first: number; // time of the first play
  mode: Mode;
  reason: string | null;
  shuffle: boolean | null;
  firstWasSkip: boolean;
  prev: string | null; // artist just before, same session, if different
  prevIsFav: boolean;
  prevRank: number | null; // rank of this artist on prev's ListenBrainz list (1 = top)
  fav: string | null; // the recent favourite that lists it best
  favRank: number | null;
  favShare: number; // share of the first session's proper plays that were recent favourites
  resolved: boolean; // the artist is a ListenBrainz node, so list facts can be read at all
  daysTo10: number | null; // falls: days from first play to the 10th proper play
}

/** One ListenBrainz lookup shared by every person: node index, and per list a rank table. */
export interface Lists { ix: Map<string, number>; rank: (lister: string, artist: string) => number | null; listedBy: Int32Array; nodes: string[] }
export function listsOf(t: LbTable): Lists {
  const ix = new Map(t.nodes.map((n, i) => [n, i]));
  const ranks = new Map<number, Map<number, number>>(), listedBy = new Int32Array(t.nodes.length);
  for (const [k, list] of Object.entries(t.lists)) {
    const m = new Map<number, number>();
    list.forEach(([n], r) => { if (!m.has(n)) m.set(n, r + 1); listedBy[n]++; });
    ranks.set(Number(k), m);
  }
  const rank = (lister: string, artist: string) => {
    const a = ix.get(lister), b = ix.get(artist);
    return a === undefined || b === undefined ? null : ranks.get(a)?.get(b) ?? null;
  };
  return { ix, rank, listedBy, nodes: t.nodes };
}

/** Top recent artists by proper plays in the 8 weeks before plays[i]. */
export function recentFavs(plays: Play[], i: number): Map<string, number> {
  const t = plays[i].t, c = new Map<string, number>();
  for (let j = i - 1; j >= 0 && t - plays[j].t < FALL_WINDOW; j--) if (!isSkip(plays[j])) c.set(plays[j].artist, (c.get(plays[j].artist) ?? 0) + 1);
  return new Map([...c].sort((a, b) => b[1] - a[1]).slice(0, FAVS));
}

/** Reads the door facts for the artist first played at plays[i]. */
export function readDoor(plays: Play[], i: number, kind: Door['kind'], L: Lists | null, daysTo10: number | null): Door {
  const p = plays[i], favs = recentFavs(plays, i);
  const q = i > 0 ? plays[i - 1] : null;
  const prev = q && p.t - (q.t + q.ms) <= SESSION_GAP && q.artist !== p.artist ? q.artist : null;
  let fav: string | null = null, favRank: number | null = null;
  if (L) for (const f of favs.keys()) { const r = L.rank(f, p.artist); if (r !== null && (favRank === null || r < favRank)) { favRank = r; fav = f; } }
  // the first session around the first play
  let a = i, b = i;
  while (a > 0 && plays[a].t - (plays[a - 1].t + plays[a - 1].ms) <= SESSION_GAP) a--;
  while (b + 1 < plays.length && plays[b + 1].t - (plays[b].t + plays[b].ms) <= SESSION_GAP) b++;
  let proper = 0, fromFav = 0;
  for (let j = a; j <= b; j++) if (j !== i && !isSkip(plays[j]) && plays[j].artist !== p.artist) { proper++; if (favs.has(plays[j].artist)) fromFav++; }
  return {
    artist: p.artist, kind, first: p.t, mode: modeOf(p.reasonStart), reason: p.reasonStart, shuffle: p.shuffle,
    firstWasSkip: isSkip(p), prev, prevIsFav: !!prev && favs.has(prev), prevRank: prev && L ? L.rank(prev, p.artist) : null,
    fav, favRank, favShare: proper ? fromFav / proper : 0, resolved: !!L && L.ix.has(p.artist), daysTo10,
  };
}

/** Every fall and every tried-and-dropped artist in a person's history (first play after MIN_HISTORY). */
export function newArtists(plays: Play[]): { i: number; kind: Door['kind']; daysTo10: number | null }[] {
  if (!plays.length) return [];
  const firstIx = new Map<string, number>(), proper = new Map<string, number[]>();
  plays.forEach((p, i) => {
    if (!firstIx.has(p.artist)) firstIx.set(p.artist, i);
    if (!isSkip(p)) (proper.get(p.artist) ?? proper.set(p.artist, []).get(p.artist)!).push(p.t);
  });
  const start = plays[0].t + MIN_HISTORY, end = plays[plays.length - 1].t;
  const out: { i: number; kind: Door['kind']; daysTo10: number | null }[] = [];
  for (const [artist, i] of firstIx) {
    const t0 = plays[i].t;
    if (t0 < start || t0 > end - FALL_WINDOW) continue;
    const ts = proper.get(artist) ?? [];
    const inWin = ts.filter((t) => t < t0 + FALL_WINDOW);
    const days = new Set(inWin.map((t) => Math.floor(t / DAY))).size;
    if (inWin.length >= FALL_PLAYS && days >= FALL_DAYS) out.push({ i, kind: 'fall', daysTo10: (inWin[FALL_PLAYS - 1] - t0) / DAY });
    else if (ts.length <= DROPPED_MAX) out.push({ i, kind: 'dropped', daysTo10: null });
  }
  return out.sort((x, y) => x.i - y.i);
}

/** For each fall, up to k dropped artists whose first play was nearest in time (same person). */
export function matchDropped(doors: Door[], k = 5): Map<Door, Door[]> {
  const dropped = doors.filter((d) => d.kind === 'dropped').sort((a, b) => a.first - b.first);
  const out = new Map<Door, Door[]>();
  for (const f of doors.filter((d) => d.kind === 'fall')) {
    // binary search then widen both ways
    let lo = 0, hi = dropped.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (dropped[m].first < f.first) lo = m + 1; else hi = m; }
    const pick: Door[] = [];
    let l = lo - 1, r = lo;
    while (pick.length < k && (l >= 0 || r < dropped.length)) {
      const dl = l >= 0 ? f.first - dropped[l].first : Infinity, dr = r < dropped.length ? dropped[r].first - f.first : Infinity;
      if (dl <= dr) pick.push(dropped[l--]); else pick.push(dropped[r++]);
    }
    if (pick.length && Math.abs(pick[pick.length - 1].first - f.first) <= 60 * DAY) out.set(f, pick);
  }
  return out;
}

/** The facets compared between falls and controls. Each returns true, false, or null (can't tell). */
export const FACETS: Record<string, (d: Door) => boolean | null> = {
  picked: (d) => (d.mode === 'unknown' ? null : d.mode === 'picked'),
  cameOn: (d) => (d.mode === 'unknown' ? null : d.mode === 'passive'),
  prevLists: (d) => (d.resolved && d.prev ? d.prevRank !== null : null),
  cameOnAfterSimilar: (d) => (d.mode === 'unknown' || !d.resolved ? null : d.mode === 'passive' && d.prevRank !== null),
  pickedAfterFav: (d) => (d.mode === 'unknown' ? null : d.mode === 'picked' && d.prevIsFav),
  favLists: (d) => (d.resolved ? d.favRank !== null : null),
  favListsTop25: (d) => (d.resolved ? d.favRank !== null && d.favRank <= 25 : null),
  prevIsFav: (d) => (d.prev ? d.prevIsFav : null),
  shuffle: (d) => d.shuffle,
  firstWasSkip: (d) => d.firstWasSkip,
};

/** The door one fall is filed under, for the summary. */
export type DoorKind = 'picked' | 'came on after a similar artist' | 'came on by itself' | 'next to a favourite' | 'out of nowhere';
export function doorKind(d: Door): DoorKind {
  if (d.mode === 'picked') return 'picked';
  if (d.mode === 'passive' && d.prevRank !== null) return 'came on after a similar artist';
  if (d.favRank !== null) return 'next to a favourite';
  if (d.mode === 'passive') return 'came on by itself';
  return 'out of nowhere';
}

const fmtDate = (t: number) => new Date(t).toISOString().slice(0, 10);
const days = (x: number) => (x < 1.5 ? 'within a day' : `${Math.round(x)} days later`);

/** A card-ready story for one fall: facts only, each with its receipt. */
export function story(d: Door): string {
  const how = d.mode === 'picked' ? (d.prev ? `you picked it yourself, straight after ${d.prev}` : 'you picked it yourself')
    : d.mode === 'passive' ? (d.prev ? `it came on after ${d.prev}${d.reason === 'fwdbtn' ? ' when you skipped ahead' : ''}` : 'it came on by itself')
      : 'we can\'t tell how it started';
  const parts = [`You first heard ${d.artist} on ${fmtDate(d.first)}: ${how}${d.firstWasSkip ? ', and you skipped it that first time' : ''}.`];
  if (d.prevRank !== null) parts.push(`${d.prev} lists ${d.artist} at #${d.prevRank} of its similar artists on ListenBrainz.`);
  else if (d.fav && d.favRank !== null) parts.push(`${d.fav}, one of your most played that month, lists ${d.artist} at #${d.favRank} on ListenBrainz.`);
  if (d.daysTo10 !== null) parts.push(`You'd played it properly 10 times ${days(d.daysTo10)}.`);
  return parts.join(' ');
}

export interface DoorSummary { n: number; kinds: [DoorKind, number][]; line: string; fastest: Door | null; own: number | null }

const TAIL: Record<DoorKind, string> = {
  picked: 'you went and picked yourself',
  'came on after a similar artist': 'came on after an artist ListenBrainz calls similar',
  'next to a favourite': 'sit right next to artists you were already playing a lot',
  'came on by itself': 'came on by themselves, after something else',
  'out of nowhere': 'came out of nowhere',
};

const WAY: Record<DoorKind, string> = {
  picked: 'you picked them yourself',
  'came on after a similar artist': 'they came on after a similar artist',
  'next to a favourite': 'they sit next to an artist you were playing a lot',
  'came on by itself': 'they came on by themselves after something else',
  'out of nowhere': 'no trail at all',
};

/**
 * Per-person summary. The headline door is claimed as "how new music reaches you" only when it is
 * clearly more common for this person's falls than for the artists they tried and dropped (1.5x,
 * with at least 10 falls and 20 dropped), or, failing their own evidence, when it beat controls
 * across the bench (`evidence`). Otherwise the line just counts, and says how it compares.
 */
export function summarise(falls: Door[], evidence: Set<DoorKind>, ownDropped: Door[] = []): DoorSummary {
  const c = new Map<DoorKind, number>();
  for (const d of falls) c.set(doorKind(d), (c.get(doorKind(d)) ?? 0) + 1);
  const kinds = [...c].sort((a, b) => b[1] - a[1]);
  const n = falls.length, fastest = [...falls].filter((d) => d.daysTo10 !== null).sort((a, b) => a.daysTo10! - b.daysTo10!)[0] ?? null;
  if (!n) return { n, kinds, line: 'No new artists stuck long enough to count yet.', fastest, own: null };
  const [top, k] = kinds[0], share = k / n;
  const own = ownDropped.length ? ownDropped.filter((d) => doorKind(d) === top).length / ownDropped.length : null;
  const pct = (x: number) => `${Math.round(100 * x)}%`;
  const words = share > 0.5 ? 'Most' : share > 0.3 ? 'A lot' : 'Some';
  const head = `${words} of your ${n} new loves (${k}) ${top === 'picked' ? 'were artists ' : ''}${TAIL[top]}.`;
  let line: string;
  if (own !== null && n >= 10 && ownDropped.length >= 20 && share >= 1.5 * own)
    line = `${head} Only ${pct(own)} of the artists you tried and dropped came in that way, so this is how new music sticks for you.`;
  else if (evidence.has(top)) line = own === null ? head : `${head} (${pct(own)} of the artists you tried and dropped came in that way.)`;
  else {
    const cmp = own === null ? '' : share > own * 1.2 ? `, a bit more often than for artists you tried and dropped (${pct(own)})`
      : share < own / 1.2 ? `, less often than for artists you tried and dropped (${pct(own)})` : `, about as often as for artists you tried and dropped (${pct(own)})`;
    line = `Your ${n} new loves came in many ways. The most common way in (${k} of them): ${WAY[top]}${cmp}.`;
  }
  return { n, kinds, line, fastest, own };
}

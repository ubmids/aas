// Will this new artist stick? For every artist a person hears for the first time, predict from what
// happened by the end of the first session (horizon 'session') or by the end of the first 7 days
// (horizon 'week') whether it becomes a new love: 10+ proper plays on 3+ days within 8 weeks of the
// first play (the same definition as falls and doors).
//
// Every feature reads only plays that started before the horizon ends, so the score could be shown
// to the person at that moment. `horizonEnd` gives that index; the bench checks that features read
// from the full history equal features read from the history cut at the horizon.
import { Play, isSkip, trackKey } from '../parse/spotify.js';
import { SESSION_GAP } from './session.js';
import { Lists, recentFavs, FALL_PLAYS, FALL_DAYS, FALL_WINDOW, MIN_HISTORY, modeOf } from './doors.js';
import { TasteMap, vecOf } from './map.js';
import { dot } from './linalg.js';

const DAY = 86_400_000;
export const WEEK_HORIZON = 7 * DAY;
export type Horizon = 'session' | 'week';

export const STICK_FEATURES_SESSION = [
  'firstSkip', 'logFirstSec', 'picked', 'cameOn', 'shuffle',
  'prevLists', 'prevRankScore', 'favLists', 'favRankScore', 'onLb', 'favShare',
  'mapNear', 'onMap', 'logPeople', 'logListed',
  'sessPlays', 'sessProper', 'sessTracks', 'sessRepeat', 'sessReturn',
  'hourSin', 'hourCos', 'sessLen', 'sessSkipRate', 'baseLogit',
] as const;
export const STICK_FEATURES_WEEK = [...STICK_FEATURES_SESSION, 'wkProper', 'wkDays', 'wkTracks', 'wkReturned', 'wkSkipRate', 'wkPicked'] as const;
export const featuresOf = (h: Horizon): readonly string[] => (h === 'session' ? STICK_FEATURES_SESSION : STICK_FEATURES_WEEK);

/**
 * The columns a history without skips, play lengths, start reasons or time zones can fill (the
 * ListenBrainz-user bench): everything except the first play's skip and seconds, how it started,
 * shuffle, the session's skip rate, the week's skip rate and picked share, and the local hour.
 * The same columns are fitted on the Spotify bench to show what the dropped ones add.
 */
export const STICK_DROPPED_NO_SKIP = ['firstSkip', 'logFirstSec', 'picked', 'cameOn', 'shuffle', 'sessSkipRate', 'hourSin', 'hourCos', 'wkSkipRate', 'wkPicked'];
export const noSkipFeaturesOf = (h: Horizon): string[] => featuresOf(h).filter((f) => !STICK_DROPPED_NO_SKIP.includes(f));

/** Feature groups for ablations (names, not indices, so both horizons share them). */
export const STICK_GROUPS: Record<string, string[]> = {
  'first play (skip, seconds)': ['firstSkip', 'logFirstSec'],
  'how it started': ['picked', 'cameOn', 'shuffle'],
  'ListenBrainz lists': ['prevLists', 'prevRankScore', 'favLists', 'favRankScore', 'onLb'],
  'favourites in the session': ['favShare'],
  'taste map': ['mapNear', 'onMap'],
  popularity: ['logPeople', 'logListed'],
  'the first session': ['sessPlays', 'sessProper', 'sessTracks', 'sessRepeat', 'sessReturn', 'sessLen', 'sessSkipRate'],
  'time of day': ['hourSin', 'hourCos'],
  'own base rate': ['baseLogit'],
  'the first week': ['wkProper', 'wkDays', 'wkTracks', 'wkReturned', 'wkSkipRate', 'wkPicked'],
};

export interface NewArtist {
  artist: string;
  i: number; // index of the first play
  t0: number;
  label: 0 | 1; // became a new love
  fallenByWeek: boolean; // already met the definition within the first 7 days
  daysTo10: number | null;
}

/** Every artist first played after MIN_HISTORY whose 8-week outcome window ends by `until`. */
export function newArtistRows(plays: Play[], until = Infinity): NewArtist[] {
  if (!plays.length) return [];
  const first = new Map<string, number>(), proper = new Map<string, number[]>();
  plays.forEach((p, i) => {
    if (!first.has(p.artist)) first.set(p.artist, i);
    if (!isSkip(p)) (proper.get(p.artist) ?? proper.set(p.artist, []).get(p.artist)!).push(p.t);
  });
  const start = plays[0].t + MIN_HISTORY, end = Math.min(until, plays[plays.length - 1].t);
  const out: NewArtist[] = [];
  for (const [artist, i] of first) {
    const t0 = plays[i].t;
    if (t0 < start || t0 + FALL_WINDOW > end) continue;
    const ts = (proper.get(artist) ?? []).filter((t) => t < t0 + FALL_WINDOW);
    const met = (lim: number) => { const w = ts.filter((t) => t < lim); return w.length >= FALL_PLAYS && new Set(w.map((t) => Math.floor(t / DAY))).size >= FALL_DAYS; };
    const label = met(t0 + FALL_WINDOW);
    // the day the definition was first met: the first proper play at which both counts hold
    let daysTo10: number | null = null;
    if (label) for (let k = FALL_PLAYS - 1; k < ts.length; k++) if (new Set(ts.slice(0, k + 1).map((t) => Math.floor(t / DAY))).size >= FALL_DAYS) { daysTo10 = (ts[k] - t0) / DAY; break; }
    out.push({ artist, i, t0, label: label ? 1 : 0, fallenByWeek: label && met(t0 + WEEK_HORIZON), daysTo10 });
  }
  return out.sort((a, b) => a.i - b.i);
}

/** Index of the last play that starts before the horizon ends (the first session's last play, or the last play inside 7 days). */
export function horizonEnd(plays: Play[], i: number, h: Horizon): number {
  let b = i;
  if (h === 'session') while (b + 1 < plays.length && plays[b + 1].t - (plays[b].t + plays[b].ms) <= SESSION_GAP) b++;
  else while (b + 1 < plays.length && plays[b + 1].t < plays[i].t + WEEK_HORIZON) b++;
  return b;
}

export interface StickContext {
  lists: Lists | null;
  map: TasteMap | null; // a map the person never shaped
  people: Map<string, number>; // how many other people played each artist
  hour: (p: Play) => number; // local hour
}

/** Recent favourites at the start of the session holding play i (shared by every new artist in that session). */
export function favsAt(plays: Play[], i: number, cache: Map<number, Map<string, number>>) {
  let a = i;
  while (a > 0 && plays[a].t - (plays[a - 1].t + plays[a - 1].ms) <= SESSION_GAP) a--;
  let f = cache.get(a);
  if (!f) cache.set(a, (f = recentFavs(plays, a)));
  return { a, favs: f };
}

const rankScore = (r: number | null) => (r === null ? 0 : 1 / Math.log2(1 + r));

/**
 * The feature row for the new artist first played at plays[i], read up to the horizon. `baseLogit`
 * is the person's own earlier rate of new artists sticking (outcomes known before plays[i]), as a
 * logit, computed by the caller.
 */
export function stickFeatures(plays: Play[], i: number, h: Horizon, ctx: StickContext, baseLogit: number, favCache = new Map<number, Map<string, number>>()): number[] {
  const p = plays[i], artist = p.artist;
  const { a, favs } = favsAt(plays, i, favCache);
  const s1 = horizonEnd(plays, i, 'session');
  const b = h === 'session' ? s1 : horizonEnd(plays, i, 'week');
  const q = i > a ? plays[i - 1] : null;
  const prev = q && q.artist !== artist ? q.artist : null;
  const L = ctx.lists;
  const prevRank = prev && L ? L.rank(prev, artist) : null;
  let favRank: number | null = null;
  if (L) for (const f of favs.keys()) { const r = L.rank(f, artist); if (r !== null && (favRank === null || r < favRank)) favRank = r; }
  const ix = L?.ix.get(artist);
  const mode = modeOf(p.reasonStart);

  // the first session (from its start to its end, which is never after the horizon)
  let sessN = 0, sessSkips = 0, own = 0, ownProper = 0, others = 0, fromFav = 0, rep = false, ret = false, left = false;
  const tracks = new Set<string>(), seen = new Set<string>();
  for (let j = a; j <= s1; j++) {
    const x = plays[j];
    sessN++; if (isSkip(x)) sessSkips++;
    if (x.artist === artist) {
      if (j >= i) { own++; if (!isSkip(x)) ownProper++; tracks.add(trackKey(x)); if (seen.has(trackKey(x))) rep = true; seen.add(trackKey(x)); if (left) ret = true; }
    } else {
      if (j > i && own) left = true;
      if (!isSkip(x)) { others++; if (favs.has(x.artist)) fromFav++; }
    }
  }

  // nearness on the map: best cosine to the top 10 favourites
  let mapNear = 0, onMap = 0;
  const v = ctx.map ? vecOf(ctx.map, artist) : null;
  if (v && ctx.map) {
    onMap = 1;
    let k = 0;
    for (const f of favs.keys()) { if (k++ >= 10) break; const u = vecOf(ctx.map, f); if (u) mapNear = Math.max(mapNear, dot(u, v)); }
  }
  const hr = ctx.hour(p), ang = (2 * Math.PI * hr) / 24;
  const row = [
    isSkip(p) ? 1 : 0, Math.log1p(Math.min(p.ms, 600_000) / 1000), mode === 'picked' ? 1 : 0, mode === 'passive' ? 1 : 0, p.shuffle ? 1 : 0,
    prevRank !== null ? 1 : 0, rankScore(prevRank), favRank !== null ? 1 : 0, rankScore(favRank), ix !== undefined ? 1 : 0, others ? fromFav / others : 0,
    mapNear, onMap, Math.log1p(ctx.people.get(artist) ?? 0), Math.log1p(ix !== undefined ? L!.listedBy[ix] : 0),
    Math.log1p(own), Math.log1p(ownProper), Math.log1p(tracks.size), rep ? 1 : 0, ret ? 1 : 0,
    Math.sin(ang), Math.cos(ang), Math.log1p(sessN), sessN ? sessSkips / sessN : 0, baseLogit,
  ];
  if (h === 'session') return row;

  // the first 7 days
  let wk = 0, wkProper = 0, wkSkip = 0, wkPicked = 0, wkKnown = 0, returned = 0;
  const days = new Set<number>(), wkTracks = new Set<string>();
  for (let j = i; j <= b; j++) {
    const x = plays[j];
    if (x.artist !== artist) continue;
    wk++; if (isSkip(x)) wkSkip++; else { wkProper++; days.add(Math.floor(x.t / DAY)); }
    wkTracks.add(trackKey(x));
    const m = modeOf(x.reasonStart);
    if (m !== 'unknown') { wkKnown++; if (m === 'picked') wkPicked++; }
    if (j > s1) returned = 1;
  }
  return [...row, Math.log1p(wkProper), days.size, Math.log1p(wkTracks.size), returned, wk ? wkSkip / wk : 0, wkKnown ? wkPicked / wkKnown : 0];
}

/** The person's own rate of new artists sticking among those whose outcome was known by time t, shrunk to `prior` with weight k, as a logit. */
export function baseRateLogit(rows: NewArtist[], t: number, prior: number, k = 20): number {
  let n = 0, y = 0;
  for (const r of rows) if (r.t0 + FALL_WINDOW <= t) { n++; y += r.label; }
  const p = (y + k * prior) / (n + k);
  return Math.log(p / (1 - p));
}

/** Average precision: the mean of precision at each positive, scores ranked high to low (ties broken pessimistically). */
export function averagePrecision(scores: number[], labels: number[]): number {
  const o = scores.map((s, i) => [s, labels[i]] as const).sort((x, y) => y[0] - x[0] || x[1] - y[1]);
  let tp = 0, sum = 0;
  o.forEach(([, y], k) => { if (y) { tp++; sum += tp / (k + 1); } });
  return tp ? sum / tp : NaN;
}

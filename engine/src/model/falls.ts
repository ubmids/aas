// Who you'll fall for next. A "fall" is an artist never played before a cutoff that the person then
// plays properly (>= 30 s) at least 10 times on at least 3 different days within the next 8 weeks.
// We rank unheard map artists by where the person's taste sits now, where it is drifting, and how
// big their past jumps to new artists were (measured on their own earlier falls).
import { Play, isSkip } from '../parse/spotify.js';
import { TasteMap, centroid, simAll, vecOf } from './map.js';
import { dot } from './linalg.js';

const DAY = 86_400_000, WEEK = 7 * DAY;
export const FALL_WEEKS = 8, FALL_PLAYS = 10, FALL_DAYS = 3;

export function knownBefore(plays: Play[], t: number): Set<string> {
  const s = new Set<string>();
  for (const p of plays) { if (p.t >= t) break; s.add(p.artist); }
  return s;
}

/** Artists that become falls in [from, from + 8 weeks). */
export function findFalls(plays: Play[], from: number, known: Set<string>): string[] {
  const to = from + FALL_WEEKS * WEEK, n = new Map<string, number>(), days = new Map<string, Set<number>>();
  for (const p of plays) {
    if (p.t < from || isSkip(p) || known.has(p.artist)) continue;
    if (p.t >= to) break;
    n.set(p.artist, (n.get(p.artist) ?? 0) + 1);
    (days.get(p.artist) ?? days.set(p.artist, new Set()).get(p.artist)!).add(Math.floor(p.t / DAY));
  }
  return [...n.keys()].filter((a) => n.get(a)! >= FALL_PLAYS && days.get(a)!.size >= FALL_DAYS);
}

/** Where the person's taste sits at time t: recent, older, all-time centroids and recent favourites. */
export function tasteState(m: TasteMap, plays: Play[], t: number) {
  const recent = new Map<string, number>(), prev = new Map<string, number>(), all = new Map<string, number>();
  for (const p of plays) {
    if (p.t >= t) break;
    if (isSkip(p)) continue;
    const age = t - p.t;
    all.set(p.artist, (all.get(p.artist) ?? 0) + 1);
    if (age < 8 * WEEK) recent.set(p.artist, (recent.get(p.artist) ?? 0) + 1);
    else if (age < 26 * WEEK) prev.set(p.artist, (prev.get(p.artist) ?? 0) + 1);
  }
  const cAll = centroid(m, all), cRecent = centroid(m, recent) ?? cAll, cPrev = centroid(m, prev) ?? cRecent;
  let cDrift: Float64Array | null = null;
  if (cRecent && cPrev) {
    cDrift = new Float64Array(m.dim);
    for (let j = 0; j < m.dim; j++) cDrift[j] = 2 * cRecent[j] - cPrev[j];
    const nn = Math.sqrt(dot(cDrift, cDrift)) || 1;
    for (let j = 0; j < m.dim; j++) cDrift[j] /= nn;
  }
  const favs = [...(recent.size ? recent : all).entries()].sort((a, b) => b[1] - a[1]).slice(0, 30)
    .map(([a]) => vecOf(m, a)).filter((v): v is Float64Array => !!v);
  return { cAll, cRecent, cDrift, favs };
}

export type TasteState = ReturnType<typeof tasteState>;

/** Proper plays per artist in the 8 weeks before t. */
export function recentCounts(plays: Play[], t: number): Map<string, number> {
  const out = new Map<string, number>();
  for (const p of plays) { if (p.t >= t) break; if (!isSkip(p) && t - p.t < 8 * WEEK) out.set(p.artist, (out.get(p.artist) ?? 0) + 1); }
  return out;
}

/** A person's typical jump: mean and sd of similarity between their past falls and their recent taste. */
export interface Jump { mu: number; sd: number; n: number }
export function jumpFrom(sims: number[], prior: Jump, k = 5): Jump {
  const n = sims.length;
  const mean = n ? sims.reduce((s, v) => s + v, 0) / n : prior.mu;
  const v = n > 1 ? sims.reduce((s, x) => s + (x - mean) ** 2, 0) / (n - 1) : prior.sd ** 2;
  return { mu: (n * mean + k * prior.mu) / (n + k), sd: Math.sqrt((n * v + k * prior.sd ** 2) / (n + k)), n };
}

export const FALL_FEATURES = ['simRecent', 'simAllTime', 'simDrift', 'maxSimFav', 'jumpFit', 'logPeople', 'logPlays', 'logListed', 'listedByRecent'];

/** ListenBrainz maps carry how many seeds list each artist; v0 has no such prior (0). */
const listedOf = (m: TasteMap) => (m as TasteMap & { listed?: Float64Array }).listed;

/** One feature row per map artist (NaN-free; artists not on the map are never candidates). */
/** `near`: optional direct list evidence from the person's recent artists (ListenBrainz maps only). */
export function fallFeatures(m: TasteMap, st: TasteState, jump: Jump, pop: { people: Map<string, number>; plays: Map<string, number> }, near?: Map<string, number>) {
  const zero = new Float64Array(m.artists.length);
  const sR = st.cRecent ? simAll(m, st.cRecent) : zero, sA = st.cAll ? simAll(m, st.cAll) : zero, sD = st.cDrift ? simAll(m, st.cDrift) : zero;
  const maxFav = new Float64Array(m.artists.length).fill(st.favs.length ? -1 : 0);
  for (const f of st.favs) {
    const s = simAll(m, f);
    for (let i = 0; i < s.length; i++) if (s[i] > maxFav[i]) maxFav[i] = s[i];
  }
  const listed = listedOf(m);
  return (i: number) => {
    const a = m.artists[i];
    return [sR[i], sA[i], sD[i], maxFav[i], -(((sR[i] - jump.mu) / jump.sd) ** 2), Math.log1p(pop.people.get(a) ?? 0), Math.log1p(pop.plays.get(a) ?? 0), Math.log1p(listed?.[i] ?? 0), Math.log1p(10 * (near?.get(a) ?? 0))];
  };
}

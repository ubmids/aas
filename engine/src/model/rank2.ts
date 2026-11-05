// New loves, ranked in two stages. Stage one proposes candidates: every artist on a recent
// favourite's ListenBrainz similar-artist list, plus the artists nearest the favourites on the
// taste map. Stage two scores each candidate with a logistic ranker over list evidence (how many
// favourites list it, at what ranks, weighted by how much and how recently each favourite was
// played), popularity, map nearness and the person's own history of how new loves reached them.
// Artists already heard before the cutoff are never candidates.
import { Play, isSkip } from '../parse/spotify.js';
import { TasteMap, simAll } from './map.js';

const DAY = 86_400_000, WEEK = 7 * DAY;
/** Favourites: the top artists by proper plays in the 8 weeks before the cutoff. */
export const FAVS = 30;
/** Map neighbours added to the candidates (by nearness to the nearest favourite). */
export const MAP_CANDS = 300;
/** Half-life of a favourite's weight by the age of its last play. */
export const FAV_HALF_LIFE = 14 * DAY;

export const RANK2_FEATURES = ['logListers', 'rrSum', 'rrMax', 'logBestRank', 'scoreSum', 'logListed', 'logPeople', 'logPlays', 'maxSimFav', 'fromMapOnly', 'propList', 'propList_x_rrSum'] as const;

export interface Favourite { artist: string; weight: number }

/** Recent favourites with weights: play share in the 8 weeks, times a decay on the last play's age. */
export function favourites(plays: Play[], t: number, k = FAVS): Favourite[] {
  const n = new Map<string, number>(), last = new Map<string, number>();
  let tot = 0;
  for (const p of plays) {
    if (p.t >= t) break;
    if (isSkip(p) || t - p.t >= 8 * WEEK) continue;
    n.set(p.artist, (n.get(p.artist) ?? 0) + 1); last.set(p.artist, p.t); tot++;
  }
  return [...n].sort((a, b) => b[1] - a[1]).slice(0, k)
    .map(([artist, c]) => ({ artist, weight: (c / tot) * 0.5 ** ((t - last.get(artist)!) / FAV_HALF_LIFE) }));
}

/** ListenBrainz lists as ranks: lister name -> (artist name -> [rank (1 = top), score / top score]). */
export interface ListIndex { get(lister: string): Map<string, [number, number]> | undefined }

export interface Candidate { artist: string; x: number[] }

/**
 * Candidates and their features at cutoff t. `known`: artists heard before t. `pop`: popularity
 * among other people. `m`: a taste map the person never shaped (with `listed`, how many of the
 * map's lists name each artist). `propList`: the person's own share of earlier new loves that a
 * recent favourite listed, shrunk to everyone's.
 */
export function candidates(
  favs: Favourite[], known: Set<string>, lists: ListIndex,
  m: TasteMap & { listed?: Float64Array }, pop: { people: Map<string, number>; plays: Map<string, number> }, propList: number,
): Candidate[] {
  const ev = new Map<string, { listers: number; rr: number; rrMax: number; best: number; score: number }>();
  for (const f of favs) {
    const l = lists.get(f.artist);
    if (!l) continue;
    for (const [a, [rank, score]] of l) {
      if (known.has(a)) continue;
      const e = ev.get(a) ?? { listers: 0, rr: 0, rrMax: 0, best: Infinity, score: 0 };
      e.listers++; e.rr += f.weight / rank; e.rrMax = Math.max(e.rrMax, 1 / rank); e.best = Math.min(e.best, rank); e.score += f.weight * score;
      ev.set(a, e);
    }
  }
  // nearness to the nearest favourite on the map, for every map artist
  const maxFav = new Float64Array(m.artists.length).fill(-1);
  let anyFav = false;
  for (const f of favs) {
    const i = m.index.get(f.artist);
    if (i === undefined) continue;
    anyFav = true;
    const s = simAll(m, m.vec.subarray(i * m.dim, (i + 1) * m.dim));
    for (let j = 0; j < s.length; j++) if (s[j] > maxFav[j]) maxFav[j] = s[j];
  }
  const fromMap = new Set<string>();
  if (anyFav) {
    const order = Array.from(maxFav.keys()).filter((j) => !known.has(m.artists[j])).sort((a, b) => maxFav[b] - maxFav[a] || a - b);
    for (const j of order.slice(0, MAP_CANDS)) if (!ev.has(m.artists[j])) fromMap.add(m.artists[j]);
  }
  const out: Candidate[] = [];
  const row = (a: string, onlyMap: boolean): number[] => {
    const e = ev.get(a), j = m.index.get(a);
    const rr = e?.rr ?? 0;
    return [
      Math.log1p(e?.listers ?? 0), rr, e?.rrMax ?? 0, e ? Math.log(e.best) : Math.log(200), e?.score ?? 0,
      Math.log1p(j !== undefined ? m.listed?.[j] ?? 0 : 0), Math.log1p(pop.people.get(a) ?? 0), Math.log1p(pop.plays.get(a) ?? 0),
      j !== undefined && anyFav ? maxFav[j] : 0, onlyMap ? 1 : 0, propList, propList * rr,
    ];
  };
  for (const a of ev.keys()) out.push({ artist: a, x: row(a, false) });
  for (const a of fromMap) out.push({ artist: a, x: row(a, true) });
  return out;
}

/** Recall@k of a ranking against the window's new loves (all of them, candidates or not). */
export function recallAt(ranked: string[], loves: Set<string>, k: number) {
  if (!loves.size) return NaN;
  let h = 0;
  for (const a of ranked.slice(0, k)) if (loves.has(a)) h++;
  return h / loves.size;
}

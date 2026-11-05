// The prediction game (docs/GAME.md): pieces shared by bench/game.ts and its tests.
//   game 1: call 10 skips at the start of a block of 10 plays (features frozen at the block start)
//   game 2: of three new artists (one stuck, two did not), pick the one that sticks
//   game 3: name next week's top 5 artists
// Plus the per-person statistics: Wilson interval and Holm adjustment.
import { Play, isSkip, trackKey } from '../parse/spotify.js';

export const BLOCK = 10;
const SESSION_GAP = 30 * 60_000;
const STARTS = ['clickrow', 'trackdone', 'fwdbtn', 'backbtn', 'playbtn', 'appload', 'remote'];

export const FROZEN_FEATURES = [
  'trackRate', 'artistRate', 'logTrackN', 'logArtistN', 'daysTrack', 'daysArtist', 'recent',
  'prevSkip', 'sessRate', 'logSessPos', 'offset', 'sameArtistPrev', 'shuffle', 'hourSin', 'hourCos',
  ...STARTS.map((s) => `start_${s}`),
] as const;
/** Index of the artist's frozen skip rate (the most-skipped-artists baseline). */
export const ARTIST_RATE = 1;

export interface Block { start: number; len: number; session: number }

/** Sessions (30-minute gap after a play ends) cut into consecutive blocks of `size` from each session's first play. */
export function blocks(plays: Play[], size = BLOCK): Block[] {
  const out: Block[] = [];
  let session = -1, pos = 0;
  for (let i = 0; i < plays.length; i++) {
    const p = plays[i], prev = i ? plays[i - 1] : null;
    if (!prev || p.t - (prev.t + prev.ms) > SESSION_GAP) { session++; pos = 0; }
    if (pos % size === 0) out.push({ start: i, len: 0, session });
    out[out.length - 1].len++;
    pos++;
  }
  return out;
}

interface Counter { n: number; skips: number; last: number }

/**
 * One feature row per play, every row read from the state at the start of its block: the skip
 * counters, the recent and session skip rates and the skip of the play before the block. Within a
 * block only the tracks themselves (which track, artist, position, same artist as the one before)
 * vary; start reason, shuffle and hour are the block's first play's. So a block's 10 rows can be
 * written before its first track starts.
 */
export function frozenSkipFeatures(plays: Play[], bs: Block[], hourOf: (p: Play) => number = (p) => new Date(p.t).getUTCHours()): number[][] {
  const tracks = new Map<string, Counter>(), artists = new Map<string, Counter>();
  const all = { n: 0, skips: 0 };
  let recent = 0.5, sessPos = 0, sessSkips = 0, lastSession = -1;
  const X: number[][] = new Array(plays.length);
  for (const b of bs) {
    if (b.session !== lastSession) { sessPos = 0; sessSkips = 0; lastSession = b.session; }
    const first = plays[b.start], g = all.n ? all.skips / all.n : 0.5;
    const prevSkip = sessPos > 0 && isSkip(plays[b.start - 1]) ? 1 : 0;
    const sessRate = sessPos ? sessSkips / sessPos : g;
    const hour = hourOf(first), hs = Math.sin((2 * Math.PI * hour) / 24), hc = Math.cos((2 * Math.PI * hour) / 24);
    const starts = STARTS.map((r) => (first.reasonStart === r ? 1 : 0));
    const rate = (c: Counter | undefined, prior: number, k: number) => ((c?.skips ?? 0) + prior * k) / ((c?.n ?? 0) + k);
    for (let o = 0; o < b.len; o++) {
      const i = b.start + o, p = plays[i];
      const tk = tracks.get(trackKey(p)), ak = artists.get(p.artist);
      const days = (c?: Counter) => (c ? Math.log1p(Math.max(0, first.t - c.last) / 86_400_000) : 8);
      X[i] = [
        rate(tk, rate(ak, g, 5), 3), rate(ak, g, 5), Math.log1p(tk?.n ?? 0), Math.log1p(ak?.n ?? 0), days(tk), days(ak), recent,
        prevSkip, sessRate, Math.log1p(sessPos + o), o / (BLOCK - 1), o > 0 && plays[i - 1].artist === p.artist ? 1 : 0,
        first.shuffle ? 1 : 0, hs, hc, ...starts,
      ];
    }
    // the block is over: update the state with its outcomes
    for (let o = 0; o < b.len; o++) {
      const p = plays[b.start + o], sk = isSkip(p) ? 1 : 0;
      for (const [m, k] of [[tracks, trackKey(p)], [artists, p.artist]] as const) {
        const c = m.get(k) ?? { n: 0, skips: 0, last: 0 };
        c.n++; c.skips += sk; c.last = p.t; m.set(k, c);
      }
      all.n++; all.skips += sk; recent = 0.97 * recent + 0.03 * sk; sessPos++; sessSkips += sk;
    }
  }
  return X;
}

/** Hits out of the block: tracks whose call (score >= threshold means skip) matches what happened. */
export function hitsAt(scores: number[], skipped: number[], threshold: number) {
  let h = 0;
  for (let k = 0; k < scores.length; k++) if ((scores[k] >= threshold ? 1 : 0) === skipped[k]) h++;
  return h;
}

/** Hits when the caller is told the number of skips k and marks its k highest scores (ties by position). */
export function hitsTopK(scores: number[], skipped: number[]) {
  const k = skipped.reduce((s, v) => s + v, 0);
  const order = scores.map((s, i) => [s, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map(([, i]) => i);
  const call = new Array(scores.length).fill(0);
  for (const i of order.slice(0, k)) call[i] = 1;
  return call.reduce((h, c, i) => h + (c === skipped[i] ? 1 : 0), 0);
}

export const THRESHOLDS = Array.from({ length: 19 }, (_, k) => +(0.05 * (k + 1)).toFixed(2));
/** The threshold with the most hits over these rounds; ties go to the one nearest 0.5. */
export function bestThreshold(rounds: { s: number[]; y: number[] }[], grid = THRESHOLDS) {
  let best = 0.5, bestH = -1;
  for (const t of grid) {
    const h = rounds.reduce((a, r) => a + hitsAt(r.s, r.y, t), 0);
    if (h > bestH || (h === bestH && Math.abs(t - 0.5) < Math.abs(best - 0.5))) { best = t; bestH = h; }
  }
  return best;
}

/**
 * Exact accuracy of picking the highest score from (one positive, two negatives), averaged over
 * every positive and every pair of negatives. Ties are broken at random: a tie with one negative
 * counts 1/2, with both 1/3.
 */
export function tripleAccuracy(pos: number[], neg: number[]) {
  const m = neg.length;
  if (!pos.length || m < 2) return NaN;
  const pairs = (m * (m - 1)) / 2;
  let sum = 0;
  for (const s of pos) {
    let below = 0, equal = 0;
    for (const v of neg) { if (v < s) below++; else if (v === s) equal++; }
    sum += ((below * (below - 1)) / 2 + (below * equal) / 2 + (equal * (equal - 1)) / 2 / 3) / pairs;
  }
  return sum / pos.length;
}

/** Artists ranked by proper plays in [from, to), ties by the most recent play, then name. */
export function topByCount(plays: Play[], from: number, to: number): string[] {
  const c = new Map<string, number>(), last = new Map<string, number>();
  for (const p of plays) if (p.t >= from && p.t < to && !isSkip(p)) { c.set(p.artist, (c.get(p.artist) ?? 0) + 1); last.set(p.artist, p.t); }
  return [...c.keys()].sort((a, b) => c.get(b)! - c.get(a)! || last.get(b)! - last.get(a)! || (a < b ? -1 : a > b ? 1 : 0));
}

/** The week's top 5 with ties at fifth place all included; null when fewer than 5 artists were played. */
export function weekTop(plays: Play[], from: number, to: number, k = 5): Set<string> | null {
  const c = new Map<string, number>();
  for (const p of plays) if (p.t >= from && p.t < to && !isSkip(p)) c.set(p.artist, (c.get(p.artist) ?? 0) + 1);
  if (c.size < k) return null;
  const fifth = [...c.values()].sort((a, b) => b - a)[k - 1];
  return new Set([...c].filter(([, v]) => v >= fifth).map(([a]) => a));
}

export const precisionAt = (pred: string[], truth: Set<string>, k = 5) => pred.slice(0, k).filter((a) => truth.has(a)).length / k;

/** Decayed artist scores ranked, ties by most recent play then name (the model of game 3). */
export function topByDecay(plays: Play[], now: number, halfLifeDays: number): string[] {
  const s = new Map<string, number>(), last = new Map<string, number>(), k = Math.LN2 / (halfLifeDays * 86_400_000);
  for (const p of plays) if (p.t < now && !isSkip(p)) { s.set(p.artist, (s.get(p.artist) ?? 0) + Math.exp(-k * (now - p.t))); last.set(p.artist, p.t); }
  return [...s.keys()].sort((a, b) => s.get(b)! - s.get(a)! || last.get(b)! - last.get(a)! || (a < b ? -1 : a > b ? 1 : 0));
}

/** 95% Wilson interval for k successes of n. */
export function wilson(k: number, n: number, z = 1.959964): [number, number] {
  if (!n) return [NaN, NaN];
  const p = k / n, d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d, h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

/** Holm step-down adjusted p-values, in the input order. */
export function holm(ps: number[]): number[] {
  const o = ps.map((p, i) => [p, i] as const).sort((a, b) => a[0] - b[0]);
  const out = new Array(ps.length).fill(1);
  let run = 0;
  o.forEach(([p, i], r) => { run = Math.max(run, Math.min(1, (ps.length - r) * p)); out[i] = run; });
  return out;
}

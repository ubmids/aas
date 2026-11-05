// Eras: weekly snapshots (taste-map position, self mix, rhythm, podcasts, silence) and change-point
// detection with a season-sized minimum segment. Silence counts: weeks with zero plays are kept.
import { Play, Episode, isSkip } from '../parse/spotify.js';
import { TasteMap, centroid } from './map.js';
import { localClock, WEEK } from './session.js';
import { SessionRow, Selves } from './selves.js';
import { fitPca, fitScaler, dot } from './linalg.js';

const DAY = 86_400_000;

export interface Weekly { start: number[]; X: number[][]; names: string[]; plays: number[]; podcast: number[]; artists: Map<string, number>[] }

/**
 * One row per calendar week in [from, to). Empty weeks carry taste/self/hour forward and are flagged
 * silent. `selves` may be null (the ListenBrainz-user bench has no skips, so no selves): the self-mix
 * columns are then left out. `hourOf` overrides the local clock (that bench has UTC times only).
 */
export function weekly(plays: Play[], episodes: Episode[], from: number, to: number, m: TasteMap, rows: SessionRow[], selves: Selves | null, hourOf?: (p: Play) => number): Weekly {
  const w0 = Math.floor(from / WEEK), nW = Math.max(1, Math.ceil(to / WEEK) - w0);
  const clk = localClock(plays), hour = hourOf ?? ((p: Play) => clk.hour(p));
  const artists = Array.from({ length: nW }, () => new Map<string, number>());
  const hours = Array.from({ length: nW }, () => [0, 0, 0, 0]);
  const count = Array(nW).fill(0), eps = Array(nW).fill(0);
  for (const p of plays) {
    if (p.t < from || p.t >= to) continue;
    const w = Math.floor(p.t / WEEK) - w0;
    count[w]++;
    hours[w][Math.floor(hour(p) / 6)]++;
    if (!isSkip(p)) artists[w].set(p.artist, (artists[w].get(p.artist) ?? 0) + 1);
  }
  for (const e of episodes) if (e.t >= from && e.t < to) eps[Math.floor(e.t / WEEK) - w0]++;
  const K = selves?.g.k ?? 0, mix = Array.from({ length: nW }, () => Array(K).fill(0)), ns = Array(nW).fill(0);
  if (selves) for (const r of rows) {
    const w = r.week - w0;
    if (w < 0 || w >= nW) continue;
    selves.mix(r).forEach((v, c) => (mix[w][c] += v)); ns[w]++;
  }
  const cents = artists.map((a) => centroid(m, a));
  const ok = cents.map((c, i) => (c ? i : -1)).filter((i) => i >= 0);
  const pca = ok.length > 5 ? fitPca(ok.map((i) => [...cents[i]!]), 4) : null;
  const live = selves?.live ?? [];
  const raw: number[][] = [];
  let lastTaste = [0, 0, 0, 0], lastMix = live.map(() => 0), lastHours = [0.25, 0.25, 0.25, 0.25];
  for (let w = 0; w < nW; w++) {
    if (cents[w] && pca) lastTaste = pca.project([...cents[w]!]);
    if (ns[w]) lastMix = live.map((c) => mix[w][c] / ns[w]);
    if (count[w]) lastHours = hours[w].map((h) => h / count[w]);
    const pod = eps[w] / (eps[w] + count[w] || 1);
    raw.push([...lastTaste, ...lastMix, Math.log1p(count[w]), ...lastHours, pod, count[w] + eps[w] === 0 ? 1 : 0]);
  }
  const names = ['taste1', 'taste2', 'taste3', 'taste4', ...live.map((c) => `self ${c} share`), 'plays/week', 'night share', 'morning share', 'afternoon share', 'evening share', 'podcast share', 'silent'];
  const sc = fitScaler(raw);
  return {
    start: Array.from({ length: nW }, (_, w) => (w0 + w) * WEEK), X: raw.map(sc), names,
    plays: count, podcast: eps, artists,
  };
}

/** Optimal partition (mean-shift, unit variance cost) with a minimum segment length. Returns segment starts. */
export function changePoints(X: number[][], minSeg: number, penaltyScale: number): number[] {
  const n = X.length, d = X[0]?.length ?? 0;
  if (n < 2 * minSeg) return [0];
  const S = [Array(d).fill(0)], Q = [0];
  for (let i = 0; i < n; i++) { S.push(S[i].map((v, j) => v + X[i][j])); Q.push(Q[i] + dot(X[i], X[i])); }
  const cost = (a: number, b: number) => { // rows [a, b)
    let s = Q[b] - Q[a];
    for (let j = 0; j < d; j++) s -= (S[b][j] - S[a][j]) ** 2 / (b - a);
    return s;
  };
  const beta = penaltyScale * d * Math.log(n);
  const F = Array(n + 1).fill(Infinity), back = Array(n + 1).fill(0);
  F[0] = -beta;
  for (let t = minSeg; t <= n; t++) for (let s = 0; s <= t - minSeg; s++) {
    if (s > 0 && s < minSeg) continue;
    if (F[s] === Infinity) continue;
    const v = F[s] + cost(s, t) + beta;
    if (v < F[t]) { F[t] = v; back[t] = s; }
  }
  const starts: number[] = [];
  for (let t = n; t > 0; t = back[t]) starts.unshift(back[t]);
  return starts;
}

/** Artist scores that weight the current era fully and earlier plays by gamma, with decay. */
export function eraScores(plays: Play[], now: number, eraStart: number, halfLifeDays: number, gamma: number): Map<string, number> {
  const out = new Map<string, number>(), k = Math.LN2 / (halfLifeDays * DAY);
  for (const p of plays) {
    if (isSkip(p) || p.t >= now) continue;
    const w = (p.t >= eraStart ? 1 : gamma) * Math.exp(-k * (now - p.t));
    if (w > 0) out.set(p.artist, (out.get(p.artist) ?? 0) + w);
  }
  return out;
}

const iso = (t: number) => new Date(t).toISOString().slice(0, 10);

/** Plain description of each era with what changed from the one before. */
export function describeEras(wk: Weekly, starts: number[], m: TasteMap): string[] {
  const out: string[] = [];
  const bounds = [...starts, wk.X.length];
  let prev: { mean: number[]; top: string[]; cent: Float64Array | null; ppw: number } | null = null;
  for (let e = 0; e < starts.length; e++) {
    const a = bounds[e], b = bounds[e + 1], n = b - a;
    const art = new Map<string, number>();
    let plays = 0, eps = 0, silent = 0;
    for (let w = a; w < b; w++) {
      for (const [k, v] of wk.artists[w]) art.set(k, (art.get(k) ?? 0) + v);
      plays += wk.plays[w]; eps += wk.podcast[w]; if (wk.plays[w] + wk.podcast[w] === 0) silent++;
    }
    const top = [...art.entries()].sort((x, y) => y[1] - x[1]).slice(0, 4).map(([k]) => k);
    const mean = wk.names.map((_, j) => wk.X.slice(a, b).reduce((s, x) => s + x[j], 0) / n);
    const cent = centroid(m, art), ppw = plays / n;
    let line = `${iso(wk.start[a])} to ${iso(wk.start[b - 1] + WEEK - DAY)} (${n} wks): ${top.join(', ') || 'nothing'} | ${Math.round(ppw)} plays/wk`;
    if (silent) line += ` | ${silent} silent wks`;
    if (eps / (eps + plays) >= 0.01) line += ` | podcasts ${Math.round((100 * eps) / (eps + plays))}%`;
    if (prev) {
      const changes = wk.names.map((nm, j) => [nm, mean[j] - prev!.mean[j]] as const)
        .filter(([nm]) => !nm.startsWith('taste')).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1])).slice(0, 2)
        .map(([nm, dlt]) => `${nm} ${dlt > 0 ? 'up' : 'down'}`);
      const sim = cent && prev.cent ? dot(cent, prev.cent) : NaN;
      const fresh = top.filter((x) => !prev!.top.includes(x)).length;
      line += ` | changed: taste similarity ${Number.isFinite(sim) ? sim.toFixed(2) : '-'}, ${fresh}/4 new favourites, ${changes.join(', ')}`;
      line += `, volume ${Math.round(prev.ppw)} -> ${Math.round(ppw)} plays/wk`;
    }
    out.push(line);
    prev = { mean, top, cent, ppw };
  }
  return out;
}

// Taste over time: who you'll play next, when you listen, and how adventurous you are.
import { Play, isSkip, trackKey } from '../parse/spotify.js';
import { fitLogReg, predict } from './logreg.js';

const DAY = 86_400_000;

/** Artist scores from listens (not skips), each decayed with a half-life in days. */
export function artistScores(plays: Play[], now: number, halfLifeDays: number): Map<string, number> {
  const out = new Map<string, number>(), k = Math.LN2 / (halfLifeDays * DAY);
  for (const p of plays) if (!isSkip(p)) out.set(p.artist, (out.get(p.artist) ?? 0) + Math.exp(-k * (now - p.t)));
  return out;
}

export const topK = (m: Map<string, number>, k: number) =>
  [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, k).map(([a]) => a);

/** NDCG@k of a ranking against held-out listen counts per artist. */
export function ndcg(ranked: string[], truth: Map<string, number>, k = 10): number {
  const gain = (c: number) => Math.log2(1 + c);
  let dcg = 0;
  ranked.slice(0, k).forEach((a, i) => (dcg += gain(truth.get(a) ?? 0) / Math.log2(i + 2)));
  const ideal = [...truth.values()].sort((a, b) => b - a).slice(0, k);
  const idcg = ideal.reduce((s, c, i) => s + gain(c) / Math.log2(i + 2), 0);
  return idcg ? dcg / idcg : NaN;
}

/** Pick the half-life that best predicts the last `valWeeks` of the training period. */
export function tuneHalfLife(train: Play[], valWeeks = 8, grid = [3, 7, 14, 30, 60, 120, 365, 3650]): number {
  const cut = train[train.length - 1].t - valWeeks * 7 * DAY;
  const fit = train.filter((p) => p.t < cut), val = train.filter((p) => p.t >= cut);
  if (fit.length < 500 || val.length < 100) return 30;
  const truth = listenCounts(val);
  let best = 30, bestScore = -1;
  for (const h of grid) {
    const s = ndcg(topK(artistScores(fit, cut, h), 10), truth);
    if (s > bestScore) { bestScore = s; best = h; }
  }
  return best;
}

export function listenCounts(plays: Play[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const p of plays) if (!isSkip(p)) m.set(p.artist, (m.get(p.artist) ?? 0) + 1);
  return m;
}

/** Hour-of-week listening clock (168 bins), recency-weighted, Dirichlet-smoothed. */
/** `how` places a play in the week: UTC by default, or local time from localHourOfWeek. */
export function clock(plays: Play[], now: number, halfLifeDays: number, alpha = 1, how: (p: Play) => number = (p) => hourOfWeek(p.t)): number[] {
  const h = Array(168).fill(alpha), k = Math.LN2 / (halfLifeDays * DAY);
  for (const p of plays) h[how(p)] += Math.exp(-k * (now - p.t));
  const s = h.reduce((a, b) => a + b, 0);
  return h.map((v) => v / s);
}
export const hourOfWeek = (t: number) => { const d = new Date(t); return d.getUTCDay() * 24 + d.getUTCHours(); };
/** Local hour of the week (DST-correct) from a person's local clock. */
export const localHourOfWeek = (clk: { hour: (p: Play) => number; weekday: (p: Play) => number }) => (p: Play) => clk.weekday(p) * 24 + clk.hour(p);

/** Features for "is this play an artist you have never heard before?", computed causally. */
export function noveltyFeatures(plays: Play[]): { X: number[][]; y: number[] } {
  const seen = new Set<string>(), X: number[][] = [], y: number[] = [];
  let recentNew = 0.1, prev: Play | null = null, prevNew = 0;
  const STARTS = ['clickrow', 'trackdone', 'fwdbtn', 'playbtn', 'appload'];
  for (const p of plays) {
    const isNew = seen.has(p.artist) ? 0 : 1;
    const gap = prev ? Math.log1p((p.t - prev.t) / 60_000) : 5;
    X.push([recentNew, prevNew, gap, p.shuffle ? 1 : 0, ...STARTS.map((s) => (p.reasonStart === s ? 1 : 0))]);
    y.push(isNew);
    seen.add(p.artist); recentNew = 0.98 * recentNew + 0.02 * isNew; prevNew = isNew; prev = p;
  }
  return { X, y };
}
export const fitNovelty = (X: number[][], y: number[]) => fitLogReg(X, y, 1);
export { predict };

export { trackKey };

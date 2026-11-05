// Will you finish this track or skip it? Features are computed causally: walking the history in
// order, each play only sees what happened before it (plus how it was started, which is known
// at the start). The same walk serves training (before a cutoff) and scoring (after it).
import { Play, isSkip, trackKey } from '../parse/spotify.js';
import { fitLogReg, predict, LogReg } from './logreg.js';

const STARTS = ['clickrow', 'trackdone', 'fwdbtn', 'backbtn', 'playbtn', 'appload', 'remote'];
const SESSION_GAP = 30 * 60_000;

interface Counter { n: number; skips: number; last: number }

/** `hourOf`: the play's local hour (defaults to UTC). */
export function skipFeatures(plays: Play[], hourOf: (p: Play) => number = (p) => new Date(p.t).getUTCHours()): number[][] {
  const tracks = new Map<string, Counter>(), artists = new Map<string, Counter>();
  let all = { n: 0, skips: 0 };
  let recent = 0.5; // exponentially weighted recent skip rate
  const X: number[][] = [];
  let prev: Play | null = null, sessionPos = 0, sessionSkips = 0;
  for (const p of plays) {
    const newSession = !prev || p.t - (prev.t + prev.ms) > SESSION_GAP;
    if (newSession) { sessionPos = 0; sessionSkips = 0; }
    const g = all.n ? all.skips / all.n : 0.5;
    const tk = tracks.get(trackKey(p)), ak = artists.get(p.artist);
    const rate = (c: Counter | undefined, prior: number, k: number) =>
      ((c?.skips ?? 0) + prior * k) / ((c?.n ?? 0) + k);
    const hour = hourOf(p);
    const days = (c?: Counter) => (c ? Math.log1p((p.t - c.last) / 86_400_000) : 8);
    X.push([
      rate(tk, rate(ak, g, 5), 3),            // this track's skip rate, shrunk to the artist's
      rate(ak, g, 5),                          // this artist's skip rate, shrunk to the user's
      Math.log1p(tk?.n ?? 0), Math.log1p(ak?.n ?? 0),
      days(tk), days(ak),
      recent,
      newSession ? 0 : prev && isSkip(prev) ? 1 : 0, // skipped the one before
      sessionPos ? sessionSkips / sessionPos : g,
      Math.log1p(sessionPos),
      prev && !newSession && prev.artist === p.artist ? 1 : 0,
      p.shuffle ? 1 : 0,
      Math.sin((2 * Math.PI * hour) / 24), Math.cos((2 * Math.PI * hour) / 24),
      ...STARTS.map((r) => (p.reasonStart === r ? 1 : 0)),
    ]);
    // update after the play is over
    const sk = isSkip(p) ? 1 : 0;
    for (const [m, k] of [[tracks, trackKey(p)], [artists, p.artist]] as const) {
      const c = m.get(k) ?? { n: 0, skips: 0, last: 0 };
      c.n++; c.skips += sk; c.last = p.t; m.set(k, c);
    }
    all.n++; all.skips += sk;
    recent = 0.97 * recent + 0.03 * sk;
    sessionPos++; sessionSkips += sk; prev = p;
  }
  return X;
}

export function trainSkip(X: number[][], plays: Play[]): LogReg {
  return fitLogReg(X, plays.map((p) => (isSkip(p) ? 1 : 0)), 1);
}
export const scoreSkip = (m: LogReg, x: number[]) => predict(m, x);
/** Baseline: the artist's smoothed skip rate so far (feature 1). */
export const baselineSkip = (x: number[]) => x[1];

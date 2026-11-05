// Selves (Option A): the music and the way you listen define a self; surroundings only describe it.
// Unit = session. Session vector = where its artists sit on the taste map (mean + spread) plus
// behaviour (pick vs autoplay share, skip rate, repeat rate, length). Gaussian mixture on a
// PCA-reduced, standardised vector, k by BIC in 2..7 (penalty and a share floor optional); a self must recur in 3+ different weeks.
// Time, device and place are NOT inputs, so they can be used to test the selves.
import { Play, isSkip } from '../parse/spotify.js';
import { TasteMap, vecOf } from './map.js';
import { Session, sessions, behaviour, deviceClass, localClock, WEEK } from './session.js';
import { fitPca, fitScaler } from './linalg.js';
import { Gmm, fitGmm, bic, assign, posterior, dropComponent, logLik } from './gmm.js';

export interface SessionRow { s: Session; music: number[]; spread: number; beh: ReturnType<typeof behaviour>; week: number }

export function sessionRows(plays: Play[], m: TasteMap): SessionRow[] {
  const out: SessionRow[] = [];
  for (const s of sessions(plays)) {
    if (s.plays.length < 3) continue;
    const v = Array(m.dim).fill(0);
    let tot = 0;
    for (const p of s.plays) {
      const a = vecOf(m, p.artist);
      if (!a) continue;
      const w = Math.min(p.ms, 300_000) + 1;
      for (let j = 0; j < m.dim; j++) v[j] += w * a[j];
      tot += w;
    }
    if (!tot) continue;
    const music = v.map((x) => x / tot);
    out.push({ s, music, spread: 1 - Math.hypot(...music), beh: behaviour(s), week: Math.floor(s.start / WEEK) });
  }
  return out;
}

export interface Selves { g: Gmm; vector: (r: SessionRow) => number[]; label: (r: SessionRow) => number; mix: (r: SessionRow) => number[]; live: number[] }

export interface SelvesOptions {
  kMin?: number; kMax?: number; minWeeks?: number;
  k?: number; // fixed number of components (skips the search)
  penalty?: number; // multiplies BIC's parameter term (1 = plain BIC)
  floor?: number; // drop selves holding less than this share of sessions
  cache?: Map<number, Gmm>; // fitted mixtures by k for these same rows, shared across options
}

export function fitSelves(rows: SessionRow[], opt: SelvesOptions | number = {}, kMax_ = 7, minWeeks_ = 3): Selves {
  const o: SelvesOptions = typeof opt === 'number' ? { kMin: opt, kMax: kMax_, minWeeks: minWeeks_ } : opt;
  const kMin = o.kMin ?? 2, kMax = o.kMax ?? 7, minWeeks = o.minWeeks ?? 3, pen = o.penalty ?? 1, floor = o.floor ?? 0;
  const pca = fitPca(rows.map((r) => r.music), 8);
  const raw = (r: SessionRow) => [...pca.project(r.music), r.spread, r.beh.pick, r.beh.skip, r.beh.repeat, r.beh.logLen];
  const scale = fitScaler(rows.map(raw));
  const vector = (r: SessionRow) => scale(raw(r));
  const X = rows.map(vector);
  let best: Gmm | null = null, bestBic = Infinity;
  const ks = o.k ? [Math.min(o.k, Math.max(1, Math.floor(X.length / 40)))] : Array.from({ length: Math.max(0, Math.min(kMax, Math.floor(X.length / 40)) - kMin + 1) }, (_, i) => kMin + i);
  for (const k of ks) {
    const g = o.cache?.get(k) ?? fitGmm(X, k, 3 + k).g;
    o.cache?.set(k, g);
    const b = bic(g, X, pen);
    if (b < bestBic) { bestBic = b; best = g; }
  }
  if (!best) best = fitGmm(X, 1, 3).g;
  const g: Gmm = { ...best, w: [...best.w] }; // the drops below only touch the weights
  // a self must be a real share of the listening: drop the smallest below the floor, one at a time
  for (let changed = floor > 0; changed; ) {
    changed = false;
    const n = Array(g.k).fill(0);
    for (const x of X) n[assign(g, x)]++;
    const live = g.w.map((w, c) => (w > 0 ? c : -1)).filter((c) => c >= 0);
    const small = live.filter((c) => n[c] / X.length < floor).sort((a, b) => n[a] - n[b]);
    if (small.length && live.length > 1) { dropComponent(g, small[0]); changed = true; }
  }
  // a self must recur: drop components whose sessions span fewer than minWeeks distinct weeks
  for (let changed = true; changed; ) {
    changed = false;
    const weeks = Array.from({ length: g.k }, () => new Set<number>());
    X.forEach((x, i) => weeks[assign(g, x)].add(rows[i].week));
    const live = g.w.map((w, c) => (w > 0 ? c : -1)).filter((c) => c >= 0);
    for (const c of live) if (weeks[c].size < minWeeks && live.length > 1) { dropComponent(g, c); changed = true; break; }
  }
  return {
    g, vector,
    label: (r) => assign(g, vector(r)),
    mix: (r) => posterior(g, vector(r)),
    live: g.w.map((w, c) => (w > 0 ? c : -1)).filter((c) => c >= 0),
  };
}

export const PENALTIES = [1, 2, 4, 8, 16];
/**
 * Selves with the BIC penalty chosen on validation: fit on sessions before `vcut`, score how well
 * each mixture explains the sessions after it (log-likelihood, music and behaviour only, so the
 * surroundings stay free for the test), keep the best one with 3 to 6 selves, then refit on all rows
 * with that penalty. Falls back to plain BIC when there is too little to validate on.
 */
export function fitSelvesValidated(rows: SessionRow[], vcut: number, opt: SelvesOptions = {}): Selves & { penalty: number } {
  const fit = rows.filter((r) => r.s.start < vcut), val = rows.filter((r) => r.s.start >= vcut);
  let best = { pen: 1, ll: -Infinity };
  if (fit.length > 200 && val.length > 30) {
    const cache = new Map<number, Gmm>();
    for (const pen of PENALTIES) {
      const s = fitSelves(fit, { ...opt, penalty: pen, cache });
      if (s.live.length < 3 || s.live.length > 6) continue;
      const ll = logLik(s.g, val.map(s.vector)) / val.length;
      if (ll > best.ll + 1e-9) best = { pen, ll };
    }
  }
  return { ...fitSelves(rows, { ...opt, penalty: best.pen }), penalty: best.pen };
}

/** Surroundings of a session: never used to build selves, only to predict and describe them. */
export function contextEncoder(trainPlays: Play[]) {
  const clk = localClock(trainPlays);
  const cc = new Map<string, number>();
  for (const p of trainPlays) if (p.country) cc.set(p.country, (cc.get(p.country) ?? 0) + 1);
  const countries = [...cc.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([c]) => c);
  const devices = ['phone', 'desktop', 'web', 'tv/speaker/car', 'other', 'unknown'];
  const device = (s: Session) => {
    const n = new Map<string, number>();
    for (const p of s.plays) { const d = deviceClass(p.platform); n.set(d, (n.get(d) ?? 0) + 1); }
    return [...n.entries()].sort((a, b) => b[1] - a[1])[0][0];
  };
  const encode = (s: Session) => {
    const p0 = s.plays[0], h = clk.hour(p0), wd = clk.weekday(p0), dev = device(s);
    const shuf = s.plays.filter((p) => p.shuffle).length / s.plays.length;
    const ctry = s.plays.find((p) => p.country && p.country !== 'ZZ')?.country ?? null;
    return [
      ...Array.from({ length: 8 }, (_, b) => (Math.floor(h / 3) === b ? 1 : 0)),
      ...Array.from({ length: 7 }, (_, d) => (wd === d ? 1 : 0)),
      ...devices.map((d) => (d === dev ? 1 : 0)),
      shuf,
      ...countries.map((c) => (c === ctry ? 1 : 0)),
    ];
  };
  return { encode, device, hour: (s: Session) => clk.hour(s.plays[0]) };
}

/** Plain description of each self from measured facts (for a sanity read). */
export function describeSelves(rows: SessionRow[], selves: Selves, ctx: ReturnType<typeof contextEncoder>) {
  const by = new Map<number, SessionRow[]>();
  for (const r of rows) { const c = selves.label(r); (by.get(c) ?? by.set(c, []).get(c)!).push(r); }
  const out: string[] = [];
  for (const [c, rs] of [...by.entries()].sort((a, b) => b[1].length - a[1].length)) {
    const art = new Map<string, number>(), hours = Array(24).fill(0), dev = new Map<string, number>(), weeks = new Set<number>();
    let pick = 0, skip = 0, mins = 0;
    for (const r of rs) {
      for (const p of r.s.plays) if (!isSkip(p)) art.set(p.artist, (art.get(p.artist) ?? 0) + 1);
      hours[ctx.hour(r.s)]++;
      const d = ctx.device(r.s); dev.set(d, (dev.get(d) ?? 0) + 1);
      pick += r.beh.pick; skip += r.beh.skip; mins += r.beh.minutes; weeks.add(r.week);
    }
    const n = rs.length;
    const top = [...art.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([a]) => a).join(', ');
    const peak = hours.map((v, h) => [v, h]).sort((a, b) => b[0] - a[0]).slice(0, 3).map(([, h]) => `${h}h`).join('/');
    const devs = [...dev.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2).map(([d, v]) => `${d} ${Math.round((100 * v) / n)}%`).join(', ');
    out.push(`self ${c}: ${Math.round((100 * n) / rows.length)}% of sessions over ${weeks.size} weeks | ${top} | peak ${peak} local | ${devs} | picked ${Math.round((100 * pick) / n)}% skipped ${Math.round((100 * skip) / n)}% | ~${Math.round(mins / n)} min`);
  }
  return out;
}

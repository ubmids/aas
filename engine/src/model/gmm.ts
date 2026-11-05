// Gaussian mixture (full covariance) by EM with k-means++ starts; k chosen by BIC.
import { cholesky, rng } from './linalg.js';

export interface Gmm { k: number; d: number; w: number[]; mu: number[][]; L: number[][][]; logDet: number[] }

function logPdfs(g: Gmm, x: number[]): number[] {
  const out: number[] = [];
  for (let c = 0; c < g.k; c++) {
    if (g.w[c] <= 0) { out.push(-Infinity); continue; }
    // solve L z = x - mu
    const L = g.L[c], z: number[] = [];
    let q = 0;
    for (let i = 0; i < g.d; i++) {
      let s = x[i] - g.mu[c][i];
      for (let j = 0; j < i; j++) s -= L[i][j] * z[j];
      z.push(s / L[i][i]);
      q += z[i] * z[i];
    }
    out.push(Math.log(g.w[c]) - 0.5 * (q + g.logDet[c] + g.d * Math.log(2 * Math.PI)));
  }
  return out;
}

const lse = (a: number[]) => { const m = Math.max(...a); return m === -Infinity ? m : m + Math.log(a.reduce((s, v) => s + Math.exp(v - m), 0)); };

/** Posterior over components for one point. */
export function posterior(g: Gmm, x: number[]): number[] {
  const l = logPdfs(g, x), z = lse(l);
  return l.map((v) => Math.exp(v - z));
}
export const assign = (g: Gmm, x: number[]) => { const p = posterior(g, x); return p.indexOf(Math.max(...p)); };

function mStep(X: number[][], R: number[][], k: number, reg: number): Gmm {
  const n = X.length, d = X[0].length;
  const w: number[] = [], mu: number[][] = [], L: number[][][] = [], logDet: number[] = [];
  for (let c = 0; c < k; c++) {
    let nk = 0;
    const m = Array(d).fill(0);
    for (let i = 0; i < n; i++) { nk += R[i][c]; for (let j = 0; j < d; j++) m[j] += R[i][c] * X[i][j]; }
    for (let j = 0; j < d; j++) m[j] /= nk || 1;
    const S = Array.from({ length: d }, () => Array(d).fill(0));
    for (let i = 0; i < n; i++) {
      const r = R[i][c];
      if (r < 1e-8) continue;
      for (let a = 0; a < d; a++) { const da = X[i][a] - m[a]; for (let b = 0; b <= a; b++) S[a][b] += r * da * (X[i][b] - m[b]); }
    }
    for (let a = 0; a < d; a++) { for (let b = 0; b <= a; b++) S[a][b] = S[b][a] = S[a][b] / (nk || 1); S[a][a] += reg; }
    const Lc = cholesky(S);
    w.push(nk / n); mu.push(m); L.push(Lc); logDet.push(2 * Lc.reduce((s, r, i) => s + Math.log(r[i]), 0));
  }
  return { k, d, w, mu, L, logDet };
}

export function fitGmm(X: number[][], k: number, seed = 3, iters = 100, reg = 1e-2): { g: Gmm; ll: number } {
  const n = X.length, r = rng(seed);
  // k-means++ seeding, then hard responsibilities to start
  const centers = [X[Math.floor(r() * n)]];
  const d2 = (a: number[], b: number[]) => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0);
  while (centers.length < k) {
    const dist = X.map((x) => Math.min(...centers.map((c) => d2(x, c))));
    let u = r() * dist.reduce((s, v) => s + v, 0), i = 0;
    while (i < n - 1 && (u -= dist[i]) > 0) i++;
    centers.push(X[i]);
  }
  let R: number[][] = X.map((x) => { const ds = centers.map((c) => d2(x, c)); const b = ds.indexOf(Math.min(...ds)); return centers.map((_, c) => (c === b ? 1 : 0)); });
  let g = mStep(X, R, k, reg), prev = -Infinity, ll = 0;
  for (let it = 0; it < iters; it++) {
    ll = 0;
    R = X.map((x) => { const l = logPdfs(g, x), z = lse(l); ll += z; return l.map((v) => Math.exp(v - z)); });
    g = mStep(X, R, k, reg);
    if (Math.abs(ll - prev) < 1e-4 * n) break;
    prev = ll;
  }
  return { g, ll };
}

export const logLik = (g: Gmm, X: number[][]) => X.reduce((s, x) => s + lse(logPdfs(g, x)), 0);

export function bic(g: Gmm, X: number[][], penalty = 1): number {
  const p = g.k * (g.d + (g.d * (g.d + 1)) / 2) + g.k - 1;
  return -2 * logLik(g, X) + penalty * p * Math.log(X.length);
}

/** Drop a component (weight 0) and renormalise. */
export function dropComponent(g: Gmm, c: number) {
  g.w[c] = 0;
  const s = g.w.reduce((a, b) => a + b, 0);
  g.w = g.w.map((v) => v / s);
}

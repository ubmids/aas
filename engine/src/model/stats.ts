// Small statistics used by the bench report: bootstrap intervals over people, the exact sign test,
// calibration (reliability bins, expected calibration error, Brier score) and correlations.
import { rng, eigSym } from './linalg.js';

export const median = (a: number[]) => {
  const s = a.filter(Number.isFinite).sort((x, y) => x - y);
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
export const mean = (a: number[]) => { const s = a.filter(Number.isFinite); return s.length ? s.reduce((x, y) => x + y, 0) / s.length : NaN; };

/**
 * Percentile bootstrap over units (people): resample the units with replacement B times, apply
 * `stat` to each resample, return the 2.5% and 97.5% points. Units are whatever `stat` reads.
 */
export function bootstrap<T>(units: T[], stat: (xs: T[]) => number, B = 4000, seed = 1): [number, number] {
  if (!units.length) return [NaN, NaN];
  const r = rng(seed), out: number[] = [];
  for (let b = 0; b < B; b++) {
    const s = Array.from({ length: units.length }, () => units[Math.floor(r() * units.length)]);
    const v = stat(s);
    if (Number.isFinite(v)) out.push(v);
  }
  out.sort((a, b) => a - b);
  return out.length ? [out[Math.floor(0.025 * (out.length - 1))], out[Math.ceil(0.975 * (out.length - 1))]] : [NaN, NaN];
}

/** log of n choose k */
function lchoose(n: number, k: number) { let s = 0; for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i); return s; }

/**
 * Exact two-sided sign test on paired differences: ties are dropped, then the count of positive
 * differences is compared with Binomial(n, 1/2). Returns wins, losses, ties and the p-value.
 */
export function signTest(diffs: number[]) {
  const d = diffs.filter(Number.isFinite);
  const wins = d.filter((x) => x > 0).length, losses = d.filter((x) => x < 0).length, ties = d.length - wins - losses, n = wins + losses;
  if (!n) return { wins, losses, ties, p: 1 };
  const k = Math.min(wins, losses);
  let tail = 0;
  for (let i = 0; i <= k; i++) tail += Math.exp(lchoose(n, i) - n * Math.LN2);
  return { wins, losses, ties, p: Math.min(1, 2 * tail) };
}

export interface Reliability { bins: { lo: number; hi: number; n: number; meanP: number; rate: number }[]; ece: number; brier: number; n: number }

/**
 * Calibration of probabilities p against 0/1 outcomes y: equal-width bins on [0, 1]; ECE is the
 * share-weighted mean |predicted - observed| over bins; Brier is mean (p - y)^2.
 */
export function reliability(p: number[], y: number[], nBins = 10): Reliability {
  const bins = Array.from({ length: nBins }, (_, i) => ({ lo: i / nBins, hi: (i + 1) / nBins, n: 0, sp: 0, sy: 0 }));
  let brier = 0;
  for (let i = 0; i < p.length; i++) {
    const b = bins[Math.min(nBins - 1, Math.floor(p[i] * nBins))];
    b.n++; b.sp += p[i]; b.sy += y[i];
    brier += (p[i] - y[i]) ** 2;
  }
  const n = p.length;
  const out = bins.map((b) => ({ lo: b.lo, hi: b.hi, n: b.n, meanP: b.n ? b.sp / b.n : NaN, rate: b.n ? b.sy / b.n : NaN }));
  const ece = out.reduce((s, b) => s + (b.n ? (b.n / n) * Math.abs(b.meanP - b.rate) : 0), 0);
  return { bins: out, ece, brier: n ? brier / n : NaN, n };
}

export function pearson(x: number[], y: number[]) {
  const n = x.length, mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : NaN;
}
/** Average ranks, ties shared. */
export function ranks(x: number[]) {
  const o = x.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]), r: number[] = Array(x.length);
  for (let i = 0; i < o.length; ) { let j = i; while (j < o.length && o[j][0] === o[i][0]) j++; for (let k = i; k < j; k++) r[o[k][1]] = (i + j - 1) / 2; i = j; }
  return r;
}
export const spearman = (x: number[], y: number[]) => pearson(ranks(x), ranks(y));

// ---------------------------------------------------------------- factor structure (round 3)

/** Inverse of the standard normal CDF (Acklam's rational approximation, relative error below 1.2e-9). */
export function qnorm(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const lo = 0.02425;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - lo) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Normal scores (van der Waerden): each value replaced by qnorm(rank / (n + 1)), ties sharing their average rank. */
export const normalScores = (x: number[]) => { const r = ranks(x); return r.map((v) => qnorm((v + 1) / (x.length + 1))); };

/** Correlation matrix of the columns of X (rows are people). */
export function corrMatrix(X: number[][]): number[][] {
  const d = X[0].length, cols = Array.from({ length: d }, (_, j) => X.map((r) => r[j]));
  return cols.map((a) => cols.map((b) => pearson(a, b)));
}

/**
 * Horn's parallel analysis: eigenvalues of the correlation matrix against the 95th percentile of
 * eigenvalues from the same data with every column shuffled independently (no shared structure).
 * Keeps components in order while the observed eigenvalue beats that percentile.
 */
export function parallelAnalysis(X: number[][], B = 1000, seed = 5) {
  const observed = eigSym(corrMatrix(X)).values, d = observed.length, r = rng(seed), sims: number[][] = Array.from({ length: d }, () => []);
  for (let b = 0; b < B; b++) {
    const Y = X.map((row) => [...row]);
    for (let j = 0; j < d; j++) for (let i = Y.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [Y[i][j], Y[k][j]] = [Y[k][j], Y[i][j]]; }
    eigSym(corrMatrix(Y)).values.forEach((v, j) => sims[j].push(v));
  }
  const p95 = sims.map((s) => { const o = s.sort((a, b) => a - b); return o[Math.floor(0.95 * (o.length - 1))]; });
  let keep = 0;
  while (keep < d && observed[keep] > p95[keep]) keep++;
  return { observed, p95, keep };
}

/** Varimax rotation of a d x k loading matrix (Kaiser's pairwise angles, with Kaiser row normalisation). */
export function varimax(Lo: number[][], iters = 100): number[][] {
  const d = Lo.length, k = Lo[0]?.length ?? 0;
  if (k < 2) return Lo.map((r) => [...r]);
  const h = Lo.map((r) => Math.sqrt(r.reduce((s, v) => s + v * v, 0)) || 1);
  const L = Lo.map((r, i) => r.map((v) => v / h[i]));
  for (let it = 0; it < iters; it++) {
    let moved = 0;
    for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) {
      let A = 0, B = 0, C = 0, D = 0;
      for (let i = 0; i < d; i++) {
        const u = L[i][p] ** 2 - L[i][q] ** 2, v = 2 * L[i][p] * L[i][q];
        A += u; B += v; C += u * u - v * v; D += 2 * u * v;
      }
      const phi = Math.atan2(D - (2 * A * B) / d, C - (A * A - B * B) / d) / 4;
      if (Math.abs(phi) < 1e-10) continue;
      moved++;
      const c = Math.cos(phi), s = Math.sin(phi);
      for (let i = 0; i < d; i++) { const a = L[i][p], b = L[i][q]; L[i][p] = c * a + s * b; L[i][q] = -s * a + c * b; }
    }
    if (!moved) break;
  }
  return L.map((r, i) => r.map((v) => v * h[i]));
}

/** Principal-component loadings (eigenvector x sqrt(eigenvalue)) of the correlation matrix, top k, varimax-rotated when k >= 2. */
export function pcaLoadings(X: number[][], k: number): number[][] {
  const { values, vectors } = eigSym(corrMatrix(X)), d = values.length;
  const L = Array.from({ length: d }, (_, i) => Array.from({ length: k }, (_, c) => vectors[c][i] * Math.sqrt(Math.max(0, values[c]))));
  const R = k >= 2 ? varimax(L) : L;
  // sign convention: each component's largest loading is positive
  for (let c = 0; c < k; c++) { let big = 0; for (let i = 0; i < d; i++) if (Math.abs(R[i][c]) > Math.abs(R[big][c])) big = i; if (R[big][c] < 0) for (let i = 0; i < d; i++) R[i][c] = -R[i][c]; }
  return R;
}

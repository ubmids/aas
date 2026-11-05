// Small L2-regularised logistic regression (Newton / IRLS). Fine for tens of features.

export interface LogReg { w: number[]; mean: number[]; sd: number[] }

export function fitLogReg(X: number[][], y: number[], l2 = 1, iters = 25): LogReg {
  const n = X.length, d = X[0].length;
  const mean = Array(d).fill(0), sd = Array(d).fill(0);
  for (const x of X) x.forEach((v, j) => (mean[j] += v / n));
  for (const x of X) x.forEach((v, j) => (sd[j] += (v - mean[j]) ** 2 / n));
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  const Z = X.map((x) => [1, ...x.map((v, j) => (v - mean[j]) / sd[j])]);
  const D = d + 1;
  let w = Array(D).fill(0);
  for (let it = 0; it < iters; it++) {
    const g = Array(D).fill(0);
    const H = Array.from({ length: D }, () => Array(D).fill(0));
    for (let i = 0; i < n; i++) {
      const z = Z[i];
      const p = sigmoid(dot(w, z));
      const r = p - y[i], h = p * (1 - p);
      for (let a = 0; a < D; a++) {
        g[a] += r * z[a];
        for (let b = a; b < D; b++) H[a][b] += h * z[a] * z[b];
      }
    }
    for (let a = 1; a < D; a++) { g[a] += l2 * w[a]; H[a][a] += l2; }
    for (let a = 0; a < D; a++) for (let b = 0; b < a; b++) H[a][b] = H[b][a];
    H[0][0] += 1e-6;
    const step = solve(H, g);
    w = w.map((v, a) => v - step[a]);
    if (Math.max(...step.map(Math.abs)) < 1e-6) break;
  }
  return { w, mean, sd };
}

export function predict(m: LogReg, x: number[]): number {
  let s = m.w[0];
  for (let j = 0; j < x.length; j++) s += m.w[j + 1] * ((x[j] - m.mean[j]) / m.sd[j]);
  return sigmoid(s);
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, z))));
const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);

function solve(A: number[][], b: number[]): number[] {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const v = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / v;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / (r[i] || 1e-12));
}

/** Area under the ROC curve (probability a random positive outranks a random negative). */
export function auc(scores: number[], labels: number[]): number {
  const idx = scores.map((s, i) => [s, labels[i]] as const).sort((a, b) => a[0] - b[0]);
  let rank = 0, posRanks = 0, pos = 0;
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j < idx.length && idx[j][0] === idx[i][0]) j++;
    const avg = (i + j + 1) / 2;
    for (let k = i; k < j; k++) if (idx[k][1]) { posRanks += avg; pos++; }
    rank = j; i = j;
  }
  const neg = idx.length - pos;
  return pos && neg ? (posRanks - (pos * (pos + 1)) / 2) / (pos * neg) : NaN;
}

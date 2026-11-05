// Small dense and sparse linear algebra, enough for the taste map, PCA and mixtures. No deps.

export type Vec = Float64Array<ArrayBufferLike>;

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
export const norm = (a: ArrayLike<number>) => Math.sqrt(dot(a, a));
export function normalize(a: Vec): Vec {
  const n = norm(a) || 1;
  for (let i = 0; i < a.length; i++) a[i] /= n;
  return a;
}
export const cosine = (a: ArrayLike<number>, b: ArrayLike<number>) => dot(a, b) / ((norm(a) * norm(b)) || 1);

/** Seeded uniform RNG (LCG), so every bench run is the same. */
export function rng(seed = 7) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}
export function gauss(r: () => number) {
  const u = Math.max(1e-12, r()), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Symmetric sparse matrix in CSR form. */
export interface Csr { n: number; rowPtr: Int32Array; col: Int32Array; val: Float64Array }

/** Y = A X where X is n x m row-major. */
export function csrMul(A: Csr, X: Vec, m: number): Vec {
  const Y = new Float64Array(A.n * m);
  for (let i = 0; i < A.n; i++) {
    const yi = i * m;
    for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
      const v = A.val[p], xj = A.col[p] * m;
      for (let c = 0; c < m; c++) Y[yi + c] += v * X[xj + c];
    }
  }
  return Y;
}

/** Orthonormalise the m columns of an n x m row-major matrix in place (modified Gram-Schmidt). */
export function orthonormalize(Y: Float64Array, n: number, m: number) {
  for (let c = 0; c < m; c++) {
    for (let k = 0; k < c; k++) {
      let d = 0;
      for (let i = 0; i < n; i++) d += Y[i * m + c] * Y[i * m + k];
      for (let i = 0; i < n; i++) Y[i * m + c] -= d * Y[i * m + k];
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += Y[i * m + c] ** 2;
    s = Math.sqrt(s) || 1;
    for (let i = 0; i < n; i++) Y[i * m + c] /= s;
  }
}

/** Eigen-decomposition of a small symmetric matrix (cyclic Jacobi). Returns values desc + column vectors. */
export function eigSym(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length, a = A.map((r) => [...r]);
  const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = a.map((r, i) => [r[i], i] as const).sort((x, y) => y[0] - x[0]);
  return { values: order.map(([e]) => e), vectors: order.map(([, i]) => v.map((r) => r[i])) };
}

/**
 * Top-k eigenpairs of a large sparse symmetric matrix by randomised range finding
 * (Halko, Martinsson, Tropp 2011). Returns n x k row-major vectors and the values.
 */
export function topEigSparse(A: Csr, k: number, oversample = 16, power = 3, seed = 11) {
  const m = k + oversample, r = rng(seed), n = A.n;
  let Y: Vec = new Float64Array(n * m);
  for (let i = 0; i < Y.length; i++) Y[i] = gauss(r);
  Y = csrMul(A, Y, m);
  for (let q = 0; q < power; q++) { orthonormalize(Y, n, m); Y = csrMul(A, Y, m); }
  orthonormalize(Y, n, m);
  const AQ = csrMul(A, Y, m);
  const B = Array.from({ length: m }, () => Array(m).fill(0));
  for (let i = 0; i < n; i++) for (let a = 0; a < m; a++) {
    const qa = Y[i * m + a];
    if (qa === 0) continue;
    for (let b = 0; b < m; b++) B[a][b] += qa * AQ[i * m + b];
  }
  for (let a = 0; a < m; a++) for (let b = 0; b < a; b++) B[a][b] = B[b][a] = (B[a][b] + B[b][a]) / 2;
  const { values, vectors } = eigSym(B);
  const U = new Float64Array(n * k);
  for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) {
    let s = 0;
    for (let a = 0; a < m; a++) s += Y[i * m + a] * vectors[c][a];
    U[i * k + c] = s;
  }
  return { U, values: values.slice(0, k) };
}

/** PCA fitted on rows of X; returns a projector to the top-k components. */
export function fitPca(X: number[][], k: number) {
  const d = X[0].length, n = X.length, mean = Array(d).fill(0);
  for (const x of X) for (let j = 0; j < d; j++) mean[j] += x[j] / n;
  const C = Array.from({ length: d }, () => Array(d).fill(0));
  for (const x of X) for (let a = 0; a < d; a++) { const da = x[a] - mean[a]; for (let b = a; b < d; b++) C[a][b] += (da * (x[b] - mean[b])) / n; }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a][b] = C[b][a];
  const { values, vectors } = eigSym(C);
  const comps = vectors.slice(0, k);
  const total = values.reduce((s, v) => s + Math.max(0, v), 0);
  return {
    explained: values.slice(0, k).reduce((s, v) => s + Math.max(0, v), 0) / (total || 1),
    project: (x: number[]) => comps.map((c) => c.reduce((s, w, j) => s + w * (x[j] - mean[j]), 0)),
  };
}

/** Column standardiser fitted on rows of X. */
export function fitScaler(X: number[][]) {
  const d = X[0].length, n = X.length, mean = Array(d).fill(0), sd = Array(d).fill(0);
  for (const x of X) for (let j = 0; j < d; j++) mean[j] += x[j] / n;
  for (const x of X) for (let j = 0; j < d; j++) sd[j] += (x[j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  return (x: number[]) => x.map((v, j) => (v - mean[j]) / sd[j]);
}

/** Cholesky of a small SPD matrix: lower-triangular L with A = L L^T. */
export function cholesky(A: number[][]): number[][] {
  const n = A.length, L = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-10)) : s / L[j][j];
  }
  return L;
}

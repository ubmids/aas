// Small gradient-boosted trees for binary outcomes (logistic loss, Newton leaf values), on features
// cut into at most 32 quantile bins. Used only as a check on whether a logistic regression leaves
// accuracy on the table.

export interface GbtOptions { trees?: number; depth?: number; lr?: number; lambda?: number; minHess?: number; bins?: number }
interface Node { f: number; bin: number; left: Node | number; right: Node | number }
export interface Gbt { cuts: number[][]; trees: (Node | number)[]; base: number; lr: number }

const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, z))));

function quantileCuts(col: number[], bins: number): number[] {
  const s = [...col].sort((a, b) => a - b), cuts: number[] = [];
  for (let k = 1; k < bins; k++) { const v = s[Math.floor((k * s.length) / bins)]; if (!cuts.length || v > cuts[cuts.length - 1]) cuts.push(v); }
  return cuts;
}
const binOf = (cuts: number[], v: number) => { let lo = 0, hi = cuts.length; while (lo < hi) { const m = (lo + hi) >> 1; if (v < cuts[m]) hi = m; else lo = m + 1; } return lo; };

export function fitGbt(X: number[][], y: number[], opt: GbtOptions = {}): Gbt {
  const T = opt.trees ?? 150, D = opt.depth ?? 3, lr = opt.lr ?? 0.1, lambda = opt.lambda ?? 1, minHess = opt.minHess ?? 10, nb = opt.bins ?? 32;
  const n = X.length, d = X[0].length;
  const cuts = Array.from({ length: d }, (_, j) => quantileCuts(X.map((x) => x[j]), nb));
  const B = Array.from({ length: d }, (_, j) => Uint8Array.from(X.map((x) => binOf(cuts[j], x[j]))));
  const mean = y.reduce((s, v) => s + v, 0) / n, base = Math.log(mean / (1 - mean));
  const F = new Float64Array(n).fill(base), g = new Float64Array(n), h = new Float64Array(n);
  const trees: (Node | number)[] = [];
  const grow = (idx: Uint32Array, depth: number): Node | number => {
    let G = 0, H = 0;
    for (const i of idx) { G += g[i]; H += h[i]; }
    const leaf = -G / (H + lambda);
    if (depth >= D || H < 2 * minHess) return leaf;
    let best = 0, bf = -1, bb = -1;
    const parent = (G * G) / (H + lambda);
    for (let j = 0; j < d; j++) {
      const k = cuts[j].length + 1, gs = new Float64Array(k), hs = new Float64Array(k), col = B[j];
      for (const i of idx) { gs[col[i]] += g[i]; hs[col[i]] += h[i]; }
      let gl = 0, hl = 0;
      for (let b = 0; b < k - 1; b++) {
        gl += gs[b]; hl += hs[b];
        const gr = G - gl, hr = H - hl;
        if (hl < minHess || hr < minHess) continue;
        const gain = (gl * gl) / (hl + lambda) + (gr * gr) / (hr + lambda) - parent;
        if (gain > best) { best = gain; bf = j; bb = b; }
      }
    }
    if (bf < 0) return leaf;
    const col = B[bf], L: number[] = [], R: number[] = [];
    for (const i of idx) (col[i] <= bb ? L : R).push(i);
    return { f: bf, bin: bb, left: grow(Uint32Array.from(L), depth + 1), right: grow(Uint32Array.from(R), depth + 1) };
  };
  const all = Uint32Array.from({ length: n }, (_, i) => i);
  const walkB = (t: Node | number, i: number): number => { while (typeof t !== 'number') t = B[t.f][i] <= t.bin ? t.left : t.right; return t; };
  for (let k = 0; k < T; k++) {
    for (let i = 0; i < n; i++) { const p = sigmoid(F[i]); g[i] = p - y[i]; h[i] = Math.max(p * (1 - p), 1e-6); }
    const t = grow(all, 0);
    trees.push(t);
    for (let i = 0; i < n; i++) F[i] += lr * walkB(t, i);
  }
  return { cuts, trees, base, lr };
}

export function predictGbt(m: Gbt, x: number[]): number {
  let z = m.base;
  for (let t of m.trees) {
    while (typeof t !== 'number') t = binOf(m.cuts[t.f], x[t.f]) <= t.bin ? t.left : t.right;
    z += m.lr * t;
  }
  return sigmoid(z);
}

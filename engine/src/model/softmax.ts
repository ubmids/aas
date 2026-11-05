// Multinomial logistic regression with L2, full-batch gradient descent with momentum.
export interface Softmax { W: number[][]; k: number }

export function fitSoftmax(X: number[][], y: number[], k: number, l2 = 1e-2, iters = 400, lr = 0.5): Softmax {
  const n = X.length, d = X[0].length + 1;
  const W = Array.from({ length: k }, () => Array(d).fill(0)), V = W.map((r) => r.map(() => 0));
  for (let it = 0; it < iters; it++) {
    const G = W.map((r) => r.map(() => 0));
    for (let i = 0; i < n; i++) {
      const p = probs({ W, k }, X[i]);
      for (let c = 0; c < k; c++) {
        const e = (p[c] - (y[i] === c ? 1 : 0)) / n;
        G[c][0] += e;
        for (let j = 1; j < d; j++) G[c][j] += e * X[i][j - 1];
      }
    }
    for (let c = 0; c < k; c++) for (let j = 0; j < d; j++) {
      if (j) G[c][j] += l2 * W[c][j];
      V[c][j] = 0.9 * V[c][j] - lr * G[c][j];
      W[c][j] += V[c][j];
    }
  }
  return { W, k };
}

export function probs(m: Softmax, x: number[]): number[] {
  const z = m.W.map((w) => w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0));
  const mx = Math.max(...z), e = z.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

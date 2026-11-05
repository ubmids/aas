import { test } from 'node:test';
import assert from 'node:assert/strict';
import { qnorm, normalScores, parallelAnalysis, pcaLoadings, varimax, corrMatrix } from '../src/model/stats.js';
import { rng, gauss } from '../src/model/linalg.js';

test('qnorm inverts the normal CDF at known points', () => {
  assert.ok(Math.abs(qnorm(0.5)) < 1e-12);
  assert.ok(Math.abs(qnorm(0.975) - 1.959963985) < 1e-6);
  assert.ok(Math.abs(qnorm(0.01) + 2.326347874) < 1e-6);
  assert.ok(Math.abs(qnorm(0.3) + qnorm(0.7)) < 1e-9);
});

test('normal scores keep the order and are symmetric', () => {
  const z = normalScores([10, 3, 7, 1, 5]);
  assert.deepEqual([...z.keys()].sort((a, b) => z[a] - z[b]), [3, 1, 4, 2, 0]);
  assert.ok(Math.abs(z.reduce((s, v) => s + v, 0)) < 1e-9);
});

// two latent dials, three indicators each, plus noise
function twoFactorData(n: number, seed = 3) {
  const r = rng(seed), X: number[][] = [];
  for (let i = 0; i < n; i++) {
    const f = gauss(r), g = gauss(r);
    X.push([f + 0.5 * gauss(r), f + 0.5 * gauss(r), -f + 0.5 * gauss(r), g + 0.5 * gauss(r), g + 0.5 * gauss(r), g + 0.5 * gauss(r)]);
  }
  return X;
}

test('parallel analysis keeps two components for two planted dials, none for noise', () => {
  assert.equal(parallelAnalysis(twoFactorData(300), 200).keep, 2);
  const r = rng(9), noise = Array.from({ length: 300 }, () => Array.from({ length: 6 }, () => gauss(r)));
  assert.equal(parallelAnalysis(noise, 200).keep, 0);
});

test('varimax loadings put each planted dial on its own component', () => {
  const L = pcaLoadings(twoFactorData(400), 2);
  const comp = (i: number) => (Math.abs(L[i][0]) > Math.abs(L[i][1]) ? 0 : 1);
  assert.equal(comp(0), comp(1)); assert.equal(comp(0), comp(2));
  assert.equal(comp(3), comp(4)); assert.equal(comp(3), comp(5));
  assert.notEqual(comp(0), comp(3));
  assert.ok(Math.sign(L[0][comp(0)]) === -Math.sign(L[2][comp(0)]));
  for (const row of L) assert.ok(Math.max(...row.map(Math.abs)) > 0.7);
});

test('varimax keeps each row\'s communality', () => {
  const L = [[0.7, 0.3], [0.6, 0.4], [0.2, 0.8], [0.1, 0.7]], R = varimax(L);
  L.forEach((row, i) => assert.ok(Math.abs(row[0] ** 2 + row[1] ** 2 - (R[i][0] ** 2 + R[i][1] ** 2)) < 1e-9));
  assert.ok(Math.abs(corrMatrix([[1, 2], [2, 4], [3, 7]])[0][1] - 0.9934) < 1e-3);
});

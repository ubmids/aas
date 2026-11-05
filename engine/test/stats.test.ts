import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bootstrap, signTest, reliability, median, spearman, ranks } from '../src/model/stats.js';

test('sign test matches the binomial by hand', () => {
  // 10 wins, 0 losses: p = 2 * 0.5^10
  const s = signTest(Array(10).fill(1));
  assert.equal(s.wins, 10);
  assert.ok(Math.abs(s.p - 2 / 1024) < 1e-12);
  // ties are dropped; 7 wins 3 losses: p = 2 * (1 + 10 + 45 + 120) / 1024
  const t = signTest([...Array(7).fill(0.1), ...Array(3).fill(-0.1), 0, 0]);
  assert.equal(t.ties, 2);
  assert.ok(Math.abs(t.p - (2 * 176) / 1024) < 1e-12);
  assert.equal(signTest([1, -1]).p, 1);
});

test('bootstrap interval covers the median of a tight sample and is ordered', () => {
  const xs = [0.8, 0.82, 0.84, 0.85, 0.86, 0.88, 0.9];
  const [lo, hi] = bootstrap(xs, median);
  assert.ok(lo <= median(xs) && median(xs) <= hi);
  assert.ok(lo >= 0.8 && hi <= 0.9);
});

test('reliability: perfectly calibrated bins give ECE 0, overconfident ones do not', () => {
  const p: number[] = [], y: number[] = [];
  for (let i = 0; i < 100; i++) { p.push(0.25); y.push(i % 4 === 0 ? 1 : 0); }
  for (let i = 0; i < 100; i++) { p.push(0.75); y.push(i % 4 === 0 ? 0 : 1); }
  const r = reliability(p, y);
  assert.ok(r.ece < 1e-12);
  assert.ok(Math.abs(r.brier - 0.1875) < 1e-12);
  const bad = reliability(p.map((v) => (v < 0.5 ? 0.05 : 0.95)), y);
  assert.ok(Math.abs(bad.ece - 0.2) < 1e-12);
});

test('ranks share ties and spearman is 1 for monotone data', () => {
  assert.deepEqual(ranks([3, 1, 3, 2]), [2.5, 0, 2.5, 1]);
  assert.ok(Math.abs(spearman([1, 2, 3, 4], [1, 4, 9, 16]) - 1) < 1e-12);
});

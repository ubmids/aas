import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Play } from '../src/parse/spotify.js';
import { blocks, frozenSkipFeatures, hitsAt, hitsTopK, bestThreshold, tripleAccuracy, weekTop, topByCount, precisionAt, wilson, holm, FROZEN_FEATURES, ARTIST_RATE } from '../src/model/game.js';

const MIN = 60_000, DAY = 86_400_000;
const play = (t: number, artist: string, o: Partial<Play> = {}): Play =>
  ({ t, ms: 200_000, track: `${artist} a`, artist, album: null, uri: null, reasonStart: 'trackdone', reasonEnd: 'trackdone', shuffle: false, skipped: null, platform: null, country: 'GB', ...o });

test('blocks tile each session from its first play in tens', () => {
  const ps: Play[] = [];
  for (let k = 0; k < 23; k++) ps.push(play(k * 4 * MIN, 'A'));
  for (let k = 0; k < 5; k++) ps.push(play(DAY + k * 4 * MIN, 'B'));
  const bs = blocks(ps);
  assert.deepEqual(bs.map((b) => [b.start, b.len, b.session]), [[0, 10, 0], [10, 10, 0], [20, 3, 0], [23, 5, 1]]);
});

test('frozen features do not move with outcomes inside the block', () => {
  // 10 plays of A then a block of 10 where the outcomes differ: rows must be identical either way
  const base = Array.from({ length: 10 }, (_, k) => play(k * 4 * MIN, 'A'));
  const mk = (ms: number) => [...base, ...Array.from({ length: 10 }, (_, k) => play((10 + k) * 4 * MIN, k % 2 ? 'A' : 'B', { ms, reasonStart: k ? 'fwdbtn' : 'clickrow' }))];
  const a = mk(5_000), b = mk(200_000);
  const Xa = frozenSkipFeatures(a, blocks(a)), Xb = frozenSkipFeatures(b, blocks(b));
  assert.equal(Xa[0].length, FROZEN_FEATURES.length);
  for (let i = 10; i < 20; i++) assert.deepEqual(Xa[i], Xb[i]);
  // the start reason is the block's first play's for every row
  const ci = FROZEN_FEATURES.indexOf('start_clickrow'), fi = FROZEN_FEATURES.indexOf('start_fwdbtn');
  for (let i = 10; i < 20; i++) { assert.equal(Xa[i][ci], 1); assert.equal(Xa[i][fi], 0); }
  // but the next block sees them: A's frozen skip rate rises after the skipped block
  const c = [...a, play(20 * 4 * MIN, 'A')], Xc = frozenSkipFeatures(c, blocks(c));
  assert.ok(Xc[20][ARTIST_RATE] > Xa[10][ARTIST_RATE] + 0.1);
});

test('hits: threshold calls and top-k calls', () => {
  assert.equal(hitsAt([0.9, 0.1, 0.6, 0.4], [1, 0, 0, 0], 0.5), 3);
  assert.equal(hitsTopK([0.9, 0.1, 0.6, 0.4], [1, 0, 0, 1]), 2);
  assert.equal(hitsTopK([0.9, 0.1, 0.6, 0.4], [0, 0, 0, 0]), 4);
  // ties in hits go to the threshold nearest 0.5
  assert.equal(bestThreshold([{ s: [0.2, 0.8], y: [0, 1] }]), 0.5);
  assert.equal(bestThreshold([{ s: [0.1, 0.15], y: [0, 1] }]), 0.15);
});

test('triple accuracy is exact, with ties shared', () => {
  assert.equal(tripleAccuracy([1], [0, 0]), 1);
  assert.equal(tripleAccuracy([0], [1, 1]), 0);
  assert.ok(Math.abs(tripleAccuracy([0.5], [0.5, 0.5]) - 1 / 3) < 1e-12);
  assert.ok(Math.abs(tripleAccuracy([0.5], [0.5, 0]) - 1 / 2) < 1e-12);
  // a positive above 2 of 4 negatives wins 1 of the 6 pairs
  assert.ok(Math.abs(tripleAccuracy([0.5], [0.1, 0.2, 0.8, 0.9]) - 1 / 6) < 1e-12);
  // all-equal scores give chance
  assert.ok(Math.abs(tripleAccuracy([0, 0], [0, 0, 0]) - 1 / 3) < 1e-12);
});

test('week top 5 keeps ties at fifth place; precision at 5', () => {
  const ps: Play[] = [];
  const counts: [string, number][] = [['A', 5], ['B', 4], ['C', 3], ['D', 2], ['E', 1], ['F', 1], ['G', 0]];
  let t = 0;
  for (const [a, n] of counts) for (let k = 0; k < n; k++) ps.push(play((t += MIN * 5), a));
  ps.push(play((t += MIN * 5), 'G', { ms: 1000 }));
  const top = weekTop(ps, 0, DAY)!;
  assert.deepEqual([...top].sort(), ['A', 'B', 'C', 'D', 'E', 'F']);
  assert.equal(weekTop(ps.filter((p) => 'AB'.includes(p.artist)), 0, DAY), null);
  assert.deepEqual(topByCount(ps, 0, DAY).slice(0, 4), ['A', 'B', 'C', 'D']);
  assert.equal(precisionAt(['A', 'B', 'G', 'F', 'Z'], top), 0.6);
});

test('Wilson interval and Holm', () => {
  const [lo, hi] = wilson(11, 13);
  assert.ok(Math.abs(lo - 0.578) < 0.002 && hi > 0.95);
  assert.ok(wilson(10, 13)[0] < 0.5);
  assert.deepEqual(holm([0.01, 0.04, 0.03]).map((p) => +p.toFixed(4)), [0.03, 0.06, 0.06]);
});

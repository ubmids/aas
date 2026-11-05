import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildLbMap, blendMaps, LbTable } from '../src/model/lbmap.js';
import { vecOf } from '../src/model/map.js';
import { dot } from '../src/model/linalg.js';

// Two clusters of 6 artists; each seed lists its own cluster first, the other cluster weakly.
function table(): LbTable {
  const nodes = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'lb:only-listed-once'];
  const seedLists: Record<string, [number, number][]> = {};
  for (const [c, o] of [['a', 'b'], ['b', 'a']] as const) for (let i = 0; i < 6; i++) {
    const own = [0, 1, 2, 3, 4, 5].filter((j) => j !== i).map((j) => [nodes.indexOf(`${c}${j}`), 100 - j] as [number, number]);
    const other = [0, 1].map((j) => [nodes.indexOf(`${o}${(i + j) % 6}`), 5] as [number, number]);
    seedLists[`${c}${i}`] = [...own, ...other];
  }
  seedLists.a0.push([12, 1]);
  return { nodes, lists: Object.fromEntries(Object.entries(seedLists).map(([k, v]) => [nodes.indexOf(k), v])) };
}

test('ListenBrainz map puts listed neighbours together and drops thin nodes', () => {
  const m = buildLbMap(table(), table().nodes, { dim: 4 });
  assert.equal(m.index.has('lb:only-listed-once'), false);
  const s = (x: string, y: string) => dot(vecOf(m, x)!, vecOf(m, y)!);
  assert.ok(s('a1', 'a2') > s('a1', 'b2'));
  assert.ok(s('b3', 'b4') > s('b3', 'a4'));
  for (let i = 0; i < m.artists.length; i++) assert.ok(Math.abs(dot(m.vec.subarray(i * 4, i * 4 + 4), m.vec.subarray(i * 4, i * 4 + 4)) - 1) < 1e-9);
});

test('only the allowed seeds shape the map', () => {
  const t = table();
  const m = buildLbMap(t, ['a0', 'a1', 'a2'], { dim: 2, hops: 1 });
  assert.equal(m.index.has('b5'), false); // b5 is listed by no allowed seed
  assert.ok(m.listed[m.index.get('a3')!] === 3);
  // one hop out: a3's own list comes in, and through it the rest of cluster a
  const m2 = buildLbMap(t, ['a0'], { dim: 2, hops: 2 });
  assert.ok(m2.index.has('a5') && m2.listed[m2.index.get('a5')!] >= 4);
});

test('blend keeps artists from both maps with unit rows', () => {
  const a = buildLbMap(table(), ['a0', 'a1', 'a2', 'a3'], { dim: 2, hops: 1 });
  const b = buildLbMap(table(), ['b0', 'b1', 'b2', 'b3'], { dim: 2, hops: 1 });
  const m = blendMaps(a, b, 0.5);
  assert.equal(m.artists.length, new Set([...a.artists, ...b.artists]).size);
  for (let i = 0; i < m.artists.length; i++) {
    const v = m.vec.subarray(i * m.dim, (i + 1) * m.dim);
    assert.ok(Math.abs(dot(v, v) - 1) < 1e-9);
  }
});

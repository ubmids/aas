import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Play } from '../src/parse/spotify.js';
import { favourites, candidates, recallAt, RANK2_FEATURES, ListIndex, FAV_HALF_LIFE } from '../src/model/rank2.js';
import { TasteMap } from '../src/model/map.js';

const DAY = 86_400_000;
const play = (t: number, artist: string): Play =>
  ({ t, ms: 200_000, track: `${artist} x`, artist, album: null, uri: null, reasonStart: 'trackdone', reasonEnd: 'trackdone', shuffle: false, skipped: null, platform: null, country: 'GB' });

test('favourites weigh play share by the age of the last play', () => {
  const t = 100 * DAY;
  const plays = [play(t - 30 * DAY, 'A'), play(t - 30 * DAY + 1, 'A'), play(t - 1, 'B'), play(t - 2, 'B'), play(t - 70 * DAY, 'C')];
  const f = favourites(plays.sort((a, b) => a.t - b.t), t);
  assert.deepEqual(f.map((x) => x.artist).sort(), ['A', 'B']); // C is older than 8 weeks
  const a = f.find((x) => x.artist === 'A')!, b = f.find((x) => x.artist === 'B')!;
  assert.ok(Math.abs(a.weight - 0.5 * 0.5 ** ((30 * DAY - 1) / FAV_HALF_LIFE)) < 1e-9);
  assert.ok(b.weight > a.weight);
});

test('candidates: list evidence, known artists excluded, map-only flag', () => {
  const L = new Map([['A', new Map<string, [number, number]>([['X', [1, 1]], ['Y', [2, 0.5]], ['K', [3, 0.4]]])], ['B', new Map<string, [number, number]>([['X', [4, 0.3]]])]]);
  const lists: ListIndex = { get: (a) => L.get(a) };
  // a 2-d map: A and Z close, X far
  const artists = ['A', 'B', 'X', 'Z'], vec = Float64Array.from([1, 0, 0, 1, 0, 1, 0.99, 0.141]);
  const m: TasteMap & { listed: Float64Array } = { artists, index: new Map(artists.map((a, i) => [a, i])), dim: 2, vec, listed: Float64Array.from([0, 0, 2, 0]) };
  const pop = { people: new Map([['X', 3]]), plays: new Map([['X', 40]]) };
  const cs = candidates([{ artist: 'A', weight: 0.6 }, { artist: 'B', weight: 0.4 }], new Set(['A', 'B', 'K']), lists, m, pop, 0.5);
  const by = new Map(cs.map((c) => [c.artist, c.x]));
  assert.ok(!by.has('K'));
  assert.equal(cs[0].x.length, RANK2_FEATURES.length);
  const x = by.get('X')!;
  assert.ok(Math.abs(x[0] - Math.log1p(2)) < 1e-12); // two listers
  assert.ok(Math.abs(x[1] - (0.6 / 1 + 0.4 / 4)) < 1e-12); // reciprocal rank sum
  assert.equal(x[3], 0); // best rank 1
  assert.equal(x[9], 0);
  assert.equal(by.get('Z')![9], 1); // only from the map
  assert.equal(recallAt(['X', 'Z'], new Set(['Z', 'Q']), 1), 0);
  assert.equal(recallAt(['X', 'Z'], new Set(['Z', 'Q']), 2), 0.5);
});

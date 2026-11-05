import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Play } from '../src/parse/spotify.js';
import { listsOf, newArtists, readDoor, matchDropped, doorKind, summarise, story, modeOf, MIN_HISTORY } from '../src/model/doors.js';

const DAY = 86_400_000, T0 = Date.UTC(2024, 0, 1);
const play = (t: number, artist: string, reasonStart: string | null = 'trackdone', ms = 200_000): Play =>
  ({ t, ms, track: `${artist} song`, artist, album: null, uri: null, reasonStart, reasonEnd: null, shuffle: false, skipped: null, platform: null, country: null });

// 13 weeks of an old favourite, then a fall that came on after it, and a tried-and-dropped artist
function history(): Play[] {
  const ps: Play[] = [];
  for (let d = 0; d < 100; d++) for (let k = 0; k < 3; k++) ps.push(play(T0 + d * DAY + k * 210_000, 'Old Fav'));
  const t = T0 + 100 * DAY;
  ps.push(play(t, 'Old Fav'), play(t + 200_000, 'New Love', 'trackdone'));
  for (let d = 1; d <= 5; d++) for (let k = 0; k < 3; k++) ps.push(play(t + d * DAY + k * 210_000, 'New Love', 'clickrow'));
  ps.push(play(t + 2 * DAY + 3600_000, 'Tried Once', 'clickrow'));
  for (let d = 10; d < 70; d++) ps.push(play(t + d * DAY, 'Old Fav'));
  return ps.sort((a, b) => a.t - b.t);
}
const table = { nodes: ['Old Fav', 'New Love', 'Tried Once', 'lb:other'], lists: { 0: [[3, 50], [1, 40]] as [number, number][] } };

test('falls and tried-and-dropped artists are found after the minimum history', () => {
  const ps = history(), found = newArtists(ps);
  const byName = new Map(found.map((f) => [ps[f.i].artist, f]));
  assert.equal(byName.get('New Love')?.kind, 'fall');
  assert.equal(byName.get('Tried Once')?.kind, 'dropped');
  assert.ok(!byName.has('Old Fav')); // first played before MIN_HISTORY
  assert.ok(byName.get('New Love')!.daysTo10! > 2 && byName.get('New Love')!.daysTo10! < 5);
  assert.ok(MIN_HISTORY > 0);
});

test('the door reads how it started, the artist before, and its list rank', () => {
  const ps = history(), L = listsOf(table), f = newArtists(ps).find((x) => ps[x.i].artist === 'New Love')!;
  const d = readDoor(ps, f.i, 'fall', L, f.daysTo10);
  assert.equal(d.mode, 'passive');
  assert.equal(d.prev, 'Old Fav');
  assert.equal(d.prevRank, 2);
  assert.equal(d.fav, 'Old Fav');
  assert.equal(doorKind(d), 'came on after a similar artist');
  assert.match(story(d), /came on after Old Fav/);
  assert.match(story(d), /#2/);
});

test('reason_start maps to picked, passive, other, unknown', () => {
  assert.equal(modeOf('clickrow'), 'picked');
  assert.equal(modeOf('fwdbtn'), 'passive');
  assert.equal(modeOf('remote'), 'other');
  assert.equal(modeOf(null), 'unknown');
});

test('controls are the nearest dropped artists in time, within 60 days', () => {
  const mk = (first: number, kind: 'fall' | 'dropped') => ({ artist: String(first), kind, first, mode: 'picked' as const, reason: 'clickrow', shuffle: null, firstWasSkip: false, prev: null, prevIsFav: false, prevRank: null, fav: null, favRank: null, favShare: 0, resolved: false, daysTo10: kind === 'fall' ? 3 : null });
  const ds = [mk(0, 'dropped'), mk(10 * DAY, 'dropped'), mk(11 * DAY, 'fall'), mk(13 * DAY, 'dropped'), mk(200 * DAY, 'fall')];
  const m = matchDropped(ds, 2);
  assert.deepEqual(m.get(ds[2])!.map((d) => d.first), [10 * DAY, 13 * DAY]);
  assert.equal(m.has(ds[4]), false);
});

test('the summary only claims a door when it beats the person\'s own dropped artists', () => {
  const mk = (mode: 'picked' | 'passive') => ({ artist: 'x', kind: 'fall' as const, first: 0, mode, reason: null, shuffle: null, firstWasSkip: false, prev: null, prevIsFav: false, prevRank: null, fav: null, favRank: null, favShare: 0, resolved: true, daysTo10: 3 });
  const falls = [...Array(8).fill(0).map(() => mk('picked')), mk('passive'), mk('passive')];
  const droppedLow = Array(30).fill(0).map((_, i) => mk(i < 3 ? 'picked' : 'passive'));
  const droppedHigh = Array(30).fill(0).map(() => mk('picked'));
  assert.match(summarise(falls, new Set(), droppedLow).line, /this is how new music sticks for you/);
  assert.doesNotMatch(summarise(falls, new Set(), droppedHigh).line, /sticks for you/);
  assert.match(summarise(falls, new Set(), droppedHigh).line, /many ways/);
});

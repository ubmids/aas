import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Play } from '../src/parse/spotify.js';
import { measureTraits, percentile, traitLine, fmtHour } from '../src/model/traits.js';
import { localClock, zoneOffset } from '../src/model/session.js';

const DAY = 86_400_000;
const play = (t: number, artist: string, ms = 200_000, reasonEnd = 'trackdone', country = 'GB'): Play =>
  ({ t, ms, track: `${artist} ${t % 7}`, artist, album: null, uri: null, reasonStart: 'trackdone', reasonEnd, shuffle: false, skipped: null, platform: null, country });

test('London local hours follow daylight saving time', () => {
  const winter = Date.UTC(2024, 0, 15, 20), summer = Date.UTC(2024, 6, 15, 20);
  assert.equal(zoneOffset('Europe/London', winter), 0);
  assert.equal(zoneOffset('Europe/London', summer), 60);
  const clk = localClock([{ country: 'GB' }]);
  assert.equal(clk.hour({ t: winter, country: 'GB' }), 20);
  assert.equal(clk.hour({ t: summer, country: 'GB' }), 21);
  assert.equal(clk.hour({ t: summer, country: 'ZZ' }), 21); // unknown country falls back to home
  assert.equal(zoneOffset('Asia/Kolkata', summer), 330);
});

test('traits read from a synthetic year of listening', () => {
  // one loyal favourite every evening, one new artist a week, a few quick skips
  const ps: Play[] = [];
  const t0 = Date.UTC(2023, 0, 2, 19);
  for (let d = 0; d < 365; d++) {
    for (let k = 0; k < 4; k++) ps.push(play(t0 + d * DAY + k * 210_000, `Fav ${(d * 4 + k) % 25}`));
    ps.push(play(t0 + d * DAY + 900_000, `New ${Math.floor(d / 7)}`));
    if (d % 3 === 0) ps.push(play(t0 + d * DAY + 1_200_000, 'Skipped', 5_000, 'fwdbtn'));
  }
  ps.sort((a, b) => a.t - b.t);
  const t = new Map(measureTraits(ps, null).map((x) => [x.key, x]));
  assert.ok(t.get('newShare')!.value > 0.1 && t.get('newShare')!.value < 0.3);
  assert.ok(t.get('oldFriends')!.value > 0.6);
  assert.equal(t.get('skipSeconds')!.value, 5);
  const h = t.get('clockHour')!.value;
  assert.ok(h > 19 && h < 21.5, `clock hour ${h}`); // 19:00 UTC is 19:00 or 20:00 in London
  assert.equal(t.get('nightShare')!.value, 0);
  assert.equal(traitLine(t.get('nightShare')!, 0.5), null); // dropped trait never reaches a card
  assert.match(traitLine(t.get('clockHour')!, 0.75)!, /later than 75%/);
});

test('percentiles and hour formatting', () => {
  assert.equal(percentile(5, [1, 2, 3, 10]), 0.75);
  assert.equal(percentile(2, [2, 2]), 0.5);
  assert.equal(fmtHour(25.5), '01:30');
});

test('clock shape traits do not depend on the time zone', async () => {
  const { clockShape } = await import('../src/model/traits.js');
  const plays: Play[] = [], mk = (t: number) => play(t, 'a');
  // every day: 6 plays around 21:00 UTC, some days a few around 08:00
  for (let d = 0; d < 120; d++) {
    for (let k = 0; k < 6; k++) plays.push(mk(Date.UTC(2025, 0, 1) + d * DAY + (21 * 60 + 10 * k) * 60_000));
    if (d % 3 === 0) for (let k = 0; k < 3; k++) plays.push(mk(Date.UTC(2025, 0, 1) + d * DAY + (8 * 60 + 10 * k) * 60_000));
  }
  const utc = clockShape(plays, () => 0), shifted = clockShape(plays, () => 9 * 60 + 30);
  for (let i = 0; i < 2; i++) assert.ok(Math.abs(utc[i].value - shifted[i].value) < 1e-9, utc[i].key);
  assert.ok(utc[0].value > 0 && utc[0].value < 0.5); // mostly one time of day
  assert.ok(utc[1].value > 0.8); // and much the same every day
});

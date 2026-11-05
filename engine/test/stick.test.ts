import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Play } from '../src/parse/spotify.js';
import { newArtistRows, stickFeatures, horizonEnd, baseRateLogit, averagePrecision, STICK_FEATURES_SESSION, STICK_FEATURES_WEEK, StickContext, NewArtist } from '../src/model/stick.js';
import { fitGbt, predictGbt } from '../src/model/gbt.js';
import { auc } from '../src/model/logreg.js';
import { rng } from '../src/model/linalg.js';

const DAY = 86_400_000, MIN = 60_000;
const play = (t: number, artist: string, o: Partial<Play> = {}): Play =>
  ({ t, ms: 200_000, track: `${artist} a`, artist, album: null, uri: null, reasonStart: 'trackdone', reasonEnd: 'trackdone', shuffle: false, skipped: null, platform: null, country: 'GB', ...o });

// 100 days of an old favourite, then X (sticks: 10 proper plays on 3 days) and Y (one skip), then filler to day 200
function history(): Play[] {
  const ps: Play[] = [];
  for (let d = 0; d < 100; d++) ps.push(play(d * DAY, 'Old'));
  const t0 = 100 * DAY + 12 * 3600_000;
  ps.push(play(t0 - 5 * MIN, 'Old'), play(t0, 'X', { reasonStart: 'clickrow' }), play(t0 + 4 * MIN, 'X', { track: 'X b' }), play(t0 + 8 * MIN, 'Y', { ms: 5_000 }));
  for (let d = 1; d <= 3; d++) for (let k = 0; k < 3; k++) ps.push(play(t0 + d * DAY + k * 4 * MIN, 'X'));
  for (let d = 104; d < 200; d++) ps.push(play(d * DAY, 'Old'));
  return ps.sort((a, b) => a.t - b.t);
}
const ctx: StickContext = { lists: null, map: null, people: new Map([['X', 3]]), hour: (p) => new Date(p.t).getUTCHours() };

test('new artists are labelled by the new-love definition', () => {
  const rows = newArtistRows(history());
  const x = rows.find((r) => r.artist === 'X')!, y = rows.find((r) => r.artist === 'Y')!;
  assert.equal(x.label, 1);
  assert.equal(x.fallenByWeek, true);
  assert.ok(x.daysTo10! > 2 && x.daysTo10! < 4);
  assert.equal(y.label, 0);
  assert.ok(!rows.some((r) => r.artist === 'Old')); // first heard inside the warm-up
});

test('features read nothing after the horizon', () => {
  const ps = history(), i = ps.findIndex((p) => p.artist === 'X');
  for (const h of ['session', 'week'] as const) {
    const full = stickFeatures(ps, i, h, ctx, 0);
    const cut = ps.slice(0, horizonEnd(ps, i, h) + 1);
    assert.deepEqual(stickFeatures(cut, i, h, ctx, 0), full);
    // changing anything after the horizon changes nothing
    const later = [...cut, ...ps.slice(cut.length).map((p) => ({ ...p, artist: 'X', reasonStart: 'clickrow' }))];
    assert.deepEqual(stickFeatures(later, i, h, ctx, 0), full);
    assert.equal(full.length, (h === 'session' ? STICK_FEATURES_SESSION : STICK_FEATURES_WEEK).length);
  }
  const f = stickFeatures(ps, i, 'session', ctx, 0), at = (n: string) => f[STICK_FEATURES_SESSION.indexOf(n as never)];
  assert.equal(at('picked'), 1);
  assert.equal(at('sessPlays'), Math.log1p(2));
  assert.equal(at('sessTracks'), Math.log1p(2));
  assert.equal(at('logPeople'), Math.log1p(3));
  const w = stickFeatures(ps, i, 'week', ctx, 0);
  assert.equal(w[STICK_FEATURES_WEEK.indexOf('wkProper')], Math.log1p(11));
  assert.equal(w[STICK_FEATURES_WEEK.indexOf('wkDays')], 4);
  assert.equal(w[STICK_FEATURES_WEEK.indexOf('wkReturned')], 1);
});

test('own base rate counts only outcomes known in time, shrunk to the prior', () => {
  const r = (t0: number, label: 0 | 1): NewArtist => ({ artist: String(t0), i: 0, t0, label, fallenByWeek: false, daysTo10: null });
  const rows = [r(0, 1), r(DAY, 1), r(100 * DAY, 1)];
  const lg = (p: number) => Math.log(p / (1 - p));
  assert.ok(Math.abs(baseRateLogit(rows, 60 * DAY, 0.1, 20) - lg((2 + 2) / 22)) < 1e-12);
  assert.ok(Math.abs(baseRateLogit(rows, 10 * DAY, 0.1, 20) - lg(0.1)) < 1e-12);
});

test('average precision', () => {
  assert.ok(Math.abs(averagePrecision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) - (1 + 2 / 3) / 2) < 1e-12);
  assert.equal(averagePrecision([0.5, 0.5], [0, 1]), 0.5); // ties broken against the model
});

test('boosted trees learn an interaction a linear model cannot', () => {
  const r = rng(3), X: number[][] = [], y: number[] = [];
  for (let k = 0; k < 2000; k++) { const a = r(), b = r(); X.push([a, b]); y.push((a > 0.5) !== (b > 0.5) ? 1 : 0); }
  const m = fitGbt(X, y, { trees: 60 });
  assert.ok(auc(X.map((x) => predictGbt(m, x)), y) > 0.95);
});

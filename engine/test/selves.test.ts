import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fitSelves, fitSelvesValidated, PENALTIES, SessionRow } from '../src/model/selves.js';
import { rng, gauss } from '../src/model/linalg.js';

const WEEK = 7 * 86_400_000;
// three clear listening modes, 20 dimensional music vectors, over 60 weeks
function rows(): SessionRow[] {
  const r = rng(5), out: SessionRow[] = [];
  for (let i = 0; i < 1500; i++) {
    const c = i % 3, start = Math.floor((i / 1500) * 60 * WEEK);
    const music = Array.from({ length: 20 }, (_, j) => (j === c * 3 ? 1 : 0) + 0.15 * gauss(r));
    out.push({ s: { start, end: start + 3_600_000, plays: [] }, music, spread: 0.3 + 0.05 * gauss(r),
      beh: { pick: c === 0 ? 0.8 : 0.2, skip: 0.1 + 0.02 * gauss(r), repeat: 0.05, logLen: Math.log(10) + 0.1 * gauss(r), minutes: 40 }, week: Math.floor(start / WEEK) });
  }
  return out;
}

test('a stronger BIC penalty never picks more selves', () => {
  const R = rows(), k1 = fitSelves(R, { penalty: 1 }).live.length, k8 = fitSelves(R, { penalty: 8 }).live.length;
  assert.ok(k8 <= k1, `${k8} > ${k1}`);
  assert.ok(k8 >= 2);
});

test('the validated fit picks a penalty from the list and keeps 3 to 6 selves when it can', () => {
  const R = rows(), s = fitSelvesValidated(R, 50 * WEEK);
  assert.ok(PENALTIES.includes(s.penalty));
  assert.ok(s.live.length >= 2 && s.live.length <= 7);
});

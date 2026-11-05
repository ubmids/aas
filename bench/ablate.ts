// Drop feature groups and re-score skip AUC, to see what the model leans on (and catch leaks).
import { readFileSync, readdirSync } from 'node:fs';
import { parseSpotify, isSkip } from '../engine/src/parse/spotify.js';
import { skipFeatures } from '../engine/src/model/skip.js';
import { fitLogReg, predict, auc } from '../engine/src/model/logreg.js';
const groups: Record<string, number[]> = {
  all: [], noHistory: [0,1,2,3,4,5], noSession: [6,7,8,9,10], noStart: [14,15,16,17,18,19,20], noClockShuffle: [11,12,13],
  onlyStart: [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
};
const res: Record<string, number[]> = {};
for (const f of readdirSync('bench/data/spotify_full')) {
  const plays = parseSpotify(readFileSync('bench/data/spotify_full/' + f, 'utf8').trim().split('\n').map((l) => JSON.parse(l)));
  if (plays.length < 2000) continue;
  const cut = plays[plays.length - 1].t - 8 * 7 * 864e5, n = plays.findIndex((p) => p.t >= cut);
  if (n < 1000 || plays.length - n < 200) continue;
  const X = skipFeatures(plays), y = plays.map((p) => (isSkip(p) ? 1 : 0)), lo = Math.max(0, n - 60000);
  for (const [g, drop] of Object.entries(groups)) {
    const keep = (x: number[]) => x.filter((_, j) => !drop.includes(j));
    const m = fitLogReg(X.slice(lo, n).map(keep), y.slice(lo, n), 1);
    (res[g] ??= []).push(auc(X.slice(n).map((x) => predict(m, keep(x))), y.slice(n)));
  }
}
const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1];
for (const [g, v] of Object.entries(res)) console.log(g.padEnd(16), med(v).toFixed(3), v.map((x) => x.toFixed(2)).join(' '));

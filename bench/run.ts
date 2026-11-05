// AAS accuracy bench. Each person is split by time: train on everything before the last
// HOLDOUT_WEEKS, score on those weeks. Anything tuned is tuned on a validation slice carved from
// the end of the training period, never on the held-out weeks.
// Prints per-person scores and the median per task. `--describe` also prints selves and eras.
import { Play, isSkip, trackKey } from '../engine/src/parse/spotify.js';
import { skipFeatures, trainSkip, scoreSkip, baselineSkip } from '../engine/src/model/skip.js';
import { auc, fitLogReg, predict as lrPredict } from '../engine/src/model/logreg.js';
import { artistScores, topK, ndcg, tuneHalfLife, listenCounts, clock, hourOfWeek, localHourOfWeek, noveltyFeatures, fitNovelty, predict } from '../engine/src/model/taste.js';
import { loadPeople, HOLDOUT_WEEKS, DAY } from './load.js';
import { buildMap, TasteMap, vecOf } from '../engine/src/model/map.js';
import { knownBefore, findFalls, tasteState, jumpFrom, fallFeatures, recentCounts, Jump, FALL_WEEKS, FALL_FEATURES } from '../engine/src/model/falls.js';
import { sessionRows, fitSelvesValidated, contextEncoder, describeSelves } from '../engine/src/model/selves.js';
import { fitSoftmax, probs } from '../engine/src/model/softmax.js';
import { weekly, changePoints, eraScores, describeEras } from '../engine/src/model/eras.js';
import { dot, rng } from '../engine/src/model/linalg.js';
import { localClock } from '../engine/src/model/session.js';
import { buildLbMap, blendMaps, listEvidence, LbTable } from '../engine/src/model/lbmap.js';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const DESCRIBE = process.argv.includes('--describe');
const WEEK = 7 * DAY, MAX_TRAIN = 60_000;
const median = (a: number[]) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '  -  ');
const hash = (s: string) => [...s].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 7);
const log = (s: string) => process.stderr.write(s + '\n');

const people = loadPeople();
// ListenBrainz table from bench/scripts/lb_build.ts; without it the bench runs v0 only
const LB_FILE = 'bench/data/lb/table.json', BLEND_W = 0.5;
const LB: LbTable | null = existsSync(LB_FILE) && !process.argv.includes('--no-lb') ? JSON.parse(readFileSync(LB_FILE, 'utf8')) : null;
const lbNodeIx = LB ? new Map(LB.nodes.map((n, i) => [n, i])) : null;
/** Seeds: artists with 3+ proper plays in these people's training weeks. */
function seedsOf(ps: typeof people) {
  const out = new Set<string>();
  for (const p of ps) { const c = new Map<string, number>(); for (const x of p.plays.slice(0, p.n)) if (!isSkip(x)) c.set(x.artist, (c.get(x.artist) ?? 0) + 1); for (const [a, v] of c) if (v >= 3) out.add(a); }
  return out;
}
const trainOf = (p: (typeof people)[number]) => p.plays.slice(0, p.n);

// ---- taste maps: one from everyone's training weeks (selves, eras, game), and one per person
// built from everyone else's training weeks only (falls: the person's own co-listening never
// shapes the map they are scored on, so past and future falls are measured the same way).
let t0 = Date.now();
const globalMap = buildMap(people.map(trainOf));
const lopo = new Map(people.map((p) => [p.name, buildMap(people.filter((q) => q !== p).map(trainOf))] as const));
const gameMaps: [string, TasteMap][] = [['gameMap', globalMap]];
if (LB) { const g = buildLbMap(LB, seedsOf(people)); gameMaps.push(['gameLb', g], ['gameBl', blendMaps(globalMap, g, BLEND_W)]); log(`ListenBrainz global map: ${g.artists.length} artists`); }
log(`maps: global ${globalMap.artists.length} artists, ${people.length} leave-one-out maps, ${((Date.now() - t0) / 1000).toFixed(1)}s`);

// popularity of each artist among other people's training weeks
const popPlays = new Map(people.map((p) => [p.name, listenCounts(trainOf(p))] as const));
function othersPop(name: string) {
  const people_ = new Map<string, number>(), plays = new Map<string, number>();
  for (const [n, c] of popPlays) if (n !== name) for (const [a, v] of c) { people_.set(a, (people_.get(a) ?? 0) + 1); plays.set(a, (plays.get(a) ?? 0) + v); }
  return { people: people_, plays };
}

// ---- 6. falls: pseudo windows inside each person's training period train the ranker. Done once
// per map variant (v0 co-listening, ListenBrainz, the two blended), each leave-one-person-out.
interface FallWindow { start: number; known: Set<string>; falls: string[]; sims: number[] }
function fallSetup(maps: Map<string, TasteMap>, evidence = false) {
  const near = (src: Play[], t: number) => (evidence && LB ? listEvidence(LB, recentCounts(src, t), lbNodeIx!) : undefined);
  const data = new Map<string, { windows: FallWindow[]; pop: ReturnType<typeof othersPop> }>();
  const allSims: number[] = [];
  for (const p of people) {
    const m = maps.get(p.name)!, train = trainOf(p), first = train[0].t, windows: FallWindow[] = [];
    for (let k = 1; k <= 12; k++) {
      const start = p.cut - k * FALL_WEEKS * WEEK;
      if (start - first < 26 * WEEK) break;
      const known = knownBefore(train, start), falls = findFalls(train, start, known), st = tasteState(m, train, start);
      const sims = falls.map((a) => vecOf(m, a)).filter((v): v is Float64Array => !!v && !!st.cRecent).map((v) => dot(v, st.cRecent!));
      windows.push({ start, known, falls, sims });
      allSims.push(...sims);
    }
    data.set(p.name, { windows, pop: othersPop(p.name) });
  }
  const prior: Jump = jumpFrom(allSims, { mu: 0, sd: 0.3, n: 0 }, 0);
  // training rows per person; each person is scored by a ranker fitted on everyone ELSE's windows
  const rowsOf = new Map<string, { X: number[][]; y: number[] }>();
  for (const p of people) {
    const m = maps.get(p.name)!, train = trainOf(p), { windows, pop } = data.get(p.name)!, r = rng(hash(p.name));
    const X: number[][] = [], y: number[] = [];
    windows.forEach((w, i) => {
      const jump = jumpFrom(windows.slice(i + 1).flatMap((x) => x.sims), prior);
      const feat = fallFeatures(m, tasteState(m, train, w.start), jump, pop, near(train, w.start)), fallSet = new Set(w.falls);
      for (const a of w.falls) { const ix = m.index.get(a); if (ix !== undefined) { X.push(feat(ix)); y.push(1); } }
      for (let j = 0; j < 100; j++) {
        const ix = Math.floor(r() * m.artists.length), a = m.artists[ix];
        if (w.known.has(a) || fallSet.has(a)) continue;
        X.push(feat(ix)); y.push(0);
      }
    });
    rowsOf.set(p.name, { X, y });
  }
  const ranker = new Map(people.map((p) => {
    const others = people.filter((q) => q !== p).map((q) => rowsOf.get(q.name)!);
    return [p.name, fitLogReg(others.flatMap((o) => o.X), others.flatMap((o) => o.y), 1)] as const;
  }));
  const all = [...rowsOf.values()], w = fitLogReg(all.flatMap((o) => o.X), all.flatMap((o) => o.y), 1).w;
  const info = `${allSims.length} past falls on the map, prior jump sim ${prior.mu.toFixed(2)} +- ${prior.sd.toFixed(2)}; ranker weights (all windows, reading only) [bias ${FALL_FEATURES.join(' ')}] ${w.map((x) => x.toFixed(2)).join(' ')}`;
  return { maps, data, prior, ranker, info, near };
}
type FallVariant = ReturnType<typeof fallSetup> & { key: string; label: string; pre: string };
const fallVariants: FallVariant[] = [{ key: 'v0', label: 'v0 co-listening map', pre: '', ...fallSetup(lopo) }];
if (LB) {
  t0 = Date.now();
  const lbLopo = new Map(people.map((p) => [p.name, buildLbMap(LB, seedsOf(people.filter((q) => q !== p)))] as const));
  const blLopo = new Map(people.map((p) => [p.name, blendMaps(lopo.get(p.name)!, lbLopo.get(p.name)!, BLEND_W)] as const));
  log(`ListenBrainz maps: median ${median([...lbLopo.values()].map((m) => m.artists.length))} artists per leave-one-out map, ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  // both get the direct list evidence from the person's recent artists as a feature (and the "lists" baseline)
  fallVariants.push({ key: 'lb', label: 'ListenBrainz map', pre: 'lb_', ...fallSetup(lbLopo, true) });
  fallVariants.push({ key: 'bl', label: `blend (v0 weight ${BLEND_W})`, pre: 'bl_', ...fallSetup(blLopo, true) });
}
for (const v of fallVariants) log(`falls ${v.key}: ${v.info}`);
const FALL_METHODS = ['model', 'popular', 'centroid', 'favourite', 'lists'] as const;
const fallPools = new Map(fallVariants.map((v) => [v.key, { heldOut: { n: 0, onMap: 0 } as Record<string, number>, rolling: { n: 0, onMap: 0 } as Record<string, number> }]));

// ---- 3. the game: track popularity across everyone, for matching decoys
const trackUsers = new Map<string, Set<string>>(), trackInfo = new Map<string, Play>();
for (const pp of people) for (const p of pp.plays) if (!isSkip(p)) {
  const k = trackKey(p); (trackUsers.get(k) ?? trackUsers.set(k, new Set()).get(k)!).add(pp.name); trackInfo.set(k, p);
}
/** One track the person really played among 4 decoys of matched popularity; same seed = same rounds. */
function game(name: string, mine: Set<string>, heldOut: Play[], heardArtists: Set<string>, score: (artist: string) => number, seed: number) {
  const r = rng(seed), buckets = new Map<number, string[]>();
  for (const [k, u] of trackUsers) if (!mine.has(k) && !u.has(name)) (buckets.get(u.size) ?? buckets.set(u.size, []).get(u.size)!).push(k);
  const heard = [...new Set(heldOut.filter((p) => !isSkip(p)).map(trackKey))].sort();
  for (let i = heard.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [heard[i], heard[j]] = [heard[j], heard[i]]; }
  let wins = 0, rounds = 0, winsNew = 0, roundsNew = 0;
  for (const k of heard.slice(0, 300)) {
    const same = buckets.get(Math.max(1, (trackUsers.get(k)?.size ?? 1) - 1)) ?? [];
    if (same.length < 4) continue;
    const cands = [k, ...Array.from({ length: 4 }, () => same[Math.floor(r() * same.length)])];
    const s = cands.map((c) => score(trackInfo.get(c)!.artist) + r() * 1e-9);
    const win = s[0] >= Math.max(...s), isNew = !heardArtists.has(trackInfo.get(k)!.artist);
    rounds++; if (win) wins++;
    if (isNew) { roundsNew++; if (win) winsNew++; }
  }
  return { acc: rounds ? wins / rounds : NaN, accNew: roundsNew >= 20 ? winsNew / roundsNew : NaN };
}

const rows: Record<string, number>[] = [];
const names: string[] = [], skipCal: { p: number[]; b: number[]; y: number[] }[] = [];
const cols: string[] = [];
for (const P of people) {
  const { name, plays, n, cut, episodes } = P;
  const train = plays.slice(0, n), test = plays.slice(n), now = train[train.length - 1].t;
  const r: Record<string, number> = {};
  const say: string[] = [];

  // 1. finish or skip
  const X = skipFeatures(plays, localClock(train).hour), lo = Math.max(0, n - MAX_TRAIN);
  const m = trainSkip(X.slice(lo, n), plays.slice(lo, n));
  const yT = test.map((p) => (isSkip(p) ? 1 : 0));
  const pT = X.slice(n).map((x) => scoreSkip(m, x)), bT = X.slice(n).map(baselineSkip);
  r.skip = auc(pT, yT);
  r.skip_b = auc(bT, yT);
  skipCal.push({ p: pT.map((v) => +v.toFixed(4)), b: bT.map((v) => +v.toFixed(4)), y: yT });

  // 2. next 8 weeks' artists
  const truth = listenCounts(test), hl = tuneHalfLife(train);
  r.hl = hl;
  r.artists = ndcg(topK(artistScores(train, now, hl), 10), truth);
  r.artists_allTime = ndcg(topK(artistScores(train, now, 1e6), 10), truth);
  r.artists_last4w = ndcg(topK(listenCounts(train.filter((p) => p.t > now - 28 * DAY)), 10), truth);

  // 3. the game, old scorer (decayed artist plays) and map scorers (plus nearness to recent favourites),
  // one per map variant; each map's weight is tuned on a validation game inside the training period
  const mapGameScore = (gm: TasteMap, tr: Play[], at: number, h: number, w: number) => {
    const aS = artistScores(tr, at, h), favs = tasteState(gm, tr, at).favs;
    const near = new Map<string, number>();
    const nearOf = (v: Float64Array) => (favs.length ? Math.max(...favs.map((f) => dot(f, v))) : 0);
    let neutral = 0; // artists not on the map get the person's average nearness, not zero
    for (let i = 0; i < gm.artists.length; i += 7) neutral += nearOf(gm.vec.subarray(i * gm.dim, (i + 1) * gm.dim)) / Math.ceil(gm.artists.length / 7);
    return (a: string) => {
      let s = near.get(a);
      if (s === undefined) { const v = vecOf(gm, a); s = v ? nearOf(v) : neutral; near.set(a, s); }
      return Math.log1p(aS.get(a) ?? 0) + w * s;
    };
  };
  {
    const mine = new Set(plays.map(trackKey)), aS = artistScores(train, now, hl), heardA = new Set(aS.keys()), seed = hash(name) + 2;
    const run = (sc: (a: string) => number) => game(name, mine, test, heardA, sc, seed);
    const old = run((a) => aS.get(a) ?? 0);
    r.game = old.acc; r.gameNew = old.accNew;
    type Scorer = (tr: Play[], at: number, w: number) => (a: string) => number;
    const scorers: [string, Scorer][] = gameMaps.map(([key, gm]) => [key, (tr, at, w) => mapGameScore(gm, tr, at, hl, w)] as [string, Scorer]);
    if (LB) scorers.push(['gameEv', (tr, at, w) => {
      const aS = artistScores(tr, at, hl), ev = listEvidence(LB, recentCounts(tr, at), lbNodeIx!);
      return (a: string) => Math.log1p(aS.get(a) ?? 0) + w * Math.log1p(10 * (ev.get(a) ?? 0));
    }]);
    for (const [key, sc] of scorers) {
      const vcut = cut - HOLDOUT_WEEKS * WEEK, fit = train.filter((p) => p.t < vcut), val = train.filter((p) => p.t >= vcut);
      let bestW = 0, bestAcc = -1;
      if (fit.length > 500 && val.length > 100) {
        const fNow = fit[fit.length - 1].t, fHeard = new Set(artistScores(fit, fNow, hl).keys());
        for (const w of [0, 0.5, 1, 2, 4, 8]) {
          const { acc } = game(name, new Set(train.map(trackKey)), val, fHeard, sc(fit, fNow, w), hash(name) + 1);
          if (acc > bestAcc + 1e-9) { bestAcc = acc; bestW = w; }
        }
      }
      const withMap = run(sc(train, now, bestW)), mapOnly = run(sc(train, now, 1e6));
      r[`${key}W`] = bestW; r[key] = withMap.acc; r[`${key}New`] = withMap.accNew; r[`${key}OnlyNew`] = mapOnly.accNew;
      const heldNew = [...new Set(test.filter((p) => !isSkip(p) && !heardA.has(p.artist)).map((p) => p.artist))];
      const gm = gameMaps.find(([k]) => k === key)?.[1];
      if (gm) r[`${key}NewOnMap`] = heldNew.length ? heldNew.filter((a) => gm.index.has(a)).length / heldNew.length : NaN;
    }
  }

  // 4. when you listen: bits per play saved vs a uniform clock, on local hours (DST-correct, from the
  // train weeks' countries) and on UTC hours as the old reading
  const clk = localClock(train), loc = localHourOfWeek(clk), utc = (p: Play) => hourOfWeek(p.t);
  for (const [key, how] of [['clock', loc], ['clockUtc', utc]] as const) {
    const c2 = now - HOLDOUT_WEEKS * WEEK, fit = train.filter((p) => p.t < c2), val = train.filter((p) => p.t >= c2);
    const sc = (x: number) => { const cl = clock(fit, c2, x, 1, how); return val.reduce((s, p) => s + Math.log(cl[how(p)]), 0); };
    const chl = fit.length < 500 || val.length < 100 ? 1e6 : [30, 90, 365, 1e6].reduce((b, h) => (sc(h) > sc(b) ? h : b), 1e6);
    const c = clock(train, now, chl, 1, how), cAll = clock(train, now, 1e6, 1, how);
    const bits = (cl: number[]) => test.reduce((s, p) => s + Math.log2(cl[how(p)] * 168), 0) / test.length;
    r[key] = bits(c); r[`${key}_allTime`] = bits(cAll);
  }

  // 5. new artist or familiar
  const nv = noveltyFeatures(plays), nm = fitNovelty(nv.X.slice(lo, n), nv.y.slice(lo, n));
  r.novel = auc(nv.X.slice(n).map((x) => predict(nm, x)), nv.y.slice(n));

  // 6. who you'll fall for next: the held-out window, plus rolling windows inside training
  // (map, popularity and ranker never saw this person; jump stats only use earlier windows)
  for (const V of fallVariants) {
    const lm: TasteMap = V.maps.get(name)!, { windows, pop } = V.data.get(name)!, ranker = V.ranker.get(name)!, priorJump = V.prior, pre = V.pre;
    const evalWindow = (start: number, known: Set<string>, falls: string[], jump: Jump, src: Play[]) => {
      const feat = fallFeatures(lm, tasteState(lm, src, start), jump, pop, V.near(src, start));
      const cands = lm.artists.map((_, i) => i).filter((i) => !known.has(lm.artists[i]));
      const F = new Map(cands.map((i) => [i, feat(i)]));
      const score: Record<(typeof FALL_METHODS)[number], (i: number) => number> = {
        model: (i) => lrPredict(ranker, F.get(i)!),
        popular: (i) => F.get(i)![5] * 1e3 + F.get(i)![6],
        centroid: (i) => F.get(i)![1],
        favourite: (i) => F.get(i)![3],
        lists: (i) => F.get(i)![8] * 1e3 + F.get(i)![5], // ListenBrainz's own "similar to your recent artists", ties by popularity
      };
      const fallSet = new Set(falls), hits: Record<string, number> = {};
      for (const meth of FALL_METHODS) {
        const s = new Map(cands.map((i) => [i, score[meth](i)]));
        const ranked = [...cands].sort((a, b) => s.get(b)! - s.get(a)! || a - b);
        for (const k of [20, 50]) hits[`${meth}${k}`] = ranked.slice(0, k).filter((i) => fallSet.has(lm.artists[i])).length;
      }
      return { n: falls.length, onMap: falls.filter((a) => lm.index.has(a)).length, hits };
    };
    const add = (pool: Record<string, number>, e: ReturnType<typeof evalWindow>) => {
      pool.n += e.n; pool.onMap += e.onMap;
      for (const [k, v] of Object.entries(e.hits)) pool[k] = (pool[k] ?? 0) + v;
    };
    const pools = fallPools.get(V.key)!;
    const known = knownBefore(plays, cut), falls = findFalls(plays, cut, known);
    const ho = evalWindow(cut, known, falls, jumpFrom(windows.flatMap((w) => w.sims), priorJump), train);
    add(pools.heldOut, ho);
    r[`${pre}falls`] = ho.n; r[`${pre}fallsOnMap`] = ho.onMap; r[`${pre}fallCeil`] = ho.n ? ho.onMap / ho.n : NaN;
    for (const meth of FALL_METHODS) for (const k of [20, 50]) r[`${pre}fall${k}_${meth}`] = ho.n ? ho.hits[`${meth}${k}`] / ho.n : NaN;
    const roll: Record<string, number> = { n: 0, onMap: 0 };
    windows.forEach((w, i) => { if (w.falls.length) add(roll, evalWindow(w.start, w.known, w.falls, jumpFrom(windows.slice(i + 1).flatMap((x) => x.sims), priorJump), train)); });
    add(pools.rolling, { n: roll.n, onMap: roll.onMap, hits: Object.fromEntries(Object.entries(roll).filter(([k]) => k !== 'n' && k !== 'onMap')) });
    r[`${pre}rollFalls`] = roll.n; r[`${pre}rollOnMap`] = roll.onMap;
    for (const meth of FALL_METHODS) for (const k of [20, 50]) r[`${pre}roll${k}_${meth}`] = roll.n ? (roll[`${meth}${k}`] ?? 0) / roll.n : NaN;
  }

  // 7. selves, and predicting a held-out session's self from its surroundings alone
  const rowsTrain = sessionRows(train, globalMap), rowsTest = sessionRows(test, globalMap);
  const selves = fitSelvesValidated(rowsTrain, cut - HOLDOUT_WEEKS * WEEK); // penalty chosen on the training period's last 8 weeks
  const ctx = contextEncoder(train);
  {
    const live = selves.live, idx = new Map(live.map((c, i) => [c, i]));
    const fitRows = rowsTrain.slice(-8000);
    const ytr = fitRows.map((x) => idx.get(selves.label(x))!), yte = rowsTest.map((x) => idx.get(selves.label(x))!);
    const freq = live.map((_, i) => (ytr.filter((y) => y === i).length + 1) / (ytr.length + live.length));
    const maj = freq.indexOf(Math.max(...freq));
    const sm = fitSoftmax(fitRows.map((x) => ctx.encode(x.s)), ytr, live.length);
    const pr = rowsTest.map((x) => probs(sm, ctx.encode(x.s)));
    r.selves = live.length; r.selfPenalty = selves.penalty; r.selfSessions = rowsTest.length;
    r.self = yte.length ? pr.filter((p, i) => p.indexOf(Math.max(...p)) === yte[i]).length / yte.length : NaN;
    r.self_majority = yte.length ? yte.filter((y) => y === maj).length / yte.length : NaN;
    r.selfNll = yte.length ? -pr.reduce((s, p, i) => s + Math.log(Math.max(1e-9, p[yte[i]])), 0) / yte.length : NaN;
    r.selfNll_prior = yte.length ? -yte.reduce((s, y) => s + Math.log(freq[y]), 0) / yte.length : NaN;
    if (DESCRIBE) say.push(`  selves (BIC penalty x${selves.penalty} chosen on validation, ${live.length} recurring):`, ...describeSelves(rowsTrain, selves, ctx).map((l) => '    ' + l));
  }

  // 8. eras, and an era-aware forecast of the held-out weeks' artists
  {
    const first = train[0].t, vcut = cut - HOLDOUT_WEEKS * WEEK;
    const fit = train.filter((p) => p.t < vcut), val = train.filter((p) => p.t >= vcut), vTruth = listenCounts(val);
    let best = { pen: 1, gamma: 1, h: hl, score: -1 };
    if (fit.length > 500 && val.length > 100 && vcut - first > 30 * WEEK) {
      const wkV = weekly(fit, episodes, first, vcut, globalMap, rowsTrain, selves), vNow = fit[fit.length - 1].t;
      for (const pen of [1, 0.5, 2, 4]) { // the forecast rarely separates these; ties keep the default 1 (about 1 to 3 eras a year)
        const st = changePoints(wkV.X, 10, pen), eraStart = wkV.start[st[st.length - 1]];
        for (const gamma of [0, 0.05, 0.2, 0.5, 1]) for (const h of [hl, 1e6]) {
          const s = ndcg(topK(eraScores(fit, vNow + 1, eraStart, h, gamma), 10), vTruth);
          if (s > best.score + 1e-9) best = { pen, gamma, h, score: s };
        }
      }
    }
    const wk = weekly(train, episodes, first, cut, globalMap, rowsTrain, selves);
    const starts = changePoints(wk.X, 10, best.pen), eraStart = wk.start[starts[starts.length - 1]];
    r.eras = starts.length; r.eraGamma = best.gamma;
    r.eraForecast = ndcg(topK(eraScores(train, now + 1, eraStart, best.h, best.gamma), 10), truth);
    r.eraOnly = ndcg(topK(eraScores(train, now + 1, eraStart, 1e6, 0), 10), truth);
    if (DESCRIBE) say.push(`  eras (min 10 weeks, penalty x${best.pen} chosen on validation):`, ...describeEras(wk, starts, globalMap).map((l) => '    ' + l));
  }

  rows.push(r); names.push(name);
  for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
  console.log(`${name.slice(0, 24).padEnd(24)} skip ${f3(r.skip)}/${f3(r.skip_b)} | artists ${f3(r.artists)} all ${f3(r.artists_allTime)} era ${f3(r.eraForecast)} (${r.eras} eras) | game ${f3(r.game)}->${f3(r.gameMap)} new ${f3(r.gameNew)}->${f3(r.gameMapNew)} | falls ${r.falls} (${r.fallsOnMap} on v0 map, ${r.lb_fallsOnMap ?? '-'} on LB), rolling ${r.rollFalls} r@50 ${f3(r.roll50_model)} pop ${f3(r.roll50_popular)} cen ${f3(r.roll50_centroid)} | selves ${r.selves} acc ${f3(r.self)}/${f3(r.self_majority)}`);
  for (const l of say) console.log(l);
}

function fallsSummary() {
  const line = (label: string, P: Record<string, number>, pre: string) => {
    const pct = (k: string) => (P.n ? (P[k] ?? 0) / P.n : NaN);
    const per = (k: string, b: string) => wins(`${pre}${k}_model`, `${pre}${k}_${b}`);
    return `                     ${label}: ${P.n} falls, ${P.onMap} on the map (ceiling ${f3(P.onMap / P.n)})\n` + [20, 50].map((k) =>
      `                       recall@${k} pooled ${f3(pct(`model${k}`))} vs popular ${f3(pct(`popular${k}`))}, centroid ${f3(pct(`centroid${k}`))}, nearest favourite ${f3(pct(`favourite${k}`))}, lists ${f3(pct(`lists${k}`))} | per-person median ${med(`${pre}${k}_model`)} (wins vs popular ${per(String(k), 'popular')}, vs centroid ${per(String(k), 'centroid')}, vs lists ${per(String(k), 'lists')})`).join('\n');
  };
  return `  6 next falls       median ${med('falls')} held-out falls/person (>=10 proper plays on >=3 days in the 8 weeks, artist never played before)\n` +
    fallVariants.map((v) => `                     [${v.label}]\n${line('held-out 8 weeks', fallPools.get(v.key)!.heldOut, v.pre + 'fall')}\n${line('rolling 8-week windows inside training (person never seen by map or ranker)', fallPools.get(v.key)!.rolling, v.pre + 'roll')}`).join('\n');
}
function gameSummary() {
  return [...gameMaps.map(([k]) => k), ...(LB ? ['gameEv'] : [])].map((k) => `                     ${k}: ${med(k)} (wins vs old ${wins(k, 'game')}, ties ${ties(k, 'game')}); new-artist rounds ${med(`${k}New`)} (wins ${wins(`${k}New`, 'gameNew')}); map only, new-artist rounds ${med(`${k}OnlyNew`)}; held-out new artists on the map ${med(`${k}NewOnMap`)}`).join('\n');
}
const med = (k: string) => f3(median(rows.map((r) => r[k])));
const has = (a: string, b: string) => rows.filter((r) => Number.isFinite(r[a]) && Number.isFinite(r[b]));
const wins = (a: string, b: string, lower = false) => { const h = has(a, b); return `${h.filter((r) => (lower ? r[a] < r[b] : r[a] > r[b])).length}/${h.length}`; };
const ties = (a: string, b: string) => has(a, b).filter((r) => r[a] === r[b]).length;
console.log(`\nMEDIAN over ${rows.length} people (wins = people where the model beats the baseline; ties shown when common)
  1 finish or skip   AUC ${med('skip')}  vs artist rate ${med('skip_b')}  (wins ${wins('skip', 'skip_b')})
  2 next artists     NDCG@10 decay ${med('artists')}  vs all-time ${med('artists_allTime')} (wins ${wins('artists', 'artists_allTime')}), last 4w ${med('artists_last4w')} (wins ${wins('artists', 'artists_last4w')})
  3 the game         ${med('game')} correct (chance 0.200); new-artist rounds ${med('gameNew')}
${gameSummary()}
  4 when you listen  local hours ${med('clock')} bits/play over uniform  vs all-time clock ${med('clock_allTime')} (wins ${wins('clock', 'clock_allTime')}); UTC hours ${med('clockUtc')} (local wins ${wins('clock', 'clockUtc')})
  5 new or familiar  AUC ${med('novel')}
${fallsSummary()}
  7 selves           ${med('selves')} recurring selves/person; held-out session's self from context: acc ${med('self')} vs majority ${med('self_majority')} (wins ${wins('self', 'self_majority')}, ties ${ties('self', 'self_majority')})
                     log-loss ${med('selfNll')} vs class prior ${med('selfNll_prior')} (wins ${wins('selfNll', 'selfNll_prior', true)})
  8 eras             ${med('eras')} eras/person in training; era-aware forecast NDCG@10 ${med('eraForecast')} vs decay ${med('artists')} (wins ${wins('eraForecast', 'artists')}, ties ${ties('eraForecast', 'artists')}), vs all-time ${med('artists_allTime')} (wins ${wins('eraForecast', 'artists_allTime')}); current era only ${med('eraOnly')} (wins vs decay ${wins('eraOnly', 'artists')})`);

// everything the report needs (bench/out is gitignored): `npm run report` reads it
mkdirSync('bench/out', { recursive: true });
writeFileSync('bench/out/bench.json', JSON.stringify({ lb: !!LB, at: new Date().toISOString(), names, rows, skipCal }));
log('wrote bench/out/bench.json');

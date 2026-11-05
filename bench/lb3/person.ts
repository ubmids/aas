// Round 3, step 2: everything that can be read from one person alone, with maps and popularity that
// person never shaped (their fold's map, popularity minus their own plays). One file per person:
//   person/<id>.json      traits (full training, and each half), next artists, clock, eras, doors
//   stick/<id>.json.gz    feature rows for "will this new artist stick" (both horizons), labels
//   rank2/<id>.json.gz    per window: the two-stage candidates' features and labels, popular's hits
// Models that learn across people are fitted in models.ts. Cached per person, so a rerun only does
// the people not yet done. Run several shards at once: npx tsx bench/lb3/person.ts 0/4 (shard 0 of 4).
import { existsSync, writeFileSync } from 'node:fs';
import { ids, loadPerson, foldOf, cachePath, readJsonGz, writeJsonGz, loadMap, lbTable, utcHour, hash, log, Pop, WEEK, DAY, HOLDOUT_WEEKS, LbPerson } from './common.js';
import { Play } from '../../engine/src/parse/spotify.js';
import { TasteMap } from '../../engine/src/model/map.js';
import { measureTraits, TRAITS } from '../../engine/src/model/traits.js';
import { artistScores, topK, ndcg, tuneHalfLife, listenCounts, clock, hourOfWeek } from '../../engine/src/model/taste.js';
import { weekly, changePoints, eraScores } from '../../engine/src/model/eras.js';
import { listsOf, newArtists, readDoor, matchDropped, FACETS, recentFavs, Door, Lists } from '../../engine/src/model/doors.js';
import { newArtistRows, stickFeatures, horizonEnd, baseRateLogit, StickContext, NewArtist, Horizon } from '../../engine/src/model/stick.js';
import { favourites, candidates, ListIndex, recallAt } from '../../engine/src/model/rank2.js';
import { knownBefore, findFalls, FALL_WEEKS } from '../../engine/src/model/falls.js';
import { rng } from '../../engine/src/model/linalg.js';

const [shard, shards] = (process.argv[2] ?? '0/1').split('/').map(Number);
const STICK_TRAIN_CAP = 4000, R2_WINDOWS = 12;
const pop = readJsonGz<Pop>(cachePath('pop.json.gz'));
const T = lbTable(), L: Lists = listsOf(T);
log(`lists: ${Object.keys(T.lists).length} artists with a ListenBrainz list`);
// lists by key for the ranker: lister -> (artist -> [rank, score / top score])
const listCache = new Map<string, Map<string, [number, number]> | undefined>();
const lists: ListIndex = {
  get(a) {
    if (listCache.has(a)) return listCache.get(a);
    const i = L.ix.get(a), raw = i === undefined ? undefined : T.lists[i];
    let m: Map<string, [number, number]> | undefined;
    if (raw?.length) { const top = raw[0][1] || 1; m = new Map(); raw.forEach(([n, s], r) => { const k = T.nodes[n]; if (!m!.has(k)) m!.set(k, [r + 1, s / top]); }); }
    listCache.set(a, m);
    return m;
  },
};
const maps = new Map<number, TasteMap>();
const mapOf = (f: number) => maps.get(f) ?? (maps.set(f, loadMap(cachePath('maps', `fold${f}`))), maps.get(f)!);
const popOrder = Object.keys(pop.people).sort((a, b) => pop.people[b] - pop.people[a] || pop.plays[b] - pop.plays[a] || (a < b ? -1 : 1)).slice(0, 20000);
// buckets of never-played controls for the list fact: ListenBrainz artists by how many lists name them
const buckets = new Map<number, number[]>();
L.listedBy.forEach((c, i) => { if (c) { const b = Math.round(2 * Math.log2(c)); (buckets.get(b) ?? buckets.set(b, []).get(b)!).push(i); } });
const rows = Object.entries(pop.rows);

const f5 = (x: number) => (Number.isFinite(x) ? Math.round(x * 1e5) / 1e5 : null);
const todo = ids().filter((id) => hash(id) % shards === shard && !existsSync(cachePath('person', `${id}.json`)));
log(`shard ${shard}/${shards}: ${todo.length} people to do`);

for (const id of todo) {
  const t0 = Date.now();
  const P = loadPerson(id);
  if (!P) { writeFileSync(cachePath('person', `${id}.json`), JSON.stringify({ id, skipped: true })); continue; }
  const map = mapOf(foldOf(id)), { plays, n, cut } = P, train = plays.slice(0, n), test = plays.slice(n), now = train[train.length - 1].t;
  const own = new Set(train.map((p) => p.artist));
  const others = { get: (a: string) => Math.max(0, (pop.people[a] ?? 0) - (own.has(a) ? 1 : 0)) } as unknown as Map<string, number>;
  const ownCounts = listenCounts(train);
  const othersPlays = { get: (a: string) => Math.max(0, (pop.plays[a] ?? 0) - (ownCounts.get(a) ?? 0)) } as unknown as Map<string, number>;
  const out: Record<string, unknown> = { id, fold: foldOf(id), plays: plays.length, train: n, days: (plays[plays.length - 1].t - plays[0].t) / DAY };

  // ---- traits on the training weeks, and on each half (the second half only uses the first to know who was familiar)
  {
    const mid = (train[0].t + train[train.length - 1].t) / 2, a = train.filter((p) => p.t < mid), b = train.filter((p) => p.t >= mid);
    const vals = (ts: ReturnType<typeof measureTraits>) => Object.fromEntries(TRAITS.map((d) => [d.key, f5(ts.find((t) => t.key === d.key)?.value ?? NaN)]));
    out.traits = { full: vals(measureTraits(train, map)), a: vals(measureTraits(a, map)), b: vals(measureTraits(b, map, a)) };
  }

  // ---- next 8 weeks' artists, and the clock (UTC hour of the week: a person's zone is fixed, so a
  // personal clock in UTC predicts as well as a local one, apart from daylight saving shifts)
  const truth = listenCounts(test), hl = tuneHalfLife(train);
  const r: Record<string, number | null> = { hl };
  r.artists = f5(ndcg(topK(artistScores(train, now, hl), 10), truth));
  r.artists_allTime = f5(ndcg(topK(artistScores(train, now, 1e6), 10), truth));
  r.artists_last4w = f5(ndcg(topK(listenCounts(train.filter((p) => p.t > now - 28 * DAY)), 10), truth));
  {
    const how = (p: Play) => hourOfWeek(p.t), c2 = now - HOLDOUT_WEEKS * WEEK, fit = train.filter((p) => p.t < c2), val = train.filter((p) => p.t >= c2);
    const sc = (x: number) => { const cl = clock(fit, c2, x, 1, how); return val.reduce((s, p) => s + Math.log(cl[how(p)]), 0); };
    const chl = fit.length < 500 || val.length < 100 ? 1e6 : [30, 90, 365, 1e6].reduce((b, h) => (sc(h) > sc(b) ? h : b), 1e6);
    const bits = (cl: number[]) => test.reduce((s, p) => s + Math.log2(cl[how(p)] * 168), 0) / test.length;
    r.clock = f5(bits(clock(train, now, chl, 1, how))); r.clock_allTime = f5(bits(clock(train, now, 1e6, 1, how)));
  }

  // ---- eras (no selves: this bench has no skips to build them from), tuned on validation as in run.ts
  {
    const first = train[0].t, vcut = cut - HOLDOUT_WEEKS * WEEK, fit = train.filter((p) => p.t < vcut), val = train.filter((p) => p.t >= vcut), vTruth = listenCounts(val);
    let best = { pen: 1, gamma: 1, h: hl, score: -1 };
    if (fit.length > 500 && val.length > 100 && vcut - first > 30 * WEEK) {
      const wkV = weekly(fit, [], first, vcut, map, [], null, utcHour), vNow = fit[fit.length - 1].t;
      for (const pen of [1, 0.5, 2, 4]) {
        const st = changePoints(wkV.X, 10, pen), eraStart = wkV.start[st[st.length - 1]];
        for (const gamma of [0, 0.05, 0.2, 0.5, 1]) for (const h of [hl, 1e6]) {
          const s = ndcg(topK(eraScores(fit, vNow + 1, eraStart, h, gamma), 10), vTruth);
          if (s > best.score + 1e-9) best = { pen, gamma, h, score: s };
        }
      }
    }
    const wk = weekly(train, [], first, cut, map, [], null, utcHour);
    const starts = changePoints(wk.X, 10, best.pen), eraStart = wk.start[starts[starts.length - 1]];
    r.eras = starts.length; r.eraGamma = best.gamma;
    r.eraForecast = f5(ndcg(topK(eraScores(train, now + 1, eraStart, best.h, best.gamma), 10), truth));
    r.eraOnly = f5(ndcg(topK(eraScores(train, now + 1, eraStart, 1e6, 0), 10), truth));
  }
  out.tasks = r;

  // ---- doors: the facts a history without start reasons can still give, falls vs tried-and-dropped
  {
    const doors = newArtists(plays).map(({ i, kind, daysTo10 }) => readDoor(plays, i, kind, L, daysTo10));
    const falls = doors.filter((d) => d.kind === 'fall'), dropped = doors.filter((d) => d.kind === 'dropped'), matched = matchDropped(doors);
    const tally = (f: (d: Door) => boolean | null) => {
      const t = { fallYes: 0, fallN: 0, ctlYes: 0, ctlN: 0 };
      for (const [fd, ctl] of matched) {
        const v = f(fd);
        if (v === null) continue;
        const cv = ctl.map(f).filter((x): x is boolean => x !== null);
        if (!cv.length) continue;
        t.fallN++; if (v) t.fallYes++;
        t.ctlN++; t.ctlYes += cv.filter(Boolean).length / cv.length;
      }
      return t;
    };
    const facets = Object.fromEntries(['prevLists', 'favLists', 'favListsTop25', 'prevIsFav'].map((k) => [k, tally(FACETS[k])]));
    // the list fact against never-played ListenBrainz artists named on as many lists
    const everPlayed = new Set(plays.map((p) => p.artist)), rr = rng(hash(id)), unheard = { fallYes: 0, fallN: 0, ctlYes: 0, ctlN: 0 };
    for (const fd of falls) {
      const ix = L.ix.get(fd.artist);
      if (ix === undefined || !L.listedBy[ix]) continue;
      const pool = (buckets.get(Math.round(2 * Math.log2(L.listedBy[ix]))) ?? []).filter((j) => !everPlayed.has(L.nodes[j]));
      if (pool.length < 5) continue;
      const i = plays.findIndex((p) => p.t === fd.first && p.artist === fd.artist);
      const favs = [...recentFavs(plays, i).keys()];
      const best = (a: string) => favs.some((f) => L.rank(f, a) !== null);
      let yes = 0;
      for (let k = 0; k < 5; k++) if (best(L.nodes[pool[Math.floor(rr() * pool.length)]])) yes++;
      unheard.fallN++; if (fd.favRank !== null) unheard.fallYes++;
      unheard.ctlN++; unheard.ctlYes += yes / 5;
    }
    out.doors = { falls: falls.length, dropped: dropped.length, matched: matched.size, facets, unheard, growth: falls.map((d) => f5(d.daysTo10 ?? NaN)) };
  }

  // ---- will it stick: feature rows (all columns; models.ts picks the ones this bench can fill)
  {
    const all = newArtistRows(plays), rr = rng(hash(id) + 3);
    let trainRows = all.filter((x) => x.t0 + 8 * WEEK <= cut);
    const held = all.filter((x) => x.t0 >= cut - 8 * WEEK && x.t0 < cut);
    if (trainRows.length > STICK_TRAIN_CAP) { trainRows = trainRows.map((x) => [rr(), x] as const).sort((a, b) => a[0] - b[0]).slice(0, STICK_TRAIN_CAP).map(([, x]) => x).sort((a, b) => a.i - b.i); }
    let on = 0, oy = 0;
    for (const [k, [nn, yy]] of rows) if (k !== id) { on += nn; oy += yy; }
    const prior = oy / Math.max(1, on), cache = new Map<number, Map<string, number>>();
    const ctx: StickContext = { lists: L, map, people: others, hour: utcHour };
    let checked = 0, failed = 0;
    const build = (rs: NewArtist[], h: Horizon) => {
      const keep = h === 'week' ? rs.filter((x) => !x.fallenByWeek) : rs;
      return {
        X: keep.map((x) => {
          const row = stickFeatures(plays, x.i, h, ctx, baseRateLogit(all, x.t0, prior), cache);
          if (rr() < 0.01) {
            const cutPlays = plays.slice(0, horizonEnd(plays, x.i, h) + 1);
            const again = stickFeatures(cutPlays, x.i, h, ctx, baseRateLogit(all, x.t0, prior));
            checked++; if (again.some((v, j) => Math.abs(v - row[j]) > 1e-12)) failed++;
          }
          return row.map((v) => f5(v));
        }),
        y: keep.map((x) => x.label),
      };
    };
    writeJsonGz(cachePath('stick', `${id}.json.gz`), { session: { train: build(trainRows, 'session'), held: build(held, 'session') }, week: { train: build(trainRows, 'week'), held: build(held, 'week') }, leak: { checked, failed }, prior });
    out.stickRows = { train: trainRows.length, held: held.length, all: all.length, pos: all.reduce((s, x) => s + x.label, 0), leakChecked: checked, leakFailed: failed };
  }

  // ---- two-stage ranker: candidates per window (rolling windows inside training, and the held-out one)
  {
    const first = train[0].t, wins: { start: number; held: boolean }[] = [];
    for (let k = 1; k <= R2_WINDOWS; k++) { const start = cut - k * FALL_WEEKS * WEEK; if (start - first < 26 * WEEK) break; wins.push({ start, held: false }); }
    wins.push({ start: cut, held: true });
    wins.sort((a, b) => a.start - b.start);
    const pp = { people: others, plays: othersPlays };
    const W = wins.map((w) => {
      const src = w.held ? plays : train, known = knownBefore(src, w.start), loves = new Set(findFalls(src, w.start, known));
      const favs = favourites(train, w.start), cs = candidates(favs, known, lists, map, pp, 0);
      const listed = [...loves].filter((a) => favs.some((f) => lists.get(f.artist)?.has(a))).length;
      const popular = popOrder.map((a) => [a, others.get(a)!] as const).filter(([a]) => !known.has(a)).sort((x, y) => y[1] - x[1] || (pop.plays[y[0]] ?? 0) - (pop.plays[x[0]] ?? 0)).slice(0, 50).map(([a]) => a);
      return {
        start: w.start, held: w.held, loves: loves.size, listed,
        popular20: recallAt(popular, loves, 20), popular50: recallAt(popular, loves, 50),
        X: cs.map((c) => c.x.map((v) => f5(v))), y: cs.map((c) => (loves.has(c.artist) ? 1 : 0)),
      };
    });
    writeJsonGz(cachePath('rank2', `${id}.json.gz`), W);
    out.rank2 = W.map((w) => ({ held: w.held, loves: w.loves, cands: w.y.length, covered: w.y.reduce((s: number, v) => s + v, 0) }));
  }
  writeFileSync(cachePath('person', `${id}.json`), JSON.stringify(out));
  log(`${id} ${plays.length} plays, done in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
log(`shard ${shard} finished`);

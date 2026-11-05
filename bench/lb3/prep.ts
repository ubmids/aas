// Round 3, step 1: what every later step shares.
//   pop.json.gz: for each artist, how many people played it (properly) in their training weeks and
//     how many plays that was; for each person, how many new artists they met in training and how
//     many stuck (the prior for everyone else's base rate).
//   maps/fold<f>: a co-listening taste map (the same recipe as map.ts) built from the training weeks
//     of the people in the other four folds. The vocabulary is every artist that at least MIN_PEOPLE
//     of those people played, so the map stays a few tens of thousands of artists. People are
//     streamed one at a time; nothing holds the whole bench in memory.
// Resumable: finished outputs are skipped. Run: npx tsx bench/lb3/prep.ts
import { existsSync } from 'node:fs';
import { ids, loadPerson, foldOf, FOLDS, cachePath, writeJsonGz, saveMap, log, Pop } from './common.js';
import { tokens, mapFromSequences } from '../../engine/src/model/map.js';
import { newArtistRows } from '../../engine/src/model/stick.js';
import { isSkip } from '../../engine/src/parse/spotify.js';

const MIN_PEOPLE = 5;
const all = ids();
const sets = new Map<string, Set<string>>();
const pop: Pop = { people: {}, plays: {}, rows: {}, loves: {} };
let t0 = Date.now();
for (const id of all) {
  const P = loadPerson(id);
  if (!P) { log(`${id} too short, left out`); continue; }
  const c = new Map<string, number>();
  for (let i = 0; i < P.n; i++) { const p = P.plays[i]; if (!isSkip(p)) c.set(p.artist, (c.get(p.artist) ?? 0) + 1); }
  sets.set(id, new Set(c.keys()));
  for (const [a, v] of c) { pop.people[a] = (pop.people[a] ?? 0) + 1; pop.plays[a] = (pop.plays[a] ?? 0) + v; }
  const rows = newArtistRows(P.plays).filter((r) => r.t0 + 8 * 7 * 86_400_000 <= P.cut);
  pop.rows[id] = [rows.length, rows.reduce((s, r) => s + r.label, 0)];
}
writeJsonGz(cachePath('pop.json.gz'), pop);
log(`pop: ${sets.size} people, ${Object.keys(pop.people).length} artists, ${((Date.now() - t0) / 1000).toFixed(0)}s`);

for (let f = 0; f < FOLDS; f++) {
  const out = cachePath('maps', `fold${f}`);
  if (existsSync(out + '.bin')) { log(`fold ${f} map cached`); continue; }
  t0 = Date.now();
  const inFold = all.filter((id) => sets.has(id) && foldOf(id) === f);
  const cnt = new Map<string, number>();
  for (const [id, s] of sets) if (foldOf(id) !== f) for (const a of s) cnt.set(a, (cnt.get(a) ?? 0) + 1);
  const vocab = [...cnt].filter(([, v]) => v >= MIN_PEOPLE).map(([a]) => a).sort();
  const vid = new Map(vocab.map((a, i) => [a, i]));
  function* seqs() {
    for (const id of all) {
      if (!sets.has(id) || foldOf(id) === f) continue;
      const P = loadPerson(id)!;
      for (const seq of tokens(P.plays.slice(0, P.n))) yield Int32Array.from(seq, (a) => vid.get(a) ?? -1);
    }
  }
  const m = mapFromSequences(vocab, seqs());
  saveMap(out, m);
  log(`fold ${f}: ${inFold.length} people held out, vocabulary ${vocab.length}, map ${m.artists.length} artists, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}

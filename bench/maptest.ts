// Quick sanity read of the taste map: size, build time, and neighbours of each person's top artists.
import { loadPeople } from './load.js';
import { buildMap, neighbours } from '../engine/src/model/map.js';
import { listenCounts, topK } from '../engine/src/model/taste.js';

const people = loadPeople();
let t = Date.now();
const map = buildMap(people.map((p) => p.plays.slice(0, p.n)));
console.log(`map: ${map.artists.length} artists x ${map.dim}, ${((Date.now() - t) / 1000).toFixed(1)}s, heap ${(process.memoryUsage().rss / 1e9).toFixed(2)} GB`);
for (const p of people.slice(0, 6)) {
  const top = topK(listenCounts(p.plays.slice(0, p.n)), 2);
  for (const a of top) console.log(`  ${a} -> ${neighbours(map, a, 6).join(', ')}`);
}

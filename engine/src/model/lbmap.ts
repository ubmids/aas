// Taste map v1: artists placed from ListenBrainz's public similar-artist lists (session based:
// two artists are similar when ListenBrainz users play them in the same listening sessions).
// Each seed artist brings its top 100 list, and (hops 2) so does every artist on those lists, so
// where an artist sits never depends on whether someone in the bench played it. The lists become
// one weighted graph (edge weight = the list's score over its top score), the graph goes through
// the same PPMI and eigenvector steps as v0, and every node with enough evidence becomes a point. Unlike v0 the points are not
// limited to artists someone in the bench played: neighbours nobody here has heard are on the map
// too, which is where most falls live.
import { TasteMap } from './map.js';
import { Csr, topEigSparse, dot } from './linalg.js';

export interface LbTable {
  nodes: string[]; // bench spelling where the artist resolved, else "lb:<ListenBrainz name>"
  lists: Record<string, [number, number][]>; // node index -> [node index, score], best first
}

export interface LbMap extends TasteMap {
  listed: Float64Array; // per artist: how many of the lists used list it (a popularity prior)
}

export interface LbOptions { dim?: number; minLists?: number; hops?: 1 | 2; seed?: number }

/** Builds the map from the lists of the allowed seeds (and, hops 2, of the artists they list). */
export function buildLbMap(t: LbTable, seeds: Iterable<string>, opt: LbOptions = {}): LbMap {
  const dim = opt.dim ?? 64, minLists = opt.minLists ?? 2, hops = opt.hops ?? 2;
  const nodeIx = new Map(t.nodes.map((n, i) => [n, i]));
  const edges = new Map<number, Map<number, number>>(), lists = new Map<number, number>(), isSeed = new Set<number>();
  const add = (a: number, b: number, w: number) => {
    const r = edges.get(a) ?? edges.set(a, new Map()).get(a)!;
    r.set(b, (r.get(b) ?? 0) + w);
  };
  const use = new Set<number>();
  for (const s of seeds) { const i = nodeIx.get(s); if (i !== undefined && t.lists[i]?.length) use.add(i); }
  if (hops === 2) for (const i of [...use]) for (const [n] of t.lists[i]) if (t.lists[n]?.length) use.add(n);
  for (const si of use) {
    const list = t.lists[si];
    isSeed.add(si);
    const top = list[0][1] || 1;
    for (const [n, score] of list) {
      if (n === si) continue;
      const w = score / top;
      add(si, n, w); add(n, si, w);
      lists.set(n, (lists.get(n) ?? 0) + 1);
    }
  }
  // a point needs its own list or to be listed by minLists seeds
  const vocab = [...edges.keys()].filter((i) => isSeed.has(i) || (lists.get(i) ?? 0) >= minLists).sort((a, b) => a - b);
  const id = new Map(vocab.map((n, i) => [n, i]));
  const V = vocab.length;
  const rowSum = new Float64Array(V);
  vocab.forEach((n, i) => { for (const [m, w] of edges.get(n)!) if (id.has(m)) rowSum[i] += w; });
  const ctx = Array.from(rowSum, (v) => v ** 0.75), ctxTotal = ctx.reduce((s, v) => s + v, 0);
  const rowPtr = new Int32Array(V + 1), cols: number[] = [], vals: number[] = [];
  for (let i = 0; i < V; i++) {
    const row = [...edges.get(vocab[i])!].map(([m, w]) => [id.get(m) ?? -1, w] as const).filter(([j]) => j >= 0).sort((x, y) => x[0] - y[0]);
    for (const [j, c] of row) {
      const v = Math.log((c * ctxTotal) / (rowSum[i] * ctx[j]));
      if (v > 0) { cols.push(j); vals.push(v); }
    }
    rowPtr[i + 1] = cols.length;
  }
  const A: Csr = { n: V, rowPtr, col: Int32Array.from(cols), val: Float64Array.from(vals) };
  const { U, values } = topEigSparse(A, dim, 16, 3, opt.seed ?? 11);
  const scale = values.map((v) => Math.sqrt(Math.max(v, 0)));
  const keep: number[] = [];
  for (let i = 0; i < V; i++) {
    let s = 0;
    for (let c = 0; c < dim; c++) { U[i * dim + c] *= scale[c]; s += U[i * dim + c] ** 2; }
    if (s > 1e-18 && rowPtr[i + 1] > rowPtr[i]) {
      s = Math.sqrt(s);
      for (let c = 0; c < dim; c++) U[i * dim + c] /= s;
      keep.push(i);
    }
  }
  const vec = new Float64Array(keep.length * dim), listed = new Float64Array(keep.length);
  keep.forEach((i, k) => { vec.set(U.subarray(i * dim, (i + 1) * dim), k * dim); listed[k] = lists.get(vocab[i]) ?? 0; });
  const artists = keep.map((i) => t.nodes[vocab[i]]);
  return { artists, index: new Map(artists.map((a, i) => [a, i])), dim, vec, listed };
}

/**
 * Two maps as one: each artist's point is [sqrt(w) * a ; sqrt(1 - w) * b], unit-normalised, so the
 * cosine of two artists on both maps is the w-weighted mix of their two cosines. An artist on only
 * one map keeps that map's half (and its cosines shrink by that half's weight).
 */
export function blendMaps(a: TasteMap, b: LbMap, w = 0.5): LbMap {
  const names = [...new Set([...a.artists, ...b.artists])].sort();
  const dim = a.dim + b.dim, vec = new Float64Array(names.length * dim), listed = new Float64Array(names.length);
  const wa = Math.sqrt(w), wb = Math.sqrt(1 - w);
  names.forEach((n, k) => {
    const o = k * dim, ia = a.index.get(n), ib = b.index.get(n);
    if (ia !== undefined) for (let j = 0; j < a.dim; j++) vec[o + j] = wa * a.vec[ia * a.dim + j];
    if (ib !== undefined) { for (let j = 0; j < b.dim; j++) vec[o + a.dim + j] = wb * b.vec[ib * b.dim + j]; listed[k] = b.listed[ib]; }
    const row = vec.subarray(o, o + dim), s = Math.sqrt(dot(row, row)) || 1;
    for (let j = 0; j < dim; j++) row[j] /= s;
  });
  return { artists: names, index: new Map(names.map((n, i) => [n, i])), dim, vec, listed };
}

/**
 * Direct evidence, no embedding: how strongly the person's recent artists list each artist,
 * sum over recent artists of (plays share) x (score / list's top score). Keys are node names.
 */
export function listEvidence(t: LbTable, recent: Map<string, number>, nodeIx?: Map<string, number>): Map<string, number> {
  const ix = nodeIx ?? new Map(t.nodes.map((n, i) => [n, i]));
  const out = new Map<string, number>();
  let tot = 0;
  for (const w of recent.values()) tot += w;
  if (!tot) return out;
  for (const [a, w] of recent) {
    const i = ix.get(a), list = i === undefined ? undefined : t.lists[i];
    if (!list?.length) continue;
    const top = list[0][1] || 1;
    for (const [n, s] of list) { const name = t.nodes[n]; out.set(name, (out.get(name) ?? 0) + (w / tot) * (s / top)); }
  }
  return out;
}

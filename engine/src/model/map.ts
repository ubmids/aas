// Taste map v0: artists as points, learned from session co-listening. Two artists are close when
// people play them near each other inside the same session (mood, not just "same fans").
// Method: windowed co-occurrence inside sessions -> PPMI (context smoothing 0.75) -> top
// eigenvectors (randomised) -> U sqrt(lambda), rows L2-normalised. Levy and Goldberg 2014.
import { Play, isSkip } from '../parse/spotify.js';
import { sessions } from './session.js';
import { Csr, topEigSparse, dot } from './linalg.js';

export interface TasteMap {
  artists: string[];
  index: Map<string, number>;
  dim: number;
  vec: Float64Array; // artists.length x dim, unit rows
}

export interface MapOptions { dim?: number; window?: number; minPlays?: number; minSessions?: number; seed?: number }

/** Artist tokens per session: proper plays only (>= 30 s), runs of the same artist collapsed. */
export function tokens(plays: Play[]): string[][] {
  const out: string[][] = [];
  for (const s of sessions(plays)) {
    const seq: string[] = [];
    for (const p of s.plays) if (!isSkip(p) && seq[seq.length - 1] !== p.artist) seq.push(p.artist);
    if (seq.length > 1) out.push(seq);
  }
  return out;
}

export function buildMap(sources: Play[][], opt: MapOptions = {}): TasteMap {
  const dim = opt.dim ?? 48, W = opt.window ?? 5, minPlays = opt.minPlays ?? 3, minSessions = opt.minSessions ?? 2;
  // vocabulary
  const plays = new Map<string, number>(), sess = new Map<string, number>();
  const toks: string[][] = [];
  for (const src of sources) {
    for (const p of src) if (!isSkip(p)) plays.set(p.artist, (plays.get(p.artist) ?? 0) + 1);
    for (const seq of tokens(src)) {
      toks.push(seq);
      for (const a of new Set(seq)) sess.set(a, (sess.get(a) ?? 0) + 1);
    }
  }
  const vocab = [...sess.keys()].filter((a) => (plays.get(a) ?? 0) >= minPlays && (sess.get(a) ?? 0) >= minSessions).sort();
  const id = new Map(vocab.map((a, i) => [a, i]));
  return mapFromSequences(vocab, toks.map((seq) => seq.map((a) => id.get(a) ?? -1)), { dim, window: W, seed: opt.seed });
}

/**
 * The map from session sequences already turned into vocabulary ids (-1 = not in the vocabulary):
 * windowed co-occurrence (weight 1/distance), PPMI with context smoothing 0.75, top eigenvectors,
 * U sqrt(lambda), unit rows. `buildMap` is this with its own vocabulary; the ListenBrainz-user bench
 * calls it directly with a vocabulary chosen by how many people play each artist and sequences
 * streamed one person at a time.
 */
export function mapFromSequences(vocab: string[], seqs: Iterable<ArrayLike<number>>, opt: { dim?: number; window?: number; seed?: number } = {}): TasteMap {
  const dim = opt.dim ?? 48, W = opt.window ?? 5;
  const V = vocab.length;
  // windowed co-occurrence (weight 1/distance), symmetric
  const rows: Map<number, number>[] = Array.from({ length: V }, () => new Map());
  for (const ix of seqs) {
    for (let i = 0; i < ix.length; i++) {
      if (ix[i] < 0) continue;
      for (let j = i + 1; j <= i + W && j < ix.length; j++) {
        const a = ix[i], b = ix[j];
        if (b < 0 || a === b) continue;
        const w = 1 / (j - i);
        rows[a].set(b, (rows[a].get(b) ?? 0) + w);
        rows[b].set(a, (rows[b].get(a) ?? 0) + w);
      }
    }
  }
  const rowSum = rows.map((r) => [...r.values()].reduce((s, v) => s + v, 0));
  const ctx = rowSum.map((v) => v ** 0.75), ctxTotal = ctx.reduce((s, v) => s + v, 0);
  // PPMI in CSR
  const rowPtr = new Int32Array(V + 1);
  const cols: number[] = [], vals: number[] = [];
  for (let i = 0; i < V; i++) {
    for (const [j, c] of [...rows[i].entries()].sort((x, y) => x[0] - y[0])) {
      const v = Math.log((c * ctxTotal) / (rowSum[i] * ctx[j]));
      if (v > 0) { cols.push(j); vals.push(v); }
    }
    rowPtr[i + 1] = cols.length;
    rows[i].clear();
  }
  const A: Csr = { n: V, rowPtr, col: Int32Array.from(cols), val: Float64Array.from(vals) };
  const { U, values } = topEigSparse(A, dim, 16, 3, opt.seed ?? 11);
  // keep artists whose vector is non-zero; scale by sqrt(lambda) then unit-normalise
  const keep: number[] = [];
  const scale = values.map((v) => Math.sqrt(Math.max(v, 0)));
  for (let i = 0; i < V; i++) {
    let s = 0;
    for (let c = 0; c < dim; c++) { U[i * dim + c] *= scale[c]; s += U[i * dim + c] ** 2; }
    if (s > 1e-18 && rowPtr[i + 1] > rowPtr[i]) {
      s = Math.sqrt(s);
      for (let c = 0; c < dim; c++) U[i * dim + c] /= s;
      keep.push(i);
    }
  }
  const vec = new Float64Array(keep.length * dim);
  keep.forEach((i, k) => vec.set(U.subarray(i * dim, (i + 1) * dim), k * dim));
  const artists = keep.map((i) => vocab[i]);
  return { artists, index: new Map(artists.map((a, i) => [a, i])), dim, vec };
}

export const vecOf = (m: TasteMap, artist: string): Float64Array | null => {
  const i = m.index.get(artist);
  return i === undefined ? null : m.vec.subarray(i * m.dim, (i + 1) * m.dim);
};

/** Weighted mean position of a set of artists (unit-normalised). Null if none are on the map. */
export function centroid(m: TasteMap, weights: Map<string, number>): Float64Array | null {
  const c = new Float64Array(m.dim);
  let tot = 0;
  for (const [a, w] of weights) {
    const v = vecOf(m, a);
    if (!v || w <= 0) continue;
    for (let j = 0; j < m.dim; j++) c[j] += w * v[j];
    tot += w;
  }
  if (!tot) return null;
  const n = Math.sqrt(dot(c, c)) || 1;
  for (let j = 0; j < m.dim; j++) c[j] /= n;
  return c;
}

/** Cosine similarity of every map artist to a unit query vector. */
export function simAll(m: TasteMap, q: ArrayLike<number>): Float64Array {
  const out = new Float64Array(m.artists.length);
  for (let i = 0; i < out.length; i++) {
    let s = 0;
    const o = i * m.dim;
    for (let j = 0; j < m.dim; j++) s += m.vec[o + j] * q[j];
    out[i] = s;
  }
  return out;
}

/** Nearest artists to a named one, for sanity reads. */
export function neighbours(m: TasteMap, artist: string, k = 8): string[] {
  const v = vecOf(m, artist);
  if (!v) return [];
  const s = simAll(m, v);
  return [...s.keys()].filter((i) => m.artists[i] !== artist).sort((a, b) => s[b] - s[a]).slice(0, k).map((i) => m.artists[i]);
}

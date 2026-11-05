// Loads the bench people one line at a time (the raw files are large) and splits each by time.
import { readFileSync, readdirSync } from 'node:fs';
import { parseSpotify, Play, Episode } from '../engine/src/parse/spotify.js';

export const DIR = 'bench/data/spotify_full';
export const HOLDOUT_WEEKS = 8;
export const DAY = 86_400_000;

export interface Person { name: string; plays: Play[]; episodes: Episode[]; n: number; cut: number }

function* lines(file: string) {
  const text = readFileSync(file, 'utf8');
  let i = 0;
  while (i < text.length) {
    let j = text.indexOf('\n', i);
    if (j < 0) j = text.length;
    if (j > i) yield JSON.parse(text.slice(i, j));
    i = j + 1;
  }
}

export function loadPeople(): Person[] {
  const pool = new Map<string, string>();
  const intern = (s: string) => pool.get(s) ?? (pool.set(s, s), s);
  const people: Person[] = [];
  for (const file of readdirSync(DIR).filter((f) => f.endsWith('.jsonl')).sort()) {
    const episodes: Episode[] = [];
    const plays = parseSpotify(lines(`${DIR}/${file}`), episodes);
    if (plays.length < 2000) continue;
    for (const p of plays) { p.artist = intern(p.artist); if (p.platform) p.platform = intern(p.platform); if (p.country) p.country = intern(p.country); }
    episodes.sort((a, b) => a.t - b.t);
    const cut = plays[plays.length - 1].t - HOLDOUT_WEEKS * 7 * DAY;
    const n = plays.findIndex((p) => p.t >= cut);
    if (n < 1000 || plays.length - n < 200) continue;
    people.push({ name: file.replace('.jsonl', ''), plays, episodes, n, cut });
  }
  return people;
}

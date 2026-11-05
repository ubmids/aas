// Measured listening traits, in place of inferred personality. Each trait is a number read straight
// from the listening history, with the receipts that produced it, and a percentile against other
// people (the bench now, the app's users later). Nothing here guesses who the person is: a trait
// says how they listen. A "people like you tend to" line is only attached where published research
// backs it, worded as a tendency (see DESIGN.md, Traits, for the sources).
//
// Hook for later: `TraitLabel` holds self-report answers from an in-app quiz, so traits can one day
// be checked against what people say about themselves. No quiz is built yet.
import { Play, isSkip } from '../parse/spotify.js';
import { TasteMap, vecOf } from './map.js';
import { localClock, zoneOffset } from './session.js';

const DAY = 86_400_000, WEEK = 7 * DAY, BLOCK = 13 * WEEK;

export interface TraitValue { key: TraitKey; value: number; receipt: string }
export type TraitKey = 'newShare' | 'spread' | 'oldFriends' | 'finished' | 'skipSeconds' | 'clockHour' | 'nightShare' | 'clockSpread' | 'clockRegular' | 'turnover';

export interface TraitDef {
  key: TraitKey;
  group: 'adventurousness' | 'loyalty' | 'patience' | 'clock' | 'era turnover';
  label: string;
  unit: string;
  /**
   * From the test-retest check (halves of each person's history): stable when Pearson and Spearman
   * are both >= 0.5, flagged when only one is, dropped when neither is. Dropped traits are measured
   * but never put on a card. Round 3 (bench/lb3, 304 ListenBrainz listeners) settles the traits it
   * can measure: there every one has both bootstrap intervals above 0.5, so spread and oldFriends,
   * flagged on the 13-person Spotify bench, are stable. Traits that need skips, play lengths or a
   * time zone keep their Spotify status.
   */
  status: 'stable' | 'flagged' | 'dropped';
  /** a published tendency, worded as a tendency; key into SOURCES. Omitted where no research backs a line. */
  tendency?: { text: string; source: string };
}

export const SOURCES: Record<string, string> = {
  ward2014: 'Ward, M. K., Goodman, J. K. and Irwin, J. R. (2014). The same old song: the power of familiarity in music choice. Marketing Letters, 25(1), 1-11.',
  park2019: 'Park, M., Thom, J., Mennicken, S., Cramer, H. and Macy, M. (2019). Global music streaming data reveal diurnal and seasonal patterns of affective preference. Nature Human Behaviour, 3(3). doi:10.1038/s41562-018-0508-z',
  roenneberg2003: 'Roenneberg, T., Wirz-Justice, A. and Merrow, M. (2003). Life between clocks: daily temporal patterns of human chronotypes. Journal of Biological Rhythms, 18(1), 80-90. doi:10.1177/0748730402239679',
};

export const TRAITS: TraitDef[] = [
  // no tendency line: the personality research we know links openness to genres, not to how much
  // someone explores, so nothing published backs a line here
  { key: 'newShare', status: 'stable', group: 'adventurousness', label: 'plays that go to artists you met in the last 4 weeks', unit: 'share' },
  { key: 'spread', status: 'stable', group: 'adventurousness', label: 'how far apart your artists sit on the taste map', unit: '0 to 1' },
  { key: 'oldFriends', status: 'stable', group: 'loyalty', label: 'listening that goes to artists who were already your favourites 4 to 6 months earlier', unit: 'share',
    tendency: { text: 'Most people pick familiar songs more often than they themselves expect to.', source: 'ward2014' } },
  { key: 'finished', status: 'stable', group: 'patience', label: 'tracks played to the end', unit: 'share' },
  { key: 'skipSeconds', status: 'stable', group: 'patience', label: 'seconds before a skip, typically', unit: 'seconds' },
  { key: 'clockHour', status: 'stable', group: 'clock', label: 'the middle of your listening day, local time', unit: 'hour',
    tendency: { text: 'People\'s natural daily timing differs by several hours from one person to the next.', source: 'roenneberg2003' } },
  { key: 'nightShare', status: 'dropped', group: 'clock', label: 'plays between midnight and 5 am, local time', unit: 'share',
    tendency: { text: 'Across millions of listeners, late-night listening tends to be calmer and more relaxed than daytime listening.', source: 'park2019' } },
  // the next two do not depend on the time zone (a fixed offset turns every time by the same angle),
  // so they can be read from histories with UTC times only, such as the ListenBrainz-user bench
  { key: 'clockSpread', status: 'stable', group: 'clock', label: 'how spread your listening is across the hours of the day', unit: '0 to 1' },
  { key: 'clockRegular', status: 'stable', group: 'clock', label: 'how closely each day\'s listening keeps to the same part of the day', unit: '0 to 1' },
  { key: 'turnover', status: 'stable', group: 'era turnover', label: 'top 10 artists replaced from one season to the next', unit: 'share' },
];

/** Self-report answers from a future in-app quiz, keyed by trait group. Not collected yet. */
export interface TraitLabel { group: TraitDef['group']; answer: number; scale: [number, number]; askedAt: number }

const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
const medianOf = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const pct = (x: number) => `${Math.round(100 * x)}%`;

/** Top-k artists by proper plays in [a, b). */
function topIn(plays: Play[], a: number, b: number, k: number): string[] {
  const c = new Map<string, number>();
  for (const p of plays) if (p.t >= a && p.t < b && !isSkip(p)) c.set(p.artist, (c.get(p.artist) ?? 0) + 1);
  return [...c].sort((x, y) => y[1] - x[1]).slice(0, k).map(([a]) => a);
}

/**
 * All traits on one stretch of history (plays sorted by time). `before`: plays from earlier, only
 * used to know which artists were already familiar. Values are NaN where the data can't say.
 */
export function measureTraits(plays: Play[], m: TasteMap | null, before: Play[] = []): TraitValue[] {
  const out: TraitValue[] = [];
  if (plays.length < 500) return out;
  const t0 = plays[0].t, t1 = plays[plays.length - 1].t;
  const proper = plays.filter((p) => !isSkip(p));

  // adventurousness: plays to artists first heard in the 28 days before (after a 12 week warm-up)
  const first = new Map<string, number>();
  for (const p of before) if (!first.has(p.artist)) first.set(p.artist, p.t);
  for (const p of plays) if (!first.has(p.artist)) first.set(p.artist, p.t);
  const warm = before.length ? t0 : t0 + 12 * WEEK, scored = proper.filter((p) => p.t >= warm);
  const fresh = scored.filter((p) => p.t - first.get(p.artist)! < 28 * DAY);
  const newShare = scored.length >= 300 ? fresh.length / scored.length : NaN;
  const freshArtists = [...new Set(fresh.map((p) => p.artist))].slice(0, 3);
  out.push({ key: 'newShare', value: newShare, receipt: `${fresh.length} of ${scored.length} proper plays went to artists you'd known under 4 weeks (e.g. ${freshArtists.join(', ')})` });

  // adventurousness: spread on the map, 1 - length of the play-weighted mean of unit artist vectors
  if (m) {
    const v = new Float64Array(m.dim);
    let w = 0;
    for (const p of proper) { const a = vecOf(m, p.artist); if (!a) continue; for (let j = 0; j < m.dim; j++) v[j] += a[j]; w++; }
    const len = w ? Math.hypot(...v) / w : NaN;
    out.push({ key: 'spread', value: w >= 300 ? 1 - len : NaN, receipt: `${w} proper plays placed on the map` });
  }

  // loyalty: in each 8-week stretch, the share of proper plays going to artists that were in your
  // top 50 two or three stretches earlier (4 to 6 months before). The first version, "top 20 still
  // played 6 months on", failed the test-retest check (r -0.01), this one passes (see DESIGN.md).
  const SHORT = 8 * WEEK, stretches: Map<string, number>[] = [];
  for (let a = t0; a + SHORT <= t1 + 1; a += SHORT) {
    const c = new Map<string, number>();
    for (const p of proper) if (p.t >= a && p.t < a + SHORT) c.set(p.artist, (c.get(p.artist) ?? 0) + 1);
    stretches.push(c);
  }
  const top50 = (c: Map<string, number>) => [...c].sort((x, y) => y[1] - x[1]).slice(0, 50).map(([a]) => a);
  const shares: number[] = [], friends = new Map<string, number>();
  for (let i = 2; i < stretches.length; i++) {
    const old = new Set([...top50(stretches[i - 2]), ...(i >= 3 ? top50(stretches[i - 3]) : [])]);
    let n = 0, s = 0;
    for (const [a, v] of stretches[i]) { n += v; if (old.has(a)) { s += v; friends.set(a, (friends.get(a) ?? 0) + v); } }
    if (n > 100) shares.push(s / n);
  }
  const loyal = shares.length >= 2 ? mean(shares) : NaN;
  const named = [...friends].sort((x, y) => y[1] - x[1]).slice(0, 3).map(([a]) => a);
  out.push({ key: 'oldFriends', value: loyal, receipt: `over ${shares.length} two-month stretches, ${pct(loyal)} of your plays went to artists you already loved 4 to 6 months before (most of all ${named.join(', ')})` });

  // era turnover, over 13-week seasons
  const blocks: [number, number][] = [];
  for (let a = t0; a + BLOCK <= t1 + 1; a += BLOCK) blocks.push([a, a + BLOCK]);
  const tops10 = blocks.map(([a, b]) => topIn(plays, a, b, 10));
  const turns: number[] = [];
  for (let i = 0; i + 1 < blocks.length; i++) {
    if (tops10[i].length < 10 || tops10[i + 1].length < 10) continue;
    const next = new Set(tops10[i + 1]);
    turns.push(tops10[i].filter((a) => !next.has(a)).length / 10);
  }
  const turn = turns.length >= 2 ? mean(turns) : NaN;
  out.push({ key: 'turnover', value: turn, receipt: `over ${turns.length} season changes, ${pct(turn)} of the top 10 was new each time` });

  // patience: tracks played to the end (reason_end trackdone), and how long before a skip
  const ended = plays.filter((p) => p.reasonEnd);
  const done = ended.filter((p) => p.reasonEnd === 'trackdone').length;
  out.push({ key: 'finished', value: ended.length >= 300 ? done / ended.length : NaN, receipt: `${done} of ${ended.length} plays ran to the end of the track` });
  const skips = plays.filter((p) => p.ms > 0 && isSkip(p) && (p.reasonEnd === null || p.reasonEnd === 'fwdbtn'));
  const sec = skips.length >= 100 ? medianOf(skips.map((p) => p.ms / 1000)) : NaN;
  out.push({ key: 'skipSeconds', value: sec, receipt: `half of your ${skips.length} skips came within ${sec.toFixed(1)} s` });

  // clock: circular mean of local listening time (weighted by minutes), and the night share
  const clk = localClock([...before, ...plays]), knowsPlace = plays.some((p) => p.country && p.country !== 'ZZ');
  let sx = 0, sy = 0, night = 0, tot = 0;
  for (const p of proper) {
    const h = clk.hour(p), mins = Math.min(p.ms, 600_000);
    sx += mins * Math.cos((2 * Math.PI * h) / 24); sy += mins * Math.sin((2 * Math.PI * h) / 24); tot += mins;
    if (h < 5) night += mins;
  }
  let hour = (Math.atan2(sy, sx) * 24) / (2 * Math.PI);
  if (hour < 4) hour += 24; // read on a 4 am to 4 am day, so late listeners sort after early ones
  out.push({ key: 'clockHour', value: tot && knowsPlace ? hour : NaN, receipt: `the middle of your listening falls at ${fmtHour(hour)} (${clk.zone(null)})` });
  out.push({ key: 'nightShare', value: tot && knowsPlace ? night / tot : NaN, receipt: `${pct(night / tot)} of your listening minutes were between midnight and 5 am` });
  out.push(...clockShape(proper, (p) => (knowsPlace ? zoneOffset(clk.zone(p.country), p.t) : 0)));
  return out;
}

/**
 * Two clock traits that do not depend on the time zone. Each play is an angle on the 24 hour dial
 * (minutes since midnight, local where known, else UTC), weighted by its minutes (capped at 10).
 *   clockSpread: 1 - R, where R is the length of the weighted mean of those unit vectors. 0 means all
 *     listening at one time of day, 1 means spread evenly round the clock.
 *   clockRegular: the person's own day starts 12 hours away from the centre of their listening (the
 *     direction of the weighted mean), so the day boundary sits in the same place relative to their
 *     listening whatever their zone. Each day with 5+ proper plays gets its circular mean time;
 *     clockRegular is the length R of the mean of those daily angles (needs 30 such days). 1 means
 *     every day's listening centres on the same time.
 * A fixed offset rotates every angle by the same amount, which changes neither length.
 */
export function clockShape(proper: Play[], offsetMin: (p: Play) => number): TraitValue[] {
  const MIN = 1440, ang = (m: number) => (2 * Math.PI * m) / MIN;
  const local = proper.map((p) => { const m = (Math.floor(p.t / 60_000) + offsetMin(p)) % MIN; return m < 0 ? m + MIN : m; });
  let sx = 0, sy = 0, tot = 0;
  proper.forEach((p, i) => { const w = Math.min(p.ms, 600_000) / 60_000; sx += w * Math.cos(ang(local[i])); sy += w * Math.sin(ang(local[i])); tot += w; });
  const R = tot ? Math.hypot(sx, sy) / tot : NaN;
  const spread = proper.length >= 500 ? 1 - R : NaN;
  const centre = ((Math.atan2(sy, sx) / (2 * Math.PI)) * MIN + MIN) % MIN, dayStart = (centre + MIN / 2) % MIN;
  const days = new Map<number, [number, number, number]>();
  proper.forEach((p, i) => {
    const d = Math.floor((p.t / 60_000 + offsetMin(p) - dayStart) / MIN);
    const e = days.get(d) ?? [0, 0, 0];
    e[0] += Math.cos(ang(local[i])); e[1] += Math.sin(ang(local[i])); e[2]++;
    days.set(d, e);
  });
  let dx = 0, dy = 0, nd = 0;
  for (const [x, y, n] of days.values()) if (n >= 5 && (x || y)) { const r = Math.hypot(x, y); dx += x / r; dy += y / r; nd++; }
  const regular = nd >= 30 ? Math.hypot(dx, dy) / nd : NaN;
  return [
    { key: 'clockSpread', value: spread, receipt: `${proper.length} proper plays on the 24 hour dial; their mean points ${f2(R)} of the way out from the centre (1 would be all at one time)` },
    { key: 'clockRegular', value: regular, receipt: `${nd} days with 5 or more plays; their centres line up ${f2(regular)} of the way (1 would be the same time every day)` },
  ];
}
const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '-');

export const fmtHour = (h: number) => { const x = ((h % 24) + 24) % 24, hh = Math.floor(x), mm = Math.round((x - hh) * 60); return `${String(hh).padStart(2, '0')}:${String(mm % 60).padStart(2, '0')}`; };

/** Share of other people below this value (ties count half). */
export function percentile(value: number, others: number[]): number {
  const o = others.filter(Number.isFinite);
  if (!Number.isFinite(value) || !o.length) return NaN;
  return (o.filter((x) => x < value).length + 0.5 * o.filter((x) => x === value).length) / o.length;
}

/** A card line for one trait: the number, its receipt, where it sits among others, and the tendency if any. */
export function traitLine(t: TraitValue, pctile: number): string | null {
  const d = TRAITS.find((x) => x.key === t.key)!;
  if (d.status === 'dropped' || !Number.isFinite(t.value)) return null;
  const stable = d.status === 'stable';
  const cmp = d.unit === 'hour' ? 'later than' : d.unit === 'seconds' ? 'longer than' : 'more than';
  const where = Number.isFinite(pctile) ? ` That's ${cmp} ${Math.round(100 * pctile)}% of listeners we've measured.` : '';
  const shown = d.unit === 'share' ? pct(t.value) : d.unit === 'hour' ? fmtHour(t.value) : d.unit === 'seconds' ? `${t.value.toFixed(1)} s` : t.value.toFixed(2);
  const flag = stable ? '' : ' (this one moves a lot over time, so read it as now, not always)';
  const tend = d.tendency ? ` Research: ${d.tendency.text}` : '';
  return `${d.label}: ${shown}. ${t.receipt[0].toUpperCase()}${t.receipt.slice(1)}.${where}${flag}${tend}`;
}

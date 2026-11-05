// Listening sessions (a 30 minute gap starts a new one) and the per-play surroundings we read.
import { Play, isSkip, trackKey } from '../parse/spotify.js';

export const SESSION_GAP = 30 * 60_000;
export const WEEK = 7 * 86_400_000;

export interface Session { start: number; end: number; plays: Play[] }

export function sessions(plays: Play[]): Session[] {
  const out: Session[] = [];
  let cur: Session | null = null;
  for (const p of plays) {
    if (!cur || p.t - cur.end > SESSION_GAP) { cur = { start: p.t, end: p.t + p.ms, plays: [] }; out.push(cur); }
    cur.plays.push(p);
    cur.end = Math.max(cur.end, p.t + p.ms);
  }
  return out;
}

// Each country's main IANA time zone (the most populous one for countries with several), so local
// hours follow daylight saving time. 'ZZ' (unknown) and countries not listed fall back to the
// person's usual country. Multi-zone countries (US, CA, MX, BR, AU, RU, ID) are a known limit:
// everyone there reads as the main zone until the app asks for a zone.
export const ZONE: Record<string, string> = {
  GB: 'Europe/London', IE: 'Europe/Dublin', PT: 'Europe/Lisbon', IS: 'Atlantic/Reykjavik',
  FR: 'Europe/Paris', DE: 'Europe/Berlin', IT: 'Europe/Rome', NL: 'Europe/Amsterdam', ES: 'Europe/Madrid', CH: 'Europe/Zurich',
  BE: 'Europe/Brussels', AT: 'Europe/Vienna', SE: 'Europe/Stockholm', NO: 'Europe/Oslo', DK: 'Europe/Copenhagen', PL: 'Europe/Warsaw',
  CZ: 'Europe/Prague', SK: 'Europe/Bratislava', HU: 'Europe/Budapest', RO: 'Europe/Bucharest', BG: 'Europe/Sofia', GR: 'Europe/Athens',
  FI: 'Europe/Helsinki', EE: 'Europe/Tallinn', LV: 'Europe/Riga', LT: 'Europe/Vilnius', UA: 'Europe/Kyiv', HR: 'Europe/Zagreb',
  RS: 'Europe/Belgrade', SI: 'Europe/Ljubljana', LU: 'Europe/Luxembourg', TR: 'Europe/Istanbul', RU: 'Europe/Moscow', CY: 'Asia/Nicosia',
  NG: 'Africa/Lagos', GH: 'Africa/Accra', ZA: 'Africa/Johannesburg', EG: 'Africa/Cairo', KE: 'Africa/Nairobi', MA: 'Africa/Casablanca',
  IL: 'Asia/Jerusalem', SA: 'Asia/Riyadh', AE: 'Asia/Dubai', QA: 'Asia/Qatar', PK: 'Asia/Karachi', IN: 'Asia/Kolkata', BD: 'Asia/Dhaka',
  LK: 'Asia/Colombo', NP: 'Asia/Kathmandu', TH: 'Asia/Bangkok', VN: 'Asia/Ho_Chi_Minh', ID: 'Asia/Jakarta', MY: 'Asia/Kuala_Lumpur',
  SG: 'Asia/Singapore', PH: 'Asia/Manila', CN: 'Asia/Shanghai', HK: 'Asia/Hong_Kong', TW: 'Asia/Taipei', MN: 'Asia/Ulaanbaatar',
  KR: 'Asia/Seoul', JP: 'Asia/Tokyo', AU: 'Australia/Sydney', NZ: 'Pacific/Auckland', PF: 'Pacific/Tahiti',
  US: 'America/New_York', CA: 'America/Toronto', MX: 'America/Mexico_City', GT: 'America/Guatemala', SV: 'America/El_Salvador',
  HN: 'America/Tegucigalpa', CR: 'America/Costa_Rica', PA: 'America/Panama', CU: 'America/Havana', DO: 'America/Santo_Domingo',
  PR: 'America/Puerto_Rico', JM: 'America/Jamaica', CO: 'America/Bogota', VE: 'America/Caracas', EC: 'America/Guayaquil',
  PE: 'America/Lima', BO: 'America/La_Paz', CL: 'America/Santiago', AR: 'America/Argentina/Buenos_Aires', UY: 'America/Montevideo',
  PY: 'America/Asuncion', BR: 'America/Sao_Paulo',
};

const HOUR_MS = 3_600_000;
const offsetCache = new Map<string, Map<number, number>>();
const fmts = new Map<string, Intl.DateTimeFormat>();
/** Minutes east of UTC for a zone at time t (cached per UTC hour; offsets only change on the hour or half hour). */
export function zoneOffset(zone: string, t: number): number {
  const h = Math.floor(t / (HOUR_MS / 2));
  let c = offsetCache.get(zone);
  if (!c) offsetCache.set(zone, (c = new Map()));
  const hit = c.get(h);
  if (hit !== undefined) return hit;
  let f = fmts.get(zone);
  if (!f) fmts.set(zone, (f = new Intl.DateTimeFormat('en-GB', { timeZone: zone, hourCycle: 'h23', year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric' })));
  const at = h * (HOUR_MS / 2), parts = Object.fromEntries(f.formatToParts(at).map((p) => [p.type, p.value]));
  const local = Date.UTC(+parts.year, +parts.month - 1, +parts.day, +parts.hour, +parts.minute);
  const off = Math.round((local - at) / 60_000);
  c.set(h, off);
  return off;
}

/** Returns local-hour and weekday readers for one person (dominant country as fallback), DST-correct. */
export function localClock(plays: Pick<Play, 'country'>[]) {
  const counts = new Map<string, number>();
  for (const p of plays) if (p.country && p.country in ZONE) counts.set(p.country, (counts.get(p.country) ?? 0) + 1);
  const home = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
  const zone = (c: string | null) => (c && c in ZONE ? ZONE[c] : home ? ZONE[home] : 'UTC');
  const local = (p: Pick<Play, 't' | 'country'>) => new Date(p.t + zoneOffset(zone(p.country), p.t) * 60_000);
  return { hour: (p: Pick<Play, 't' | 'country'>) => local(p).getUTCHours(), weekday: (p: Pick<Play, 't' | 'country'>) => local(p).getUTCDay(), zone };
}

/** Device class from Spotify's free-text platform field. */
export function deviceClass(platform: string | null): string {
  const s = (platform ?? '').toLowerCase();
  if (!s) return 'unknown';
  if (/cast|partner|speaker|sonos|tv|tizen|webos|roku|xbox|playstation|ps4|ps5|alexa|echo|car|auto/.test(s)) return 'tv/speaker/car';
  if (/web/.test(s)) return 'web';
  if (/ios|iphone|ipad|android/.test(s)) return 'phone';
  if (/windows|os x|osx|mac|linux/.test(s)) return 'desktop';
  return 'other';
}

const PICK = new Set(['clickrow', 'playbtn', 'backbtn']);

/** How a session was listened to: deliberate picking, skipping, repeating, length. */
export function behaviour(s: Session) {
  const n = s.plays.length, seen = new Set<string>();
  let pick = 0, skip = 0, rep = 0, known = 0;
  for (const p of s.plays) {
    if (p.reasonStart) { known++; if (PICK.has(p.reasonStart)) pick++; }
    if (isSkip(p)) skip++;
    const k = trackKey(p);
    if (seen.has(k)) rep++;
    seen.add(k);
  }
  return { pick: known ? pick / known : 0.5, skip: skip / n, repeat: rep / n, logLen: Math.log(n), minutes: (s.end - s.start) / 60_000 };
}

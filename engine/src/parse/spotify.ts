// Spotify "Extended streaming history" (Streaming_History_Audio_*.json) and the older
// account-data "StreamingHistory_music_*.json" ({endTime, artistName, trackName, msPlayed}).

export interface Play {
  t: number; // start time, ms since epoch (extended `ts` is the END time; we subtract ms played)
  ms: number;
  track: string;
  artist: string;
  album: string | null;
  uri: string | null;
  reasonStart: string | null;
  reasonEnd: string | null;
  shuffle: boolean | null;
  skipped: boolean | null;
  platform: string | null;
  country: string | null;
}

/** A podcast or audiobook episode play: kept only as time, for the rhythm and silence story. */
export interface Episode { t: number; ms: number }

/** A play shorter than this counts as a skip. Spotify's own threshold for a "stream". */
export const SKIP_MS = 30_000;

type Raw = Record<string, unknown>;
const s = (v: unknown) => (typeof v === 'string' && v.length ? v : null);

export function parseSpotify(records: Iterable<Raw>, episodes?: Episode[]): Play[] {
  const out: Play[] = [];
  for (const r of records) {
    if ('master_metadata_track_name' in r || 'ts' in r) {
      if (episodes && (s(r.episode_name) || s(r.episode_show_name))) {
        const end = Date.parse(String(r.ts)), ms = Number(r.ms_played) || 0;
        if (Number.isFinite(end)) episodes.push({ t: end - ms, ms });
        continue;
      }
      const track = s(r.master_metadata_track_name);
      const artist = s(r.master_metadata_album_artist_name);
      if (!track || !artist) continue; // podcasts, videos, missing metadata
      const end = Date.parse(String(r.ts));
      const ms = Number(r.ms_played) || 0;
      if (!Number.isFinite(end)) continue;
      out.push({
        t: end - ms, ms, track, artist,
        album: s(r.master_metadata_album_album_name),
        uri: s(r.spotify_track_uri),
        reasonStart: s(r.reason_start), reasonEnd: s(r.reason_end),
        shuffle: typeof r.shuffle === 'boolean' ? r.shuffle : null,
        skipped: typeof r.skipped === 'boolean' ? r.skipped : null,
        platform: s(r.platform), country: s(r.conn_country),
      });
    } else if ('endTime' in r && 'trackName' in r) {
      const track = s(r.trackName), artist = s(r.artistName);
      if (!track || !artist) continue;
      const end = Date.parse(String(r.endTime).replace(' ', 'T') + 'Z');
      const ms = Number(r.msPlayed) || 0;
      if (!Number.isFinite(end)) continue;
      out.push({ t: end - ms, ms, track, artist, album: null, uri: null, reasonStart: null,
        reasonEnd: null, shuffle: null, skipped: null, platform: null, country: null });
    }
  }
  out.sort((a, b) => a.t - b.t);
  // Exports overlap across files; drop exact duplicates.
  return out.filter((p, i) => i === 0 || p.t !== out[i - 1].t || p.track !== out[i - 1].track);
}

export const trackKey = (p: Play) => p.uri ?? `${p.artist}\u0000${p.track}`;
export const isSkip = (p: Play) => p.ms < SKIP_MS;

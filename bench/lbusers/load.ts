// Loader for the ListenBrainz user bench. Gives each kept user as pseudonym + plays in the
// engine's Play shape, with the fields ListenBrainz does not have set to null.
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import type { Play } from '../../engine/src/parse/spotify.js';

/** One cleaned listen as stored in bench/data/lbusers/plays/<pseudonym>.json.gz. */
export interface LbPlay {
  t: number;                 // start time, ms since epoch (ListenBrainz listened_at)
  durMs: number | null;      // track length when the client sent it; NOT time played
  track: string;
  artist: string;            // the artist credit as submitted
  album: string | null;
  uri: string | null;        // spotify:track:... when the listen came through Spotify
  artistMbids: string[] | null;
  recMbid: string | null;
  client: string | null;     // submission client (listenbrainz = Spotify link or importer, navidrome, web scrobblers...)
  imported: boolean;         // inserted more than 7 days after it was played (history imports)
}

const PLAYS = join(import.meta.dirname, '..', 'data', 'lbusers', 'plays');

/**
 * ListenBrainz only records a listen once half the track or four minutes have played, so every
 * listen counts as a proper play and none is a skip. `ms` is therefore the track length when
 * known (else 180 s) so length filters treat it as played; `skipped`, `reasonStart`, `reasonEnd`,
 * `shuffle`, `platform` and `country` do not exist here and are null.
 */
export function toPlay(p: LbPlay): Play & { artistMbids: string[] | null } {
  return {
    t: p.t, ms: p.durMs && p.durMs > 30_000 ? p.durMs : 180_000, track: p.track, artist: p.artist,
    album: p.album, uri: p.uri, reasonStart: null, reasonEnd: null, shuffle: null, skipped: null,
    platform: null, country: null, artistMbids: p.artistMbids,
  };
}

export function lbUserIds(): string[] {
  if (!existsSync(PLAYS)) return [];
  return readdirSync(PLAYS).filter((f) => f.endsWith('.json.gz') && statSync(join(PLAYS, f)).size > 0)
    .map((f) => f.replace('.json.gz', '')).sort();
}

export function loadLbUser(id: string): LbPlay[] {
  return JSON.parse(gunzipSync(readFileSync(join(PLAYS, id + '.json.gz'))).toString());
}

export function* loadLbUsers(): Generator<{ id: string; plays: Play[] }> {
  for (const id of lbUserIds()) yield { id, plays: loadLbUser(id).map(toPlay) };
}

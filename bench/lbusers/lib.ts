// Shared helpers for the ListenBrainz user bench: a rate-limited client, paths and pseudonyms.
// Data lives in bench/data/lbusers (gitignored). Usernames never leave that folder.
import { createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { execFile } from 'node:child_process';

export const API = 'https://api.listenbrainz.org/1';
export const UA = 'aas-bench/0.1'; // no contact address, by rule
export const DATA = join(import.meta.dirname, '..', 'data', 'lbusers');
// LB_SET=fresh selects the confirmation set (round 2 of the statements lane): listeners found by the
// same walk who are not in the first download. It shares the salt and the candidate list, and keeps
// its pages and plays in raw_fresh/ and plays_fresh/ so the first bench is never touched.
// LB_SET=third selects the identity round's confirmation set (py/docs/identity.md): listeners found by
// the same walk who are in neither of the first two downloads, kept in raw_third/ and plays_third/.
export const SET = process.env.LB_SET === 'fresh' ? 'fresh' : process.env.LB_SET === 'third' ? 'third' : 'main';
const SUFFIX = SET === 'main' ? '' : '_' + SET;
export const RAW_MAIN = join(DATA, 'raw');
export const RAW_FRESH = join(DATA, 'raw_fresh');
export const RAW = join(DATA, 'raw' + SUFFIX);
export const PLAYS = join(DATA, 'plays' + SUFFIX);
export const SUMMARY = join(DATA, 'summary' + SUFFIX + '.json');
for (const d of [DATA, RAW, PLAYS]) mkdirSync(d, { recursive: true });

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// One shared budget for every request in the process, driven by the X-RateLimit headers.
let remaining = 1, resetAt = 0, chain: Promise<void> = Promise.resolve();
async function slot() {
  const prev = chain; let release!: () => void;
  chain = new Promise((r) => (release = r));
  await prev;
  if (remaining <= 1 && Date.now() < resetAt) await sleep(resetAt - Date.now() + 250);
  remaining--;
  release();
}

function curl(url: string): Promise<{ status: number; headers: Record<string, string>; body: string }> {
  return new Promise((resolve, reject) => {
    execFile('curl', ['-s', '--compressed', '--connect-timeout', '15', '--max-time', '90', '-A', UA, '-H', 'Accept: application/json', '-D', '-', url],
      { maxBuffer: 64 * 1024 * 1024 }, (err, stdout) => {
        if (err) return reject(err);
        // header blocks (possibly several, e.g. after a 100 Continue) then the body
        let rest = stdout, status = 0, headers: Record<string, string> = {};
        while (rest.startsWith('HTTP/')) {
          const end = rest.indexOf('\r\n\r\n');
          const block = rest.slice(0, end).split('\r\n');
          status = Number(block[0].split(' ')[1]);
          headers = {};
          for (const line of block.slice(1)) { const k = line.indexOf(':'); headers[line.slice(0, k).toLowerCase()] = line.slice(k + 1).trim(); }
          rest = rest.slice(end + 4);
        }
        resolve({ status, headers, body: rest });
      });
  });
}

export async function get(path: string, tries = 12): Promise<any> {
  for (let i = 0; i < tries; i++) {
    await slot();
    // curl rather than Node's fetch: on this Mac Node's sockets to api.listenbrainz.org time out
    // while curl and Python connect fine (a per-app network rule), so the transport is curl.
    let out: { status: number; headers: Record<string, string>; body: string };
    try { out = await curl(API + path); } catch { await sleep(Math.min(30_000, 3000 * (i + 1))); continue; } // network drops: back off, about 4 min in all
    const rem = Number(out.headers['x-ratelimit-remaining']);
    const resetIn = Number(out.headers['x-ratelimit-reset-in']);
    if (Number.isFinite(rem)) remaining = rem;
    if (Number.isFinite(resetIn)) resetAt = Date.now() + resetIn * 1000;
    if (out.status === 429) { remaining = 0; resetAt = Date.now() + (resetIn || 10) * 1000; continue; }
    if (out.status === 404) return null;
    if (out.status >= 500 || out.status === 0) { await sleep(3000 * (i + 1)); continue; }
    if (out.status >= 400) throw new Error(`${out.status} ${path}`);
    try { return JSON.parse(out.body); } catch { await sleep(Math.min(60_000, 5000 * (i + 1))); continue; } // an HTML error page sent as 200
  }
  throw new Error(`gave up on ${path}`);
}

// Pseudonyms: salted SHA-256 of the username, first 16 hex chars. The salt never leaves DATA.
const saltFile = join(DATA, 'salt');
if (!existsSync(saltFile)) writeFileSync(saltFile, randomBytes(32).toString('hex'));
const salt = readFileSync(saltFile, 'utf8').trim();
export const pseud = (user: string) => 'u' + createHash('sha256').update(salt + '\n' + user).digest('hex').slice(0, 15);

export function readJson<T>(file: string, fallback: T): T {
  return existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : fallback;
}
export const writeJson = (file: string, v: unknown) => writeFileSync(file, JSON.stringify(v));

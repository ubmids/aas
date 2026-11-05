// Step 1: find active public ListenBrainz users by walking the similar-users graph from a few
// well-known public accounts, and keep the ones whose history is long and big enough.
// Resumable: state in bench/data/lbusers/candidates.json (usernames stay in DATA only).
// Usage: npx tsx bench/lbusers/discover.ts [target=600]
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DATA, get, readJson, writeJson } from './lib.js';

// Seed usernames live in bench/data/lbusers/seeds.txt (gitignored), one per line, so no username
// is ever committed. Any few long-standing public accounts work as seeds.
const seedFile = join(DATA, 'seeds.txt');
if (!existsSync(seedFile)) throw new Error('put a few public ListenBrainz usernames in ' + seedFile);
const SEEDS = readFileSync(seedFile, 'utf8').split('\n').map((x) => x.trim()).filter(Boolean);
const MIN_LISTENS = 5_000;          // enough to fit and hold out
const MAX_LISTENS = 2_000_000;      // beyond this the account is almost surely automated or merged
const MIN_SPAN_DAYS = 365;          // at least 12 months between first and latest listen
const ACTIVE_DAYS = 120;            // latest listen within ~4 months of today

type Cand = { user: string; count?: number; first?: number; latest?: number; ok?: boolean; why?: string; expanded?: boolean };
const file = join(DATA, 'candidates.json');
const target = Number(process.argv[2] ?? 600);
const cands: Record<string, Cand> = readJson(file, {});
for (const s of SEEDS) cands[s] ??= { user: s };
const save = () => writeJson(file, cands);
const now = Date.now() / 1000;

async function check(c: Cand) {
  const n = await get(`/user/${encodeURIComponent(c.user)}/listen-count`);
  c.count = n?.payload?.count ?? 0;
  if (c.count! < MIN_LISTENS) return void ((c.ok = false), (c.why = 'too few listens'));
  if (c.count! > MAX_LISTENS) return void ((c.ok = false), (c.why = 'too many listens'));
  const newest = await get(`/user/${encodeURIComponent(c.user)}/listens?count=1`);
  const oldest = await get(`/user/${encodeURIComponent(c.user)}/listens?min_ts=1&count=5`);
  c.latest = newest?.payload?.listens?.[0]?.listened_at;
  const olds = (oldest?.payload?.listens ?? []).map((l: any) => l.listened_at);
  c.first = olds.length ? Math.min(...olds) : undefined;
  if (!c.latest || !c.first) return void ((c.ok = false), (c.why = 'no listens returned'));
  if (now - c.latest > ACTIVE_DAYS * 86400) return void ((c.ok = false), (c.why = 'inactive'));
  if ((c.latest - c.first) / 86400 < MIN_SPAN_DAYS) return void ((c.ok = false), (c.why = 'short history'));
  c.ok = true;
}

const okCount = () => Object.values(cands).filter((c) => c.ok).length;
let round = 0;
while (okCount() < target) {
  const todo = Object.values(cands).filter((c) => c.ok === undefined).slice(0, 300);
  let k = 0; // a few checks in flight; the shared limiter in lib.ts keeps us inside the rate limit
  await Promise.all(Array.from({ length: 6 }, async () => {
    while (k < todo.length) {
      const c = todo[k++];
      try { await check(c); } catch (e) { c.ok = false; c.why = 'error ' + (e as Error).message.slice(0, 60); }
      if (k % 25 === 0) save();
    }
  }));
  save();
  // expand from users not yet expanded; qualified users first, they lead to similar active people
  const frontier = Object.values(cands).filter((c) => !c.expanded && c.ok !== undefined)
    .sort((a, b) => Number(!!b.ok) - Number(!!a.ok)).slice(0, 40);
  if (!frontier.length) break;
  await Promise.all(frontier.map(async (c) => {
    c.expanded = true;
    const sim = await get(`/user/${encodeURIComponent(c.user)}/similar-users`).catch(() => null);
    for (const s of sim?.payload ?? []) cands[s.user_name] ??= { user: s.user_name };
  }));
  save();
  const all = Object.values(cands);
  console.log(`round ${++round}: ${all.length} seen, ${all.filter((c) => c.ok !== undefined).length} checked, ${okCount()} qualify`);
}
console.log(`done: ${okCount()} qualifying users`);

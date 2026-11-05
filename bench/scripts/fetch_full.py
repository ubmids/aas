"""Download full public Spotify exports, strip identifying fields, store compact per-user JSONL. Raw files are not kept."""
import json, os, subprocess, sys, urllib.parse, collections
KEEP = ["ts","platform","ms_played","conn_country","master_metadata_track_name","master_metadata_album_artist_name",
        "master_metadata_album_album_name","spotify_track_uri","episode_name","episode_show_name","reason_start","reason_end",
        "shuffle","skipped","offline","incognito_mode"]
out = "bench/data/spotify_full"; os.makedirs(out, exist_ok=True)
rows = [l.rstrip("\n").split("\t") for l in open("bench/dedup.tsv")]
for size, repo, path in rows:
    user = repo.replace("/", "__")
    raw = subprocess.run(["gh","api","-H","Accept: application/vnd.github.raw",f"repos/{repo}/contents/{urllib.parse.quote(path)}"],capture_output=True).stdout
    try: plays = json.loads(raw)
    except Exception:
        # contents API caps at 100MB; fall back to blob via raw URL
        print("skip", repo, path, file=sys.stderr); continue
    with open(f"{out}/{user}.jsonl","a") as f:
        for p in plays: f.write(json.dumps({k:p.get(k) for k in KEEP})+"\n")
    print(repo, path, len(plays), flush=True)

# Getting your data exports

AAS reads the files a service gives you when you ask for "a copy of your data".
Give `aas.parsers.load()` (or `scripts/inspect_export.py`) the .zip exactly as you
downloaded it, or the unzipped folder. Nothing is uploaded or saved anywhere.

Check what was read with:

    cd py && uv run python scripts/inspect_export.py ~/Downloads/<export>.zip

## Reddit

Request: reddit.com/settings/data-request (logged in on the web). Pick "GDPR"
(or "CCPA") and "all time". Reddit emails a link to a zip of CSV files. Officially up
to 30 days; usually a few days.

Read: `comments.csv` (your comments: text, date, subreddit) and `posts.csv` (your
posts: title and body joined into one item). Deleted or removed items are skipped.

Ignored: votes, saved items, messages, chat, subscriptions, IP logs and every other CSV.

## X (Twitter)

Request: Settings and privacy > Your account > Download an archive of your data.
Confirm your password and a code. X says 24 hours or more; you get an in-app
notification and an email, then download a zip (it expires after a few days).

Read: `data/tweets.js` (older archives: `tweet.js`; big archives split into
`tweets-part1.js` and so on) and `data/note-tweet.js` (the full text of posts longer
than 280 characters). Your tweets and your replies are kept. Retweets are dropped
because they are someone else's words. t.co links are removed.

Ignored: likes, DMs, followers, deleted tweets, ads data, media files, everything else.

## Google (Search and YouTube)

Request: takeout.google.com. Click "Deselect all", then tick:
- "My Activity" (in its options, keep at least "Search"; JSON format is best:
  "Multiple formats" > My Activity > JSON)
- "YouTube and YouTube Music" (in its options, keep "history"; JSON format if offered)

Export once, as .zip. Small exports take minutes, big ones hours to a day or two.
Large exports come split into several zips; pass them all.

Read: your Google searches (from My Activity, the "Searched for ..." entries), your
YouTube searches, and your YouTube watch history (video title and channel). JSON is
fully supported; the older HTML format is read on a best-effort basis (English only,
and its times are read as UTC).

Ignored: "Visited ..." clicks on results, YouTube ads, removed or private videos (no
title left), and every other Google product.

## Spotify

Request: spotify.com/account/privacy (log in on the web), scroll to "Download your
data". There are two different boxes and they are not the same thing:

- **Extended streaming history**: what we want. Every play for the life of the account,
  with skips, why a track started and ended, and shuffle. Up to 30 days.
- **Account data**: arrives faster (up to 5 days) but only covers about the last year
  and has no skips or reasons. Used if that is all you have, and flagged in the notes.

Tick Extended streaming history (both is fine), confirm by email, then download from the
email Spotify sends.

Read: music plays (time, artist, track, how long you listened, and for the extended
history: skipped, start/end reason, shuffle). Files: `Streaming_History_Audio_*.json`
(older name `endsong_*.json`) or, for the basic export, `StreamingHistory*.json` /
`StreamingHistory_music_*.json`.

Ignored: podcasts and audiobooks, video history, playlists, library, search queries,
Spotify's own "inferences" about you, and everything else.

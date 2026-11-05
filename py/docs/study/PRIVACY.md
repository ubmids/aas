# Privacy, data protection and ethics review

Version 0.1, 1 Nov 2026. This is a design and a first draft, not legal advice. Before launch, the
consent text and the data protection notes should be read by whoever does the ethics review
(section 7), and ideally by someone who knows UK GDPR.

## 1. The core idea: the raw data never leaves the participant's computer

A data export is among the most revealing files a person owns: every comment, every search, every
song, with timestamps. The study does not need any of it as such. It needs a few hundred numbers per
person. So the participant runs our extractor **on their own machine**, sees what it produced, and
sends only that. We never receive, store or see a post, a search query, a video title, an artist
name or a channel name.

This is the same approach as the data-donation tool Port (Boeschoten et al. 2023), where data
download packages "can be processed (aggregated or de-identified) locally on the participant's
device. Only the locally processed data are sent to the researcher upon consent of the participant."

### 1.1 Why in the browser, not an app

| how the extractor runs | trust and friction | verdict |
|---|---|---|
| command-line (`uv run ...`) | impossible for most Prolific participants | no |
| downloadable desktop app | an unsigned binary from an unknown developer triggers macOS Gatekeeper and Windows SmartScreen warnings; looks like malware; code signing costs money and still needs trust | no |
| **web page running Python in the browser (Pyodide / WebAssembly)** | nothing to install; files are opened by the browser, not uploaded; the page can be inspected; the Content-Security-Policy can forbid every network request except the final POST | **yes** |

Prolific lets a study declare that it needs a software download; with the browser route we do not
need to, which widens the pool.

## 2. What the derived payload contains

One JSON object per person. Every field is a number, a count, a short fixed label from a closed list,
or a version string. The feature builder will have a test that walks the payload and fails if any
string is not from a whitelist of known labels. Proposed blocks (frozen after the pilot):

**A. Bookkeeping (always).** `extractor_version`, `extractor_hash`, which sources were present, which
export format (e.g. Spotify extended vs basic), floors passed or failed and why (fixed reason codes).

**B. Text (Reddit, X).**
- Volume: number of comments, posts, tweets (bucketed: 1-49, 50-199, 200-999, 1,000+), total words
  (rounded to the nearest 100), span of activity in whole months.
- Reader outputs, exactly as the person would see them: `text-bigfive` P(above median), z, interval
  and validity for O, C, E, A, N; `text-mbti` P(first letter) and bootstrap range for the four letters.
  Also the raw head scores before calibration (one number per trait), so the study can refit the
  calibration.
- Style statistics: mean words per item, share of items that are questions, first-person singular
  rate, first-person plural, second-person, negations, exclamation rate, emoji rate, share of items
  with a URL (URLs themselves removed), mean word length, type-token ratio on a fixed 1,000-word
  window. About 15 numbers.
- Rhythm: share of items in each of 24 UTC hours and 7 weekdays (shares rounded to 0.01), split
  Reddit vs X.
- **Optional, separate consent: common-word rates.** Relative frequency (per 1,000 words, 3 significant
  figures) of a fixed list of about 2,000 words: the words that are both in the top 5,000 of a general
  English frequency list and in the trained readers' vocabulary, minus a published blocklist of
  terms that point at special categories (health conditions, sexuality, religion, ethnicity, political
  parties, trade unions, crime). Only computed when the person has at least 500 words. This is what
  lets the study retrain the text models on ordinary people (PREREG secondary analysis S4).

**C. Google search and YouTube.**
- Volume: number of searches, YouTube searches, YouTube watches (bucketed), activity span in months.
- Rhythm: 24-hour and 7-day shares for searches and for watches, separately; share of days active.
- Query form (computed locally, then the queries are discarded): mean query length in words, share
  phrased as a question, share containing "near me", share in English.
- Viewing form: distinct channels / watches, entropy of the channel distribution, share of watches
  going to the top 10 channels, share of repeat watches of the same video, share of titles that look
  like music videos. **No channel names.**
- Broad topic shares for searches and for video titles from a local keyword lexicon with about 20
  non-sensitive categories (e.g. sport, gaming, cooking, tech, finance, travel, music, film/TV,
  science, education, cars, fashion, parenting, pets, DIY, news in general, comedy, other). There is
  deliberately **no** health, religion, sexuality or politics category; such items fall into "other".

**D. Spotify.**
- The `music` reader's measured traits with their percentiles and reliabilities: new share, old
  friends, season turnover, clock spread, clock regularity, finish share, skip seconds, and the two
  dials. Plus play count (bucketed), span in months, share of plays on shuffle, export type.
- **No artist or track names**, and no per-play rows. (We lose genre: there is no offline genre table,
  and Spotify's audio-features API has been closed to new apps since November 2024.)

Size: roughly 150 numbers without the common-word block, about 2,150 with it.

### 2.1 Why this cannot rebuild a post, and what it still reveals

- No block holds a sequence. Without order, text cannot be reconstructed; at most one learns *which
  common words* someone uses and how often.
- Rare words are where content lives (names, places, usernames, diseases, products). None are
  included: the word list is common words only, filtered again by the blocklist, and URLs, handles and
  numbers are stripped before counting.
- Searches and titles are reduced to lengths, shares and one of 20 topic labels before anything is
  kept.
- We decided against sending text embeddings (MiniLM means). Single-sentence embeddings can be
  inverted back to close paraphrases with published methods (e.g. vec2text, Morris et al. 2023); means
  over many items are much harder to invert but that is not proven, and embeddings lost to the bag of
  words in our own tests anyway (`readers_text.md`: 0.05 to 0.07 AUC worse on MBTI).

**What it still reveals, honestly.** This is *pseudonymised personal data*, not anonymous data:

1. **Linkage to public posts.** Reddit and X posts are public. Someone with a large scrape could
   compute the same common-word rates and posting-hour histogram for every public account and look
   for the nearest match. Word-rate vectors are what stylometry uses for authorship attribution, so a
   determined attacker with the right data might re-identify a participant. That is the main reason
   the common-word block is opt-in and is never published.
2. **Uniqueness.** With 150 numbers, most rows are unique, which matters if the file leaks together
   with other information about a person.
3. **Inference.** The personality estimates are themselves inferences about the person.

So: the payload is safer than a raw export by a very large margin, but it is treated as personal data
throughout (sections 4 and 5).

### 2.2 Compared with uploading raw exports

| | derived payload (this design) | raw exports |
|---|---|---|
| what a breach exposes | numbers; possible re-identification by linkage | every comment, search, video and song with time; third parties' names in replies; often location and device data |
| participant trust and consent rate | much higher (the data-donation literature names privacy concerns as a main reason for drop-out) | lower; would drive the expensive dropout higher |
| special-category exposure | designed out (blocklist, no sensitive topic bins) | certain: search histories routinely contain health, sexuality, religion and politics |
| GDPR burden | moderate | heavy: large-scale processing of data likely to include special categories, so a DPIA, Art. 9 condition, stronger security and probably a UK representative |
| storage | megabytes | tens of gigabytes |

**What we lose analytically**, and how much it hurts:

1. **No new text models from raw text.** We cannot fine-tune a transformer or try new n-grams on
   study people. Mitigation: the reader outputs and raw head scores (always) plus the opt-in common-word
   block cover recalibration and refitting of the current bag-of-words approach. Real loss for
   anything beyond that.
2. **No new features later.** If a better feature is thought of after collection, it cannot be computed.
   Mitigation: freeze the feature list only after the pilot, and include generous counts and
   histograms.
3. **No error analysis by reading.** We cannot look at why a reader failed on a person (sarcasm,
   quoting others, non-English). Mitigation: the pilot participants can be asked, with their consent, to
   look at their own results and tell us.
4. **Extractor bugs are permanent.** A bug discovered later cannot be fixed by re-running on stored
   data. Mitigation: version and hash every payload; pilot thoroughly; unit tests on fixture exports.
5. **No leak auditing by subreddit.** We cannot check after the fact whether r/intj-style communities
   leaked type. Mitigation: the extractor drops personality-typology subreddits locally, as the
   PANDORA loader does, and counts how many items it dropped.

The trade is worth it: the study's questions are about readers and combinations of numbers, and every
one of them can be answered from the payload.

## 3. Sensitive traits: we never infer them

Personality is not a special category under (UK) GDPR Article 9. But text, searches and viewing
can reveal health, sexual orientation, religion, political opinion and ethnicity, and inferences of
those are special-category data. The design rules:

- No reader, feature or topic bin targets a special category. No model in the study predicts one.
- The blocklist removes special-category terms from the common-word block before anything leaves
  the computer.
- The demographic questions do not ask about ethnicity, religion, sexuality, health or politics.
- Neuroticism will be estimated; it is a normal-range personality trait, not a mental health measure,
  and the participant-facing text says so.

## 4. Law: what applies, briefly

- **UK GDPR / EU GDPR.** The owner is a controller based in Nigeria; UK participants are "monitored"
  in the Art. 3(2) sense (their behaviour is analysed), so UK GDPR applies to their data. Lawful basis:
  **consent** (Art. 6(1)(a)), because it is clear, specific and matches what we are asking; it also
  makes withdrawal simple to explain. Purpose: research into predicting personality from digital
  traces, and training the AAS models built from it, **including in a future commercial product**.
  That last part must be in the consent text, not hidden.
- **UK representative (Art. 27).** A non-UK controller monitoring UK people must appoint one unless the
  processing is occasional, does not include large-scale special-category data, and is unlikely to
  be risky. A one-off study of a few hundred people with derived data plausibly meets the exemption,
  but this is a judgment call; services exist that act as representative for roughly a few hundred
  dollars a year. Flagged for the owner (README).
- **DPIA.** Not clearly mandatory at this scale, but the work involves profiling and novel methods, so
  write a short Data Protection Impact Assessment anyway (2 to 3 pages, mostly this document).
- **Transfers.** Participants send their own data directly to the controller; storage is in the EU
  (section 5). The owner will access it from Nigeria. Whether that counts as a restricted transfer
  should be confirmed in the review; using an EU/UK host and keeping analysis files on encrypted
  storage keeps the risk low either way.
- **Nigeria (NDPA 2023)** applies to the owner as a controller in Nigeria; the same consent, minimisation
  and security measures satisfy its core duties. Registration with the NDPC depends on data volume;
  check before launch.
- **US participants.** No federal privacy law covers this; state laws (e.g. CCPA) apply to businesses
  above thresholds the owner does not meet. We apply the GDPR standard to everyone.
- **Prolific's own rules.** Do not ask participants to "disclose information that could directly
  identify them, such as their names, home addresses, or social media accounts." The design never asks
  for handles; the extractor never sends them.

## 5. Storage, retention, withdrawal

- **Where.** A small dedicated server in the EU (e.g. Hetzner Germany or Finland) in the owner's own
  account, used for nothing else. Not any shared or client machine. Full-disk encryption, SSH keys only,
  TLS, firewall to HTTPS only, nightly backups encrypted with `age` to a second EU location.
- **What, separately.** (1) Study data: quiz answers + payloads keyed by random study ID. (2) Link
  table: Prolific ID to study ID, for payment, reminders and withdrawal only. (3) Hash of each
  participant's withdrawal code next to their study row.
- **Who.** The owner only. Any collaborator gets the analysis parquet (no link table) under a written
  data-sharing note.
- **Retention.** Link table deleted 60 days after the final payment. Study data kept for **5 years** after
  the study ends (to allow replication and checking published results), then deleted. Backups roll off
  within 30 days of deletion.
- **Withdrawal.** Any time, no reason needed, no loss of payment for sessions already done. Before the link
  table is deleted: message via Prolific. After: email the withdrawal code (shown at Session 1 and on the
  final page) to the study address; we delete the matching row. Once results are published, a withdrawn
  person's data is removed from all future analyses but cannot be removed from numbers already
  published.
- **Sharing.** No raw sharing. A public release, if any, would contain only quiz scores, reader outputs
  and coarse aggregates, never the common-word block or rhythm histograms (linkage risk, section 2.1).

## 6. Consent text (draft, plain English)

> **What is this study?** We are building a computer model that guesses someone's personality from
> what they do online: what they write, search for, watch and listen to. To find out if it works, we need
> people who are willing to take a personality questionnaire and also share some summary numbers about
> their online activity. The study is run by [owner name], an independent researcher. It is not run by a
> university.
>
> **What you will do.** Session 1 (about 25 minutes): a 50-question personality questionnaire, a short
> 32-question type questionnaire, a few questions about you, and step-by-step help to ask Reddit or X,
> Google, and Spotify for a copy of your data. Those companies email you a download link, usually within
> a few days but sometimes up to 30 days. Session 2 (about 25 minutes): you open our study page, give it
> the downloaded files, check what it will send, and press Send. Some people will also be invited to
> retake the questionnaire two weeks later (5 minutes).
>
> **What stays on your computer.** Your files are opened by your own web browser and are never uploaded.
> We never receive your posts, your searches, the videos you watched, the songs or artists you played, or
> your usernames.
>
> **What we receive.** Numbers worked out on your computer: for example how many words you wrote, what
> time of day you are usually active, what share of your YouTube viewing is sport or music, how often you
> skip songs, and our model's personality guesses. You will see all of it before you press Send, and you
> can leave any service out. There is an optional extra: how often you use about 2,000 common English
> words (like "the", "really", "weekend"). It helps us improve the writing model. You can say no and still
> take part.
>
> **What we will not do.** We will not try to work out your health, sexuality, religion, politics or
> ethnicity, and the study does not collect anything aimed at them. We will not ask for your name or your
> account names. We will not sell your data.
>
> **How it will be used.** To test and improve personality models, to write up the results (only
> group-level numbers are published), and to build the models behind a future app or product made by the
> researcher. Your individual data will not be published.
>
> **Risks.** The numbers we receive cannot be turned back into your posts. Because your public posts exist
> online, someone with special tools and a large copy of Reddit or X might in theory match the optional
> word numbers back to your account. That is why we keep them private, encrypted, and never publish them.
> The questionnaire asks about your feelings and habits; you can skip any question you prefer not to answer.
>
> **Payment.** [amounts] through Prolific, paid within 7 days of each session. If your exports turn out to
> be too small for the study, you are still paid for Session 2.
>
> **Your choices.** Taking part is voluntary. You can stop at any time. You can ask us to delete your data
> at any time without giving a reason: message us through Prolific, or email [study address] with your
> withdrawal code: **[code]**. Please save it. After results are published we can remove your data from
> all future work, but not from numbers already published.
>
> **Where your data is kept.** On an encrypted server in the European Union, for 5 years after the study
> ends, then deleted. The link between your Prolific ID and your answers is deleted 60 days after your last
> payment. Only the researcher has access.
>
> **Your rights.** Under UK data protection law you can ask to see, correct or delete your data, and you can
> complain to the Information Commissioner's Office (ico.org.uk). This study was reviewed by [ethics
> body, reference number].
>
> Tick each box you agree to:
> - [ ] I am 18 or over and I have read this page.
> - [ ] I agree to take the questionnaires and to request my data exports.
> - [ ] I agree to send the summary numbers from my exports in Session 2 (I will see them first).
> - [ ] (Optional) I agree to send the common-word numbers.
> - [ ] (Optional) I agree to be invited to retake the questionnaire in two weeks.

## 7. Ethics review for an unaffiliated researcher

There is no legal requirement in the US or UK for a privately funded, non-medical, minimal-risk online
survey to get ethics approval. It is still needed in practice: Prolific's guidance says studies on the
platform are ethically reviewed by an IRB or equivalent committee, and journals require an approval
statement for research with people.

Options:

| route | what it is | cost | time | fit |
|---|---|---|---|---|
| **A. Commercial (independent) IRB, US** | Advarra, WCG, BRANY, Pearl IRB, Sterling IRB and similar accept unaffiliated investigators. This study plausibly qualifies as **exempt** (45 CFR 46.104(d)(2): survey procedures, with protections for identifiable data) or limited review | Exempt determinations on published fee schedules run about $500 to $1,300 (e.g. institutional external-review schedules list $500, $1,000 and $1,315); commercial IRBs quote individually, likely $1,000 to $3,000 for initial review plus amendment fees. Get two quotes | 1 to 4 weeks | Works for a commercial purpose; accepted by Prolific and most journals |
| B. Academic collaborator | A co-investigator at a US/UK university submits to their ethics committee | Free in money; costs time and shared authorship | 1 to 3 months | Strongest credibility; may also unlock Prolific's academic fee (33.3% instead of 42.857%) if the account is institutional. **But** universities may claim the data or restrict commercial use, which conflicts with training a product |
| C. Non-university research ethics committees elsewhere | e.g. paid review by private committees (Australian private HRECs quote A$3,000 to 6,000) | high | weeks | Only if A falls through |
| D. No review, public protocol only | Pre-register on OSF, publish this document | 0 | 0 | Legal, but Prolific expects review and journals will refuse. Not recommended |

**Recommendation: route A.** Ask two commercial IRBs for an exempt-determination quote using PROTOCOL.md,
this document and the consent text. Budget $1,500 to $3,000 including one amendment (there will be one
after the pilot). Do it before the pilot, because the pilot recruits real people. If the owner wants
academic publication more than commercial freedom, route B is better, but settle data ownership in writing
before anyone submits anything.

# Recruitment and budget

Version 0.1, 1 Nov 2026. All prices in USD, checked on 1 Nov 2026 (sources at the end). No account was
created and nothing was spent.

## 1. Prolific: the numbers that drive cost

- **Participant pay.** Recommended minimum **$12.00 / GBP 9.00 per hour**; absolute minimum allowed
  **$8.00 / GBP 6.00 per hour**. We plan at the recommended rate or above: this study asks for real
  effort (export requests, a second session), and underpaying it would raise dropout, which is the
  expensive failure here.
- **Service fee.** **42.857%** of participant payments for corporate customers, **33.3%** for academic and
  non-profit customers (the academic rate needs an institutional email). The owner is an independent
  developer, so we budget the corporate rate. Bonuses carry the same fee.
- **VAT** is charged on the fee only, not on participant pay, and depends on the customer's location. A
  Nigeria-based individual may or may not be charged UK VAT; we show totals with and without 20% VAT on
  the fee.
- Prescreeners (country, age, language, social media use) cost nothing extra.
- Prolific allows multi-part studies (allowlist + follow-up study) and publishes retention of 90% over 30
  days (300 people), 82% over 4 weekly waves (2,500 people) and 66% over 60 daily waves. Those are for
  ordinary survey waves. Ours has a much harder step between waves (request exports, wait, download, run a
  page), so we assume far lower return (section 3).

## 2. Session lengths and pay

| step | content | time (to confirm in pilot) | pay | rate |
|---|---|---|---|---|
| Screener | accounts used, amount of writing/listening, laptop access, willingness | 3 min | $0.60 | $12/h |
| Session 1 | consent, IPIP-50, OEJTS, demographics, guided export requests (2 to 4 services) | 25 min | $6.00 | $14.40/h |
| Session 2 | download exports, open the extractor page, drop zips, review, send | 25 min | $6.00 + $8.00 completion bonus = $14.00 | $33.60/h effective |
| Session 3 (optional) | late Spotify extended history, Spotify only | 10 min | $3.00 | $18/h |
| Retest (random subset) | IPIP-50 at day 14 | 5 min | $2.00 | $24/h |

Why Session 2 pays well above the minimum: the work is mostly outside the clock (logging in to four
services, confirming emails, waiting, finding downloads), and Session 2 completion is the number the whole
budget depends on. A $8 bonus on completion is the cheapest lever we have against the main risk.
People whose exports turn out too small are still paid for Session 2.

Export timing (from `docs/exports.md`): Reddit officially up to 30 days, usually a few days; X 24 hours or
more; Google Takeout minutes to a day or two for a Search + YouTube-only export; Spotify extended history up
to 30 days (basic account data up to 5 days). Session 2 opens as soon as the person has writing + one other
source and stays open until day 35.

## 3. Funnel assumptions

| stage | assumption | why | pessimistic |
|---|---|---|---|
| screened -> eligible and willing | 25% | needs Reddit/X writing + Google or Spotify + a laptop + willingness to request exports. Prolific's free social media prescreeners should raise this; the pilot measures it | 15% |
| invited -> finishes Session 1 | 90% | ordinary survey | 85% |
| Session 1 -> finishes Session 2 | **50%** | data-donation studies report high, non-random drop-out at the request/download/upload steps; export waits of up to 30 days; we counter with reminders, the bonus, and opening Session 2 early | **35%** (optimistic 65%) |
| Session 2 -> usable (floors met, checks passed) | 80% | some exports will be smaller than claimed; attention and speed exclusions | 70% |
| Session 2 completers who come back for Session 3 | 40% | only those with a late Spotify export | |

Combined: about **2.5 Session-1 completers per usable person** (central) and about 3.6 (pessimistic).

## 4. Cost per scenario

n = usable participants in the analysis. Retest subset: 30 / 60 / 100. Totals = participant pay + 42.857%
fee. "With VAT" adds 20% VAT on the fee.

| scenario | n | Session-2 return | screened | Session 1 | Session 2 paid | participant pay | fee | **total** | with VAT |
|---|---|---|---|---|---|---|---|---|---|
| Pilot | 10 | 50% | 116 | 26 | 13 | $426 | $182 | **$608** | $644 |
| Minimal | 60 | 65% | 516 | 116 | 75 | $2,206 | $945 | $3,151 | $3,340 |
| Minimal | 60 | **50%** | 667 | 150 | 75 | $2,500 | $1,072 | **$3,572** | $3,786 |
| Minimal | 60 | 35% | 956 | 215 | 75 | $3,064 | $1,313 | $4,377 | $4,639 |
| Main | 250 | 65% | 2,143 | 482 | 313 | $9,058 | $3,882 | $12,940 | $13,716 |
| Main | 250 | **50%** | 2,783 | 626 | 313 | $10,306 | $4,417 | **$14,723** | $15,606 |
| Main | 250 | 35% | 3,978 | 895 | 313 | $12,637 | $5,416 | $18,053 | $19,136 |
| Ideal | 600 | 65% | 5,129 | 1,154 | 750 | $21,601 | $9,258 | $30,859 | $32,711 |
| Ideal | 600 | **50%** | 6,667 | 1,500 | 750 | $24,600 | $10,543 | **$35,143** | $37,252 |
| Ideal | 600 | 35% | 9,525 | 2,143 | 750 | $30,173 | $12,931 | $43,104 | $45,690 |

Per usable person, all-in (central): about **$60 (n = 60), $59 (n = 250), $59 (n = 600)**. The cost scales
almost linearly; there is no volume discount.

Fixed costs on top of any scenario:

| item | estimate |
|---|---|
| Ethics review (commercial IRB, exempt determination + one amendment) | $1,500 to $3,000 |
| EU server for the study page and storage, 6 months | about $50 |
| Domain for the study page (optional; a subdomain of an existing one works) | $0 to $15 |
| UK GDPR representative, if the owner decides it is needed | a few hundred dollars per year |
| Owner's time to build the extractor page, study pages and server | not costed; the largest real cost |

**Bottom line (central case, corporate fee, no VAT, ethics at $2,000):**

| scenario | all-in |
|---|---|
| Pilot (10) | about $2,600 including ethics (the review is needed before the pilot) |
| Minimal (60) | about $3,600 more after the pilot, so about **$6,200** for pilot + minimal |
| Main (250) | about $14,700 more, so about **$17,300** for pilot + main |
| Ideal (600) | about $35,100 more, so about **$37,800** for pilot + ideal |

Ways to cut cost, roughly in order of size:

1. **Academic co-investigator** (fee 33.3% instead of 42.857%): saves about 7% of the total. See PRIVACY.md
   section 7 for the catch (data ownership).
2. **Shorter Session 1**: drop OEJTS (4 min) and keep pay at $5.
3. **Tighter screener** using Prolific's free prescreeners for Reddit/Twitter users so fewer paid screeners
   are wasted; each 10 points of eligibility saves a little (screeners are cheap; Session 1 non-returners are
   the real cost).
4. **Raise Session 2 return.** Going from 50% to 65% saves about 12% at every scale. Worth more than any fee
   saving; that is why the bonus is there.

## 5. Alternatives to Prolific

| channel | cost | quality and fit | verdict |
|---|---|---|---|
| **CloudResearch Connect** | Service fee 40% standard, 25% academic (since 1 Jan 2025); participant minimum $7.50/h, $10/h suggested; new accounts get 0% fee for the first 10 days | US-heavy pool, good quality, similar tooling. Fewer UK participants | Real alternative; the 10-day fee-free window could cover the pilot. Compare quotes |
| **Reddit communities** (r/SampleSize, interest subreddits) | free to post; pay via gift cards (e.g. Tremendous) | Reddit users are exactly the writers we need. But: self-selected, no identity or country verification, payment handling from Nigeria to US/UK people is awkward, some subreddits ban paid recruitment | Supplementary only, flagged as a separate stratum in analysis |
| **Own network / social media** | free or small gift cards | Highly non-random, often not US/UK, friends know the hypotheses | Good for the pilot's technical testing, not for confirmatory data |
| **Paid ads (Meta, Reddit ads)** | $2 to $10+ per click-through to a long study, unpredictable | Reaches people who never do surveys; very low conversion for a two-session task | Not now |
| **MTurk** | 20% to 40% fee | Data quality problems well documented | No |

Recommendation: Prolific for the main study (UK and US both, verified, built-in follow-up studies);
consider Connect's fee-free window for the pilot if the owner opens an account there anyway.

## 6. Pilot plan: 10 people, before anything else

Purpose: find out whether the instrument works on real exports on ordinary laptops, and replace every
assumption in sections 2 and 3 with a measured number. Not a test of any hypothesis.

Who: 10 usable participants from Prolific (US/UK, same screener), plus 2 to 3 people from the owner's network
on screen-share purely for usability (their data are deleted, not used). Requires ethics approval first.

What to measure, with pass/fail thresholds decided now:

| question | measure | threshold to go ahead |
|---|---|---|
| Do people finish? | Session 1 -> Session 2 return | at least 40% (else redesign, e.g. bigger bonus, more reminders) |
| How long do exports take? | days from request to arrival, per service | median under 10 days for writing + Google |
| Does the extractor run? | share of Session 2 attempts that produce a valid payload, by browser and OS; peak memory; time | at least 90% success; under 5 minutes processing for a 1 GB Takeout |
| Are the floors right? | share of eligible-by-screener people who pass the data floors at Session 2; word counts, play counts, search counts | at least 70% |
| Session lengths | median and 90th percentile minutes for each session | adjust pay to keep at least $12/h at the median |
| Do people understand what leaves their computer? | the privacy comprehension question; free-text feedback | at least 80% right first time |
| Is the payload sane? | every feature within plausible range; no free text anywhere (automated check) | zero free-text fields; fix and re-version otherwise |
| Do people feel OK about it? | 1 to 5 comfort rating, open comment | median 4 or higher |

After the pilot: freeze the feature list and extractor version, update this budget with measured numbers,
file the ethics amendment, then post the pre-registration (PREREG.md) on OSF before the main study opens.
Pilot data are never part of confirmatory analyses.

Pilot cost: about $608 to $802 in participant pay and fees (table above), plus the ethics review that must
come first.

## 7. Timeline (main study, after a successful pilot)

| week | |
|---|---|
| 0 | OSF pre-registration posted; screener live |
| 0 to 4 | Session 1 in batches of about 100 per week (keeps the reminder load manageable and lets us stop if return collapses) |
| 1 to 9 | Sessions 2 and 3 open on a rolling basis (35-day window from each person's Session 1) |
| 2 to 6 | retest invitations at day 14 |
| 10 | close; delete link table at day 60 after last payment |

## Sources

- Prolific pricing: https://researcher-help.prolific.com/en/articles/445239-what-is-your-pricing (fees 42.857% corporate,
  33.3% academic; pay $12 / GBP 9 recommended, $8 / GBP 6 minimum; VAT on fees only; read 1 Nov 2026)
- Prolific attention checks: https://researcher-help.prolific.com/en/articles/445260-failed-attention-checks
- Prolific ethics and data rules: https://researcher-help.prolific.com/en/articles/449628-irb-applications-and-ethics-approval
- Prolific retention figures: https://www.prolific.com/longitudinal-studies
- CloudResearch Connect pricing: https://connect-researcher-help.cloudresearch.com/hc/en-us/articles/5046181555732-Project-Cost
  and https://www.cloudresearch.com/cloudresearch-connect-faqs
- Data-donation drop-out: Welbers et al. 2024, "Anything you would like to share: Evaluating a data donation application in a
  survey and field study", Computational Communication Research (https://aup-online.com/content/journals/10.5117/CCR2024.2.5.WELB);
  Keusch, Pankowska, Cernat and Bach 2024, "Do you have two minutes to talk about your data?", Field Methods 36(4)
  (doi 10.1177/1525822X231225907). Neither gave us a single reusable completion rate, hence the pilot.
- Local processing tool: Boeschoten et al. 2023, Port (https://github.com/eyra/port); described in "Best practices for studies
  using digital data donation" (https://pmc.ncbi.nlm.nih.gov/articles/PMC11971172/)

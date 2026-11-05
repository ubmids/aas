# Zero-shot LLM ratings of the Big Five from text, across kinds of writing

Status (updated 2026-11-05): the pre-registered run with Claude Haiku 4.5 is done. It rated the same 150 people
in each of Essays, PAN 2015 and the vlogs (450 ratings, none missing) on 2026-11-01, at a dry-run estimate of
about $1.13. On the two held-out domains the LLM's mean AUC is 0.618 against 0.558 for bag of words, delta +0.060
[0.025, 0.093], so the verdict is REPLACE; the combined arm is +0.050 [0.031, 0.069]. Per dataset: PAN 2015
0.557 against 0.508, vlogs 0.679 against 0.607, Essays (in domain for bag of words, outside the rule) 0.620 against
0.605. Numbers: `llm_crossdomain_results_anthropic_claude-haiku-4-5-20251001.json` and `.csv`. The verdict is about
the idea, since an API model can't go in the product (Privacy, below), and contamination is not ruled out (last
section). The local small-model run stopped on the disk rule with no verdict. The design below is unchanged from
before the run. Code: `src/aas/readers/llm_rater.py`, `scripts/llm_crossdomain.py`, tests
`tests/test_llm_rater.py` (mocked, no network).

## The question

Our trained bag-of-words model learned from 2,467 student essays and barely transfers to other kinds of writing
(`text_crossdomain.md`: held-out AUC 0.54 on PAN 2015 tweets, 0.59 on YouTube vlog transcripts, openness at
chance). A large language model asked to rate a person from their text has never been trained on any of our
datasets, so it has no "student essay" habits to lose. Does it travel across kinds of writing better?

## Decision rule (pre-registered 2026-11-01, written before any rating was obtained)

- **People.** Essays: the 494 people in the saved Essays model's held-out test split (seed 0). PAN 2015 (294) and
  vlogs (404): every person, because the bag-of-words model never saw any of them. A fixed random sample of
  **150 per dataset** (seed 20261101) is rated. The same people are used for every arm.
- **Same input.** Each person's text is leak-stripped (`readers.text.clean`, the trained readers' stripper) and cut
  to an estimated 1,500 tokens (~1,100 words): an essay or vlog keeps its first N words; a set of tweets becomes a
  seeded random sample of whole tweets, kept in order. The bag-of-words model scores **the same cut text** (primary)
  and, for reference, the full text.
- **Arms.** `llm` (the model's 1-5 rating), `bow` (bag of words on the same text), `combined`
  (z(llm) + z(bow), z-scored within dataset from the scores alone, no labels, nothing fitted), `bow_full`.
  For Essays, bag-of-words is a model refitted on the 80% dev split with the saved card's settings, because the
  saved artifact was finally refit on everyone (its Essays numbers on these people would be in-sample).
- **Primary metric.** AUC against each dataset's own median split (as in `text_crossdomain.md`), mean over the five
  traits, averaged over **PAN 2015 and vlogs** (the bag-of-words model's held-out domains), each weighted equally.
- **Contrast and CI.** delta = LLM mean minus bag-of-words mean on the same people; 95% CI by paired bootstrap
  (2,000 resamples of people within each dataset).
- **Rule.**
  - **REPLACE**: delta(llm − bow) ≥ 0.03 and its CI excludes 0.
  - **AUGMENT**: otherwise, if delta(combined − bow) ≥ 0.03 and its CI excludes 0.
  - **KEEP**: otherwise. Bag of words stays and the LLM is not adopted.
  - A REPLACE or AUGMENT verdict from an *API* model does not put an API in the product (see Privacy). It means "worth testing the
    same prompt on a model that can run on the participant's own machine".
- **Reported but not in the rule.** Essays (in-domain for bag-of-words), Pearson r against the continuous scores
  (PAN, vlogs), per-trait AUCs with CI, and a 1,000-shuffle null for every number (the `text_crossdomain` scorer).
- **Missing ratings.** A refusal, or an invalid reply twice in a row, makes a person missing. They are dropped from
  every arm and counted in the results.
- **One prompt, one model per run.** The prompt is `b5-ipip50-v1` (below). Changing it after seeing results makes a
  new, post-hoc version that is labelled as such.

**How precise is n = 150?** In a simulation (bag of words r ≈ 0.10, LLM errors correlated 0.2 with it), the
contrast's standard error is about 0.019, so the CI half-width is ±0.037. "CI excludes 0" therefore needs an observed
delta of about 0.04. A true delta of 0.067 (LLM r ≈ 0.25, the literature's range) would be detected about 94% of
the time. A true delta of 0.045 would be detected about 65% of the time. Rating every PAN and vlog person
(698 instead of 300) narrows the CI to about ±0.025 for about $1 more on Haiku 4.5 (table below).

## The prompt (`b5-ipip50-v1`)

System: "You are an experienced personality psychologist. You rate people's Big Five personality traits from samples
of their own writing or speech, as a careful observer would. You always answer with a single JSON object and nothing else."

User: the person's text inside `<text>` tags, after an instruction to rate each trait as their **IPIP-50** score
would come out (mean of ten 1-5 items, so 3.0 is the middle), with a one-line description of each trait. Neuroticism
is stated as "high = more neurotic" to match our N, and the model is told to stay near the middle when the text says
little. Each trait's answer is a score from 1.0 to 5.0 with one decimal and a one-line reason, all as JSON. The
full text is in `llm_rater.py` (`SYSTEM`, `INSTRUCTIONS`), and `--dry-run` prints a filled example.

Safeguards in code:

- **No labels go out.** The prompt is built from the template and the person's text only, and no function takes a
  label. The text goes through the same leak stripper as the trained readers (MBTI types, typology words, Big Five and
  questionnaire words, URLs). `assert_no_labels` refuses to build a prompt if anything label-like survives, and the
  tests check that the template itself contains no MBTI type.
- **Parsing.** The reply must be one JSON object with exactly the five traits, each score a number from 1 to 5,
  each reason non-empty. An invalid reply is retried once. A refusal is recorded and **not** sent to another
  model, because a silent fallback would change which model is being measured.
- **Cache.** Answers are stored as JSON files under `data/llm_cache/` (gitignored). Each file is keyed
  by the sha256 of the prompt version, provider, model, request settings, max_tokens and full prompt. A re-run costs
  nothing and gives identical numbers. Network or auth errors are never cached.
- **Spend switch.** The Anthropic provider will not make a call unless `AAS_LLM_ALLOW_SPEND=1` is set.
- **Local means local.** The local provider takes only `localhost`, `127.0.0.1` or `::1` URLs, so it can't point at a
  remote GPU box.

## Providers and settings

| provider | env | model | settings |
|---|---|---|---|
| Anthropic | `AAS_LLM_PROVIDER=anthropic` (default), `ANTHROPIC_API_KEY`, `AAS_LLM_ALLOW_SPEND=1`, `AAS_LLM_MODEL` | `claude-haiku-4-5-20251001` (default), `claude-sonnet-5-5`, `claude-haiku-5-5` | Haiku 4.5: no thinking, max_tokens 600. Sonnet 5.5: `thinking: between_tools` (up-front thinking off; it rejects `disabled`), effort low, max_tokens 600. Haiku 5.5: adaptive thinking at effort low, max_tokens 2,000 (thinking counts against it) |
| local | `AAS_LLM_PROVIDER=local`, `AAS_LLM_LOCAL_URL` (e.g. `http://127.0.0.1:11434/v1` for Ollama, `http://127.0.0.1:8080/v1` for llama.cpp `llama-server`), `AAS_LLM_LOCAL_MODEL` | whatever the local server serves | OpenAI-compatible `/chat/completions`, temperature 0 |

The installed SDK (`anthropic` 2.3.0) has no `temperature` argument, so Anthropic answers can vary from call to call.
The cache freezes each answer once it's obtained. No local model weights are downloaded by this work (the disk is
nearly full). The local path is supported but untested against a live server.

**Model note for the owner.** The requested cheap default is Claude Haiku 4.5. Anthropic's current small model,
**Claude Haiku 5.5**, lists at about a tenth of the price ($0.10 / $0.50 per million tokens against $1 / $5). It is
configured too, so it's worth considering as the default.

## Dry run: estimated cost (no API call made)

Token method: Anthropic publishes no offline tokenizer, and OpenAI's tokenizer undercounts Claude, so tokens are
estimated as max(characters / 4, words × 1.33) per request, plus 15 for message overhead. For the newer tokenizer
used by Claude 4.7-and-later models (Haiku 5.5, Sonnet 5.5), that is multiplied by 1.30, because Anthropic's pricing
page says it "produces approximately 30% more tokens for the same text". Output is assumed to be 250 tokens per
reply. The upper bound is every reply hitting max_tokens, and it doesn't include the one retry on an invalid reply.
For exact counts, use the free `count_tokens` endpoint, which needs an API key and was not called. Prices are the list prices on
[platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing),
read 2026-11-01. The Batch API halves them.

Planned sample (150 per dataset = 450 requests; median words sent: Essays 637, PAN 885, vlogs 495; text was cut for
3% of Essays, 35% of PAN and 13% of vlogs):

| model | price in / out per MTok | input tokens (est.) | expected | upper bound | expected via Batch API |
|---|---|---|---|---|---|
| Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | $1 / $5 | 572k | **$1.13** | $1.92 | $0.57 |
| Claude Haiku 5.5 | $0.10 / $0.50 | 742k | $0.13 | $0.52 | $0.07 |
| Claude Sonnet 5.5 | $2 / $10 | 742k | $2.61 | $4.18 | $1.30 |
| local model on this laptop | n/a | 572k | $0 | $0 | n/a |

All test people (494 Essays + 294 PAN + 404 vlogs = 1,192 requests):

| model | input tokens (est.) | expected | upper bound | via Batch API |
|---|---|---|---|---|
| Claude Haiku 4.5 | 1.46M | **$2.95** | $5.04 | $1.47 |
| Claude Haiku 5.5 | 1.89M | $0.34 | $1.38 | $0.17 |
| Claude Sonnet 5.5 | 1.89M | $6.76 | $10.93 | $3.38 |

Files: `llm_cost_sample.csv`, `llm_cost_full.csv`. The script makes synchronous calls. Batch submission isn't
built yet; it would halve the cost if wanted.

## What the literature says (checked 2026-11-01)

- **Peters & Matz (2024)**, "Large language models can infer psychological dispositions of social media users",
  *PNAS Nexus* 3(6) pgae231, [doi:10.1093/pnasnexus/pgae231](https://doi.org/10.1093/pnasnexus/pgae231).
  - **Setup:** GPT-3.5 and GPT-4, zero-shot, on 1,000 myPersonality Facebook users with up to 200 status updates
    each, against the 100-item IPIP self-report.
  - **Accuracy:** mean r = 0.29 (GPT-4 0.31, GPT-3.5 0.27), range 0.22-0.33 across traits. Openness is highest and
    conscientiousness lowest.
  - **Amount of text:** accuracy plateaus at about 60-80 posts.
  - **Bias:** errors are larger for men on some traits.
- **Ganesan, Lal, Nilsson & Schwartz (2023)**, "Systematic Evaluation of GPT-3 for Zero-Shot Personality
  Estimation", WASSA 2023, [aclanthology.org/2023.wassa-1.34](https://aclanthology.org/2023.wassa-1.34).
  - **Setup:** 142 Facebook users with 20 posts each, self-reported Big Five, scored as high/low.
  - **Accuracy:** zero-shot GPT-3 gets macro-F1 0.40, or 0.45 with the questionnaire items in the prompt. The
    supervised lexicon model gets 0.52 and the majority-class baseline 0.38.
  - **Verdict:** GPT-3 **lost** to the supervised model and was near chance on a three-way split.
- **Rao, Leung & Miao (2023)**, "Can ChatGPT Assess Human Personalities? A General Evaluation Framework",
  [arXiv:2303.01248](https://arxiv.org/abs/2303.01248). **Not evidence of accuracy.** It has ChatGPT answer MBTI
  questionnaires *about* generic groups of people and measures consistency and fairness. It never compares
  against any real person's self-report, so it shouldn't be cited for "r 0.2-0.4".
- **The datasets used here:**
  - **Ji et al. (2023)**, "Is ChatGPT a Good Personality Recognizer?", [arXiv:2307.03952](https://arxiv.org/abs/2307.03952), preprint.
    - **Essays:** ChatGPT zero-shot gets 57.4% accuracy (chain-of-thought 57.8%), against 57.3% for fine-tuned RoBERTa.
    - **PAN 2015 English:** 57.3% (chain-of-thought 60.7%) against 55.3% for RoBERTa, on only about 30 test users.
  - **Yang et al. (2023)**, "PsyCoT", *Findings of EMNLP 2023*, [arXiv:2310.20256](https://arxiv.org/abs/2310.20256).
    - **Setup:** gpt-3.5 on 200 Essays test texts.
    - **Results:** plain zero-shot gets accuracy 58.7 and macro-F1 54.2. Fine-tuned RoBERTa gets 58.4 and 57.6. The
      questionnaire-guided prompt (PsyCoT) gets 59.6 and 58.4.
- **Observer-rated video transcripts.** Amin, Cambria & Schuller (2023), *IEEE Intelligent Systems*,
  [arXiv:2303.03186](https://arxiv.org/abs/2303.03186), used First Impressions transcripts (observer labels, not
  Essays). ChatGPT scored 45-57% accuracy per trait, below RoBERTa (62-67%) on every trait.

**Summary.** Zero-shot LLMs reach r ≈ 0.3 against self-report when they read a lot of social-media text (Peters &
Matz). On the benchmark datasets used here, they land at about the same near-chance level as fine-tuned models
(Essays about 57-59% accuracy). No published study we found tests what we're asking: whether an LLM degrades
*less* than a trained model when the kind of writing changes. That makes the experiment worth doing, but the prior
for a large win is modest. Our own Essays-only model already reaches 0.59 AUC on vlogs and 0.54 on PAN.

**Contamination caveat.** Essays, PAN 2015 and the vlog transcripts are public and may be in an LLM's training data.
The labels sit in separate columns and are unlikely to be memorised per person, but a good LLM score on these
datasets is still slightly less convincing than one on unseen text.

## Privacy and terms

**The research datasets.** None of the three datasets' terms mention sending text to a third-party service for
processing. Scoring text through an API is processing, not redistribution, and no derived dataset is published.

| dataset | terms found | sending for scoring |
|---|---|---|
| Essays (Pennebaker & King 1999, SenticNet mirror) | no data licence stated; the repo's MIT licence covers code only | not addressed. Research use is how it has always been used, including in published ChatGPT studies (Ji 2023, PsyCoT 2023). The authors were students who consented to a 1990s study, not to AI processing. The text is anonymous |
| PAN 2015 (Zenodo 3745945) | "open access", no licence; public tweets with @mentions already anonymised | not addressed. Used with ChatGPT in Ji et al. 2023 |
| YouTube vlogs (WCPR14, Hugging Face) | CC BY-NC-ND 4.0 | non-commercial research is allowed. Sending a transcript to be scored is neither sharing nor publishing an adaptation, so it looks allowed. Our ratings are our results and are not redistributed as a dataset. Still evaluation-only, as before |

Anthropic's API terms: inputs and outputs are not used for training by default
([privacy.claude.com, data use](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training)).
They are deleted within 30 days, except where flagged for usage-policy enforcement
([retention](https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data)).
**Verdict:** for these public research datasets, the API run looks acceptable. It's still the owner's decision,
because none of the terms address it explicitly.

**Real users: this conflicts with the study's design.** `SPEC.md` says "Nothing leaves the device", and
`docs/study/PRIVACY.md` section 1 says the raw data never leaves the participant's computer and we never
receive a post. An LLM rater that sends a participant's text to Anthropic (or to any server, including one of ours)
breaks that promise outright. Being careful, short retention or no training doesn't change it. So:

- An API LLM rater can only ever be a **research benchmark on public datasets**, never part of the product or the study.
- The only form compatible with the design is a model running **on the participant's own machine**, in the browser
  (WebGPU) or as a local app, with only the five numbers leaving. This repo's `local` provider is the bench version
  of that. Rated on the same people with a small model (a few billion parameters), it would show whether a
  laptop-sized model keeps whatever advantage an API model shows. That's the result that would matter for the product.
- A local model on **our** laptop doesn't solve it for participants either: their text would still have to reach us.

## Plain English (for the owner)

- **What "zero-shot" means.** We don't teach the LLM anything. We hand it someone's writing and ask, like a
  psychologist reading a diary: "how would this person score on the standard personality questionnaire?" Our own
  model is the opposite: it learned which words high and low scorers used in one particular set of 1990s student
  essays.
- **Why it might travel better.** Our model learned "student essay" habits, and those stop working on tweets. An LLM
  has read enormous amounts of every kind of writing, so its idea of "sounds anxious" or "sounds warm" isn't tied to
  one genre.
- **Why it might not.** Published tests on these same datasets find LLMs about as weak as trained models (around 57%
  accuracy when 50% is a coin flip). The best evidence (r ≈ 0.3) needed hundreds of posts per person.
- **How we stop ourselves fooling ourselves.** We wrote the pass mark before seeing any answer. The LLM must beat our
  model by at least 0.03 AUC on the two kinds of writing our model never saw, with a 95% CI that excludes zero,
  on the same people and the same text. If it doesn't, we keep what we have.
- **What it costs.** About $1.13 for 450 ratings with Haiku 4.5, or about $0.13 with the newer Haiku 5.5. The full test
  sets cost about $3 with Haiku 4.5. The 450-rating Haiku 4.5 run has been made (about $1.13 estimated); any
  further run needs your go-ahead and `AAS_LLM_ALLOW_SPEND=1`.
- **Why it can't go in the product as is.** The product promises that people's words never leave their computer. An
  API means sending their words to a company. This experiment only tells us whether the *idea* works. If it does,
  the next step is a small model that runs on their machine.

## Reproduce

```bash
cd py
uv run pytest tests/test_llm_rater.py -q                       # 35 tests, mocked, no network
uv run python scripts/llm_crossdomain.py --dry-run             # sample: prompts, tokens, cost
uv run python scripts/llm_crossdomain.py --dry-run --full      # every test person
# after approval:
AAS_LLM_ALLOW_SPEND=1 AAS_LLM_MODEL=claude-haiku-4-5-20251001 uv run python scripts/llm_crossdomain.py --live
# writes docs/llm_crossdomain_results_<provider>_<model>.csv/.json with the decision
```

## Local small models on this laptop (pre-registered 2026-11-01, written before any local rating was obtained)

**Why.** The Claude Haiku 4.5 run (`llm_crossdomain_results_anthropic_claude-haiku-4-5-20251001.json`) came out
**REPLACE**: held-out-domain mean AUC 0.618 against 0.558 for bag of words, delta +0.060 [0.025, 0.093]. The product
can't send a participant's text to any API, so the question that matters is whether a model small enough to run on
the participant's own laptop keeps that advantage.

**Models (chosen before any rating, by quality per download size, each at most 1.6 GB on disk):**

| model (Ollama tag) | params | download | why | public scores (non-thinking) |
|---|---|---|---|---|
| `qwen3:1.7b` (Q4_K_M) | 1.7B | 1.4 GB | best instruction-following under the size cap | MMLU-Redux 64.4, IFEval strict 68.2 (Qwen3 tech report, arXiv:2505.09388, Table 20; Gemma-3-1B-IT there scores 33.3 / 54.5, Qwen2.5-1.5B 50.7 / 42.5) |
| `qwen3.5:0.8b-q8_0` | 0.8B | 1.0 GB | newest small generation, under half the size: tests how far down the advantage survives | MMLU-Redux 48.5, MMLU-Pro 29.7, IFEval 52.1 (Qwen3.5-0.8B model card, Hugging Face) |

Ruled out by size: `qwen3.5:2b` (1.9 GB at Q4_K_M), Gemma 4 E2B (4.3 GB), Llama 3.2 3B / Phi-4-mini (> 2 GB).
`gemma3:1b` (0.8 GB) and `llama3.2:1b` (1.3 GB) were the alternatives; they score lower on the same tables.

**Identical to the Haiku run:** prompt `b5-ipip50-v1` (system and user text byte for byte), the same 150 people per
dataset (seed 20261101), the same leak stripping and 1,500-token cut, the same bag-of-words arms, the same scoring,
bootstrap and decision code (`scripts/llm_crossdomain.py --live`). Settings: temperature 0, max_tokens 600, via
Ollama's OpenAI-compatible endpoint on `127.0.0.1` only. Thinking is switched off with a request flag
(`reasoning_effort: "none"` / `think: false` in the request body, recorded in the cache key), **not** by adding
`/no_think` to the prompt, so the text the model reads is the same as Haiku's. The server context is set to 8,192
tokens so no prompt is silently cut; prompt token counts are checked afterwards.

**Rule (same as for Haiku).** A local model **keeps the advantage** if its held-out-domain (PAN 2015 + vlogs) mean
AUC beats bag of words by **≥ 0.03 with the 95% paired-bootstrap CI excluding 0** (the REPLACE condition). AUGMENT /
KEEP as defined above are reported too. Also reported, not in the rule: the gap to Haiku (local minus Haiku held-out
mean AUC on the people both rated, paired bootstrap, 2,000 resamples), Essays numbers, and per-trait AUCs.

**Missing ratings.** As the rule says: a person whose reply is invalid twice (or refused) is dropped from every arm
of that model's comparison. The JSON failure rate per model is reported. If missingness is above 5% in a dataset, the
bag-of-words AUC on the dropped people is compared with that on the kept people, to say whether dropping them could
favour the LLM arm (e.g. if the model fails mostly on short or odd texts that bag of words also finds hard).

**Time and fallback.** Runtime is timed on 5 people first. If a model would take more than 90 minutes for 450 people,
the smaller model is used or the run is cut to **the first 100 of the same 150 people per dataset** (a subset, so the
Haiku comparison stays paired), and this is said.

**Disk.** One model on disk at a time; free space must never fall below 1.8 GB (a watchdog stops the run if it
does); each model's weights are deleted when its run ends.

### What happened (2026-11-01, same day): run stopped on the disk rule, no verdict

- **Disk.** Free space was 3.46 GiB before the download and 2.2 GiB after pulling `qwen3:1.7b`. Two minutes into
  the full run it fell to **1.2 GiB**. The watchdog stopped Ollama and the script, and the weights were deleted.
  Free space is now **2.4 GiB**, about 1 GiB below the start, and nothing of this run is left on disk (`~/.ollama`
  96 KB). The missing gigabyte isn't our files. The likeliest causes are the other session's iOS build, or macOS
  growing its swap while the model sat in memory (swap total 5.1 GB, nearly full, on a 16 GB machine). At 2.4 GiB,
  no candidate fits the rule: `qwen3:1.7b` would leave about 1.0 GiB, `qwen3.5:0.8b-q8_0` about 1.4 GiB and even
  `gemma3:1b` about 1.6 GiB. **Not resumed.** It needs about 3.5 GiB free (model plus 1.8 GiB floor plus swap
  margin) or a model on an external drive.
- **Thinking off.** Ollama's OpenAI endpoint honours `reasoning_effort: "none"` for qwen3: about 2 s and 170 output
  tokens per reply, against 600 tokens (cut off) with thinking on. `think: false` was ignored on that endpoint. The flag goes
  in through `AAS_LLM_LOCAL_EXTRA` (small addition to `scripts/llm_crossdomain.py`: `LocalProviderExtra`). The
  prompt is unchanged.
- **Timing.** 4.3 s per person on the M2 Pro, so about 32 min for 450 people (within the 90-minute limit).
- **Pilot, not a result (17 people rated before the stop).** One reply was invalid JSON twice. The retry runs at
  temperature 0, so it repeats the same output and adds nothing. Of the 16 valid replies, **15 were 3.0 on all five
  traits**: the model copied the example JSON in the prompt (`"score": 3.0`). A model that gives everyone 3.0 has
  AUC 0.5, so if this rate held, `qwen3:1.7b` would come out **KEEP** by a wide margin under the prompt as
  registered. That isn't measured; it's what 17 people suggest. A prompt without a numeric example would be a new,
  post-hoc version (`b5-ipip50-v2`), not this test.
- Cached replies (17) stay in `data/llm_cache` and will be reused when the run resumes.

## Contamination check for the Haiku result (2026-11-01, no API calls)

**Dates.** Every dataset is public long before every model's cutoff, so nothing rules contamination out:

| dataset | text written | public online copy | Claude Haiku 4.5 (training data to Jul 2025, reliable knowledge to Feb 2025) | Qwen3 (released Apr 2025) / Qwen3.5 (Feb 2026) |
|---|---|---|---|---|
| Essays | 1990s | SenticNet GitHub mirror and many Kaggle/GitHub copies, for years | predates | predates |
| PAN 2015 | 2014-15 tweets | PAN 2015 (2015); Zenodo 3745945 (2020) | predates | predates |
| YouTube vlogs | 2011-13 | WCPR 2014; Hugging Face `facells/...` | predates | predates |

**What would be memorised.** In the Essays and vlog CSVs, the labels sit **on the same row as the text**
(`cEXT`... / 1-7 scores). A crawl of GitHub or Hugging Face could therefore have put text and label side by side in
pre-training. Memorising five yes/no bits per essay from a few copies is unlikely for a small, generalist model, but
it can't be excluded. PAN's truth files are separate from the tweets (`truth.txt` keyed by user id), so the
text-label pairing is much less direct there.

**Cheap evidence we have (no spend).** In all 450 Haiku reasons, none names a dataset, a corpus or a score.
One essay mentions "Professor Pennebaker" in its own text, so the source is visible in that one input. The advantage over bag of words is
largest on vlogs. Those labels are **observer** impressions of the vlogger, which is exactly the judgment the prompt
asks the model to imitate, so a real (non-memorised) advantage there is plausible. Overall: suggestive, not proof.

**A clean check, designed now, to run when PANDORA or study data arrives (not run):**

1. **Unseen text, primary.** Rate people whose text and labels can't be in any training set: the study's own
   participants, rated on their own machine with the five numbers coming back. PANDORA is a weaker option: its
   comments are public on Reddit, but its labels are behind a request. Use the same prompt (`b5-ipip50-v1`), the
   same budget, and a pre-registered sample (n ≥ 300 for a ±0.03 CI). Compare Haiku's AUC on unseen text with its
   AUC on PAN and vlogs. **Contamination flag:** if unseen-text AUC falls more than 0.05 below the public-dataset AUC
   *and* bag of words doesn't fall by a similar amount on the same data, the public-dataset margin is suspect.
2. **Paraphrase test, on the public sets.** Have a different model (a local one, so no spend) paraphrase each text so
   that meaning is kept but no 8-word span survives, then re-rate with Haiku. A memorised text-label link should break under
   paraphrase while a real judgment shouldn't: a drop in AUC of more than 0.03 (paired CI) points to memorisation.
3. **Verbatim recall probe.** Give Haiku the first 50 words of each essay or vlog and ask it to continue; score the
   longest exact overlap with the true next 50 words. Use paraphrased or post-cutoff text as the control. A rate of
   exact 13-word or longer continuations well above the control means the texts were seen in training.
   It costs about $0.10 per 450 people on Haiku 4.5.
4. **Label-recall probe.** Ask directly for the dataset label of a given essay ("in the Pennebaker essays corpus,
   is this author high or low on cEXT?"). Above-chance accuracy beyond what plain rating gives means label leakage.

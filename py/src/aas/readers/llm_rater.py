"""LLMRater: ask a large language model to rate a person's Big Five from their writing, zero-shot.

Nothing here is trained on our data. The model reads up to a fixed budget of the person's own
(leak-stripped) words and answers with a 1.0-5.0 score per trait on the IPIP-50 scale plus a one-line
rationale, as JSON. See docs/llm_rater.md for the experiment, the prompt and the privacy position.

    from aas.readers.llm_rater import LLMRater
    r = LLMRater.from_env()             # provider + model from env vars (below)
    p = r.build_prompt(text)            # what would be sent (no network)
    rating = r.rate(text)               # network call unless the answer is cached

Providers (env AAS_LLM_PROVIDER):
  anthropic  official anthropic SDK; needs ANTHROPIC_API_KEY (or an `ant auth login` profile) AND
             AAS_LLM_ALLOW_SPEND=1, because every uncached call costs money. Model: AAS_LLM_MODEL
             (default claude-haiku-4-5-20251001; claude-sonnet-5-5 and claude-haiku-5-5 also configured).
  local      an OpenAI-compatible /chat/completions endpoint ON THIS MACHINE (llama.cpp server, Ollama,
             LM Studio). AAS_LLM_LOCAL_URL (e.g. http://127.0.0.1:11434/v1) and AAS_LLM_LOCAL_MODEL.
             Only loopback hosts are accepted: text never leaves the laptop on this path.

Cache: one JSON file per sha256(prompt version, provider, model, request settings, prompt text) under
data/llm_cache (gitignored). A cached answer is returned without any network call, so
re-running an experiment is free and gives identical numbers.

Privacy: the text is leak-stripped exactly as the trained text readers strip it (readers.text.clean), and
build_prompt refuses to emit a prompt whose person-text still contains a typology label or a Big Five
self-report word. No label of any kind (dataset labels, MBTI types, scores) is ever put in the prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from ..slices.text import TYPE as MBTI_TYPE
from ..slices.text_pandora import BIG5_LEAK
from . import text as R

PROMPT_VERSION = "b5-ipip50-v1"
TRAITS = ("O", "C", "E", "A", "N")
TRAIT_KEYS = {"O": "openness", "C": "conscientiousness", "E": "extraversion", "A": "agreeableness", "N": "neuroticism"}
CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "llm_cache"
TEXT_TOKEN_BUDGET = 1500       # estimated tokens of the person's text per request (~1,100 words)
MAX_OUTPUT_TOKENS = 600        # five ratings + five one-line rationales is ~250 tokens; headroom for JSON slack
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
LOOPBACK = {"localhost", "127.0.0.1", "::1"}

# Per-model request settings. Kept here (and in the cache key) so a change in settings never reuses an
# answer produced under different settings. Thinking is kept off or minimal: the task is a short judgment,
# and thinking tokens are billed as output.
#   Haiku 4.5: no thinking unless asked (omitting `thinking` = off).
#   Sonnet 5.5: `thinking: {type: "disabled"}` is rejected; `between_tools` turns off up-front thinking
#     and is allowed at effort high or below.
#   Haiku 5.5: adaptive thinking; effort "low" keeps it short.
# The current SDK (anthropic 2.x) exposes no `temperature` argument, so answers are not guaranteed to be
# identical across calls; the cache makes each (model, text) answer fixed once obtained.
ANTHROPIC_MODELS = {
    "claude-haiku-4-5-20251001": {"tokenizer": "pre-4.7", "max_tokens": MAX_OUTPUT_TOKENS, "request": {}},
    "claude-haiku-4-5": {"tokenizer": "pre-4.7", "max_tokens": MAX_OUTPUT_TOKENS, "request": {}},
    "claude-sonnet-5-5": {"tokenizer": "4.7+", "max_tokens": MAX_OUTPUT_TOKENS,
                          "request": {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}},
    # adaptive thinking counts against max_tokens, so leave room for it
    "claude-haiku-5-5": {"tokenizer": "4.7+", "max_tokens": 2000, "request": {"output_config": {"effort": "low"}}},
}

SYSTEM = (
    "You are an experienced personality psychologist. You rate people's Big Five personality traits from "
    "samples of their own writing or speech, as a careful observer would. You always answer with a single "
    "JSON object and nothing else."
)

INSTRUCTIONS = """Below is a sample of text written or spoken by one person. It may be informal, cut short, or made of separate short posts.

Rate this person on each Big Five trait as their score on the 50-item IPIP Big Five questionnaire would come out: the mean of ten items answered on a 1-5 scale (1 = very inaccurate, 5 = very accurate), so 3.0 is the middle and most people score between 2.0 and 4.5.

- openness: imagination, curiosity, interest in ideas and art
- conscientiousness: organised, dependable, careful, hard-working
- extraversion: sociable, talkative, energetic, assertive
- agreeableness: warm, kind, trusting, cooperative
- neuroticism: anxious, moody, easily stressed or upset (high = more neurotic)

Use only the text. If the text says little about a trait, stay near the middle rather than guessing an extreme.

Answer with exactly this JSON shape, scores as numbers from 1.0 to 5.0 with one decimal, each reason one short line:
{"openness": {"score": 3.0, "reason": "..."}, "conscientiousness": {"score": 3.0, "reason": "..."}, "extraversion": {"score": 3.0, "reason": "..."}, "agreeableness": {"score": 3.0, "reason": "..."}, "neuroticism": {"score": 3.0, "reason": "..."}}

<text>
{TEXT}
</text>"""


# ---------------------------------------------------------------- token estimate and text budget

def estimate_tokens(text: str, tokenizer: str = "pre-4.7") -> int:
    """Rough Claude token count WITHOUT a network call. Anthropic publishes no offline tokenizer, and
    OpenAI's (tiktoken) undercounts Claude. Method: the larger of 4 characters per token and 1.33 tokens
    per whitespace word (the usual English rule of thumb for the pre-4.7 tokenizer), times 1.30 for the
    tokenizer used by Claude 4.7 and later models (incl. Sonnet 5.5, Haiku 5.5), which Anthropic's pricing
    page says yields ~30% more tokens for the same text. Exact counts need POST /v1/messages/count_tokens
    (free, but it is an API call with a key, so the dry run does not use it)."""
    base = max(len(text) / 4.0, len(text.split()) * 1.33)
    return int(math.ceil(base * (1.30 if tokenizer == "4.7+" else 1.0)))


def budget_text(text: str, budget_tokens: int = TEXT_TOKEN_BUDGET, seed: int = 0) -> tuple[str, dict]:
    """Cut a person's text to about `budget_tokens` (pre-4.7 estimate), the same way for every model.

    Many short items (tweets: one per line) -> a seeded random sample of whole items, kept in their
    original order, so the model sees a spread of the person's posts and not just the first few days.
    One long text (an essay, a vlog transcript) -> the first N words, so it stays readable prose.
    Returns (text, info) with words before/after and whether it was cut."""
    text = text.strip()
    words_in = R.n_words(text)
    if estimate_tokens(text) <= budget_tokens:
        return text, {"words_in": words_in, "words_out": words_in, "cut": False, "how": "whole"}
    items = [x for x in text.split("\n") if x.strip()]
    if len(items) >= 10:
        order = np.random.default_rng(seed).permutation(len(items))
        keep, used = [], 0
        for i in order:
            t = estimate_tokens(items[i]) + 1
            if used + t > budget_tokens:
                continue
            keep.append(i)
            used += t
        out = "\n".join(items[i] for i in sorted(keep))
        how = f"random sample of {len(keep)} of {len(items)} items (seed {seed})"
    else:
        w = text.split()
        n = int(budget_tokens / 1.33)
        while n > 0 and estimate_tokens(" ".join(w[:n])) > budget_tokens:
            n -= 10
        out = " ".join(w[:n])
        how = f"first {n} words"
    return out, {"words_in": words_in, "words_out": R.n_words(out), "cut": True, "how": how}


# ---------------------------------------------------------------- prompt and label guard

class LabelLeak(ValueError):
    """The person-text part of a prompt still contains a personality label or self-report word."""


def assert_no_labels(person_text: str) -> None:
    """Refuse text that names an MBTI type or a Big Five / questionnaire word. clean() removes these, so
    this only fires if a caller bypasses clean() or the stripper regresses."""
    for pat, what in ((MBTI_TYPE, "MBTI type"), (BIG5_LEAK, "Big Five / questionnaire word")):
        m = pat.search(person_text)
        if m:
            raise LabelLeak(f"{what} {m.group(0)!r} in the text that would be sent")


def build_messages(person_text: str) -> tuple[str, str]:
    """(system, user) for one person. person_text must already be cleaned and budgeted."""
    assert_no_labels(person_text)
    return SYSTEM, INSTRUCTIONS.replace("{TEXT}", person_text)


# ---------------------------------------------------------------- reply parsing

class BadReply(ValueError):
    pass


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse_reply(reply: str) -> dict:
    """Model reply -> {"O": {"score": float, "reason": str}, ...}. Raises BadReply on anything off-spec:
    no JSON, missing or extra traits, scores outside 1-5 or not numbers, empty reasons."""
    if not isinstance(reply, str) or not reply.strip():
        raise BadReply("empty reply")
    s = reply.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)  # tolerate a code fence
    m = _JSON_OBJ.search(s)
    if not m:
        raise BadReply("no JSON object in reply")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise BadReply(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise BadReply("JSON is not an object")
    keys = {k.strip().lower(): v for k, v in obj.items()}
    want = set(TRAIT_KEYS.values())
    if set(keys) != want:
        raise BadReply(f"traits {sorted(keys)} != {sorted(want)}")
    out = {}
    for t, name in TRAIT_KEYS.items():
        v = keys[name]
        if not isinstance(v, dict) or "score" not in v:
            raise BadReply(f"{name}: expected {{score, reason}}")
        sc = v["score"]
        if isinstance(sc, bool) or not isinstance(sc, (int, float)):
            try:
                sc = float(str(sc).strip())
            except ValueError:
                raise BadReply(f"{name}: score {v['score']!r} is not a number") from None
        sc = float(sc)
        if not (math.isfinite(sc) and 1.0 <= sc <= 5.0):
            raise BadReply(f"{name}: score {sc} outside 1-5")
        reason = str(v.get("reason", "")).strip()
        if not reason:
            raise BadReply(f"{name}: empty reason")
        out[t] = {"score": round(sc, 2), "reason": reason[:300]}
    return out


# ---------------------------------------------------------------- providers

@dataclass
class Reply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, client=None):
        if model not in ANTHROPIC_MODELS:
            raise ValueError(f"model {model!r} not configured; add its request settings to ANTHROPIC_MODELS")
        self.model = model
        self.settings = ANTHROPIC_MODELS[model]["request"]
        self.max_tokens = ANTHROPIC_MODELS[model]["max_tokens"]
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if os.environ.get("AAS_LLM_ALLOW_SPEND") != "1":
                raise PermissionError("Anthropic calls cost money: set AAS_LLM_ALLOW_SPEND=1 only after the owner approves the spend")
            import anthropic

            self._client = anthropic.Anthropic(max_retries=4)
        return self._client

    def complete(self, system: str, user: str) -> Reply:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}], **self.settings)
        # A refusal is recorded as such, NOT retried on another model: a silent fallback would change which
        # model is being evaluated.
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        u = getattr(r, "usage", None)
        return Reply(text, getattr(u, "input_tokens", None), getattr(u, "output_tokens", None), r.stop_reason)


class LocalProvider:
    """OpenAI-compatible chat endpoint on this machine only (llama.cpp `llama-server`, Ollama, LM Studio)."""

    name = "local"

    def __init__(self, url: str, model: str, timeout: float = 600.0):
        host = urlparse(url).hostname or ""
        if host not in LOOPBACK:
            raise ValueError(f"local provider must be on this machine (localhost/127.0.0.1/::1), got host {host!r}")
        self.url = url.rstrip("/")
        self.model = model
        self.settings = {"temperature": 0}
        self.max_tokens = MAX_OUTPUT_TOKENS
        self.timeout = timeout

    def complete(self, system: str, user: str) -> Reply:
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as f:
            r = json.loads(f.read())
        ch = r["choices"][0]
        u = r.get("usage") or {}
        return Reply(ch["message"].get("content") or "", u.get("prompt_tokens"), u.get("completion_tokens"), ch.get("finish_reason"))


# ---------------------------------------------------------------- the rater

@dataclass
class Rating:
    scores: dict | None             # {"O": 3.4, ...} or None when the reply was unusable
    reasons: dict | None
    ok: bool
    error: str | None
    cached: bool
    key: str
    meta: dict = field(default_factory=dict)


class LLMRater:
    def __init__(self, provider, cache_dir: Path | str = CACHE_DIR, budget_tokens: int = TEXT_TOKEN_BUDGET,
                 retries_on_bad_reply: int = 1):
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.budget_tokens = budget_tokens
        self.retries = retries_on_bad_reply

    @classmethod
    def from_env(cls, **kw) -> "LLMRater":
        p = os.environ.get("AAS_LLM_PROVIDER", "anthropic")
        if p == "anthropic":
            return cls(AnthropicProvider(os.environ.get("AAS_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)), **kw)
        if p == "local":
            url, model = os.environ.get("AAS_LLM_LOCAL_URL"), os.environ.get("AAS_LLM_LOCAL_MODEL")
            if not url or not model:
                raise ValueError("local provider needs AAS_LLM_LOCAL_URL and AAS_LLM_LOCAL_MODEL")
            return cls(LocalProvider(url, model), **kw)
        raise ValueError(f"unknown AAS_LLM_PROVIDER {p!r} (anthropic | local)")

    # -- prompt
    def prepare(self, raw_text: str, seed: int = 0) -> tuple[str, str, dict]:
        """clean (leak strip) -> budget -> (system, user, info). No network."""
        cleaned = "\n".join(c for c in (R.clean(x) for x in raw_text.split("\n")) if c)
        text, info = budget_text(cleaned, self.budget_tokens, seed)
        system, user = build_messages(text)
        return system, user, info

    def build_prompt(self, raw_text: str, seed: int = 0) -> str:
        s, u, _ = self.prepare(raw_text, seed)
        return s + "\n\n" + u

    def key(self, system: str, user: str) -> str:
        blob = json.dumps({"v": PROMPT_VERSION, "provider": self.provider.name, "model": self.provider.model,
                           "settings": self.provider.settings, "max_tokens": self.provider.max_tokens,
                           "system": system, "user": user}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def cached(self, raw_text: str, seed: int = 0) -> Rating | None:
        s, u, _ = self.prepare(raw_text, seed)
        k = self.key(s, u)
        p = self._path(k)
        return self._from_record(json.loads(p.read_text()), k, True) if p.exists() else None

    # -- rate
    def rate(self, raw_text: str, seed: int = 0) -> Rating:
        system, user, info = self.prepare(raw_text, seed)
        k = self.key(system, user)
        p = self._path(k)
        if p.exists():
            return self._from_record(json.loads(p.read_text()), k, True)
        attempts, last_err, rep = [], None, None
        for _ in range(1 + self.retries):
            t0 = time.time()
            rep = self.provider.complete(system, user)
            attempts.append({"seconds": round(time.time() - t0, 2), "input_tokens": rep.input_tokens,
                             "output_tokens": rep.output_tokens, "stop_reason": rep.stop_reason})
            if rep.stop_reason == "refusal":
                last_err = "refusal"
                break
            try:
                parsed = parse_reply(rep.text)
                last_err = None
                break
            except BadReply as e:
                last_err = f"bad reply: {e}"
        rec = {"prompt_version": PROMPT_VERSION, "provider": self.provider.name, "model": self.provider.model,
               "settings": self.provider.settings, "text_info": info, "attempts": attempts,
               "reply": rep.text if rep else None, "error": last_err,
               "parsed": None if last_err else parsed, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
        # Only definitive outcomes are cached (a parsed rating, or a refusal / repeated bad reply).
        # Exceptions (network, auth, rate limit) propagate and leave no cache entry.
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1))
        return self._from_record(rec, k, False)

    @staticmethod
    def _from_record(rec: dict, key: str, cached: bool) -> Rating:
        parsed = rec.get("parsed")
        meta = {k: rec.get(k) for k in ("model", "provider", "prompt_version", "text_info", "attempts")}
        if not parsed:
            return Rating(None, None, False, rec.get("error"), cached, key, meta)
        return Rating({t: parsed[t]["score"] for t in TRAITS}, {t: parsed[t]["reason"] for t in TRAITS}, True, None, cached, key, meta)


def as_dict(r: Rating) -> dict:
    return asdict(r)

"""LLMRater with a fake provider: no network, no API key, no spend."""

from __future__ import annotations

import json

import pytest

from aas.readers import llm_rater as L

GOOD = json.dumps({k: {"score": 3.4, "reason": "line"} for k in L.TRAIT_KEYS.values()})


class FakeProvider:
    name = "fake"
    model = "fake-1"
    settings: dict = {}
    max_tokens = 600

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        r = self.replies.pop(0)
        return r if isinstance(r, L.Reply) else L.Reply(r, 100, 50, "end_turn")


def rater(tmp_path, replies):
    p = FakeProvider(replies)
    return L.LLMRater(p, cache_dir=tmp_path), p


# ---------------------------------------------------------------- prompt building

def test_prompt_has_text_and_fixed_instructions(tmp_path):
    r, _ = rater(tmp_path, [])
    s, u, info = r.prepare("I went hiking with my sister and we got lost.")
    assert "I went hiking with my sister" in u and "<text>" in u and u.rstrip().endswith("</text>")
    assert "IPIP" in u and '"neuroticism"' in u and "JSON" in s
    assert not info["cut"]


def test_prompt_is_deterministic_and_versioned(tmp_path):
    r, _ = rater(tmp_path, [])
    a, b = r.build_prompt("same text here"), r.build_prompt("same text here")
    assert a == b
    s, u, _ = r.prepare("same text here")
    k1 = r.key(s, u)
    old = L.PROMPT_VERSION
    try:
        L.PROMPT_VERSION = "other"
        assert r.key(s, u) != k1
    finally:
        L.PROMPT_VERSION = old


def test_long_essay_cut_to_first_words(tmp_path):
    r, _ = rater(tmp_path, [])
    text = " ".join(f"w{i}" for i in range(5000))
    _, u, info = r.prepare(text)
    assert info["cut"] and info["how"].startswith("first") and "w0 w1 w2" in u and "w4999" not in u
    assert L.estimate_tokens(u.split("<text>")[1]) <= L.TEXT_TOKEN_BUDGET + 5


def test_many_tweets_sampled_seeded_in_order(tmp_path):
    r, _ = rater(tmp_path, [])
    tweets = "\n".join(f"tweet number {i} about my day and the weather and other things" for i in range(400))
    _, u1, info = r.prepare(tweets, seed=0)
    _, u2, _ = r.prepare(tweets, seed=0)
    _, u3, _ = r.prepare(tweets, seed=1)
    assert info["cut"] and "random sample" in info["how"] and u1 == u2 and u1 != u3
    body = u1.split("<text>\n")[1].split("\n</text>")[0].split("\n")
    nums = [int(x.split()[2]) for x in body]
    assert nums == sorted(nums) and len(nums) < 400


# ---------------------------------------------------------------- no labels ever sent

def test_labels_stripped_from_text(tmp_path):
    r, _ = rater(tmp_path, [])
    raw = ("As an INTJ I scored high on openness and low on neuroticism in the Big Five test (90th percentile).\n"
           "My ENFP friend is so extraverted. Anyway the weather is nice.")
    _, u, _ = r.prepare(raw)
    body = u.split("<text>")[1]
    for bad in ("INTJ", "ENFP", "openness", "neuroticism", "Big Five", "percentile", "extraverted"):
        assert bad.lower() not in body.lower()
    assert "the weather is nice" in body


def test_whole_prompt_has_no_mbti_type():
    full = L.SYSTEM + L.INSTRUCTIONS
    assert not L.MBTI_TYPE.search(full)


def test_guard_refuses_unstripped_label():
    with pytest.raises(L.LabelLeak):
        L.build_messages("I am an ISTP and proud")
    with pytest.raises(L.LabelLeak):
        L.build_messages("my conscientiousness is low")


def test_dataset_labels_not_in_prompt(tmp_path):
    """Only text goes in: a caller can't pass labels, and label-looking values from the data don't appear."""
    r, p = rater(tmp_path, [GOOD])
    r.rate("I love cooking for people.")
    system, user = p.calls[0]
    for leak in ("cEXT", "cNEU", "cAGR", "cCON", "cOPN", "\ny\n", "stability", "BFI"):
        assert leak not in system + user


# ---------------------------------------------------------------- reply parsing

def test_parse_good_and_fenced():
    out = L.parse_reply(GOOD)
    assert set(out) == set(L.TRAITS) and out["O"]["score"] == 3.4
    fenced = "```json\n" + GOOD + "\n```"
    assert L.parse_reply(fenced)["N"]["score"] == 3.4
    assert L.parse_reply("Here you go: " + GOOD)["A"]["reason"] == "line"
    s = json.loads(GOOD)
    s["openness"]["score"] = "4.1"
    assert L.parse_reply(json.dumps(s))["O"]["score"] == 4.1


@pytest.mark.parametrize("bad", [
    "", "I cannot rate people.", "{not json}", "[1,2,3]",
    json.dumps({"openness": {"score": 3, "reason": "x"}}),                                         # missing traits
    json.dumps({**{k: {"score": 3, "reason": "x"} for k in L.TRAIT_KEYS.values()}, "honesty": {"score": 3, "reason": "x"}}),
    json.dumps({k: {"score": 6.5, "reason": "x"} for k in L.TRAIT_KEYS.values()}),                 # out of range
    json.dumps({k: {"score": 0, "reason": "x"} for k in L.TRAIT_KEYS.values()}),
    json.dumps({k: {"score": "high", "reason": "x"} for k in L.TRAIT_KEYS.values()}),              # not a number
    json.dumps({k: {"score": True, "reason": "x"} for k in L.TRAIT_KEYS.values()}),
    json.dumps({k: {"score": 3, "reason": ""} for k in L.TRAIT_KEYS.values()}),                    # empty reason
    json.dumps({k: 3 for k in L.TRAIT_KEYS.values()}),                                             # wrong shape
    json.dumps({k: {"score": float("nan"), "reason": "x"} for k in L.TRAIT_KEYS.values()}),
])
def test_parse_rejects(bad):
    with pytest.raises(L.BadReply):
        L.parse_reply(bad)


# ---------------------------------------------------------------- rate: retries, refusals, cache

def test_bad_then_good_reply_retries_once(tmp_path):
    r, p = rater(tmp_path, ["nope", GOOD])
    out = r.rate("some text about my weekend")
    assert out.ok and out.scores["E"] == 3.4 and len(p.calls) == 2 and not out.cached


def test_two_bad_replies_recorded_as_missing(tmp_path):
    r, p = rater(tmp_path, ["nope", "still nope"])
    out = r.rate("some text about my weekend")
    assert not out.ok and out.scores is None and out.error.startswith("bad reply")


def test_refusal_not_retried(tmp_path):
    r, p = rater(tmp_path, [L.Reply("", 10, 0, "refusal")])
    out = r.rate("some text")
    assert not out.ok and out.error == "refusal" and len(p.calls) == 1


def test_cache_hit_makes_no_call(tmp_path):
    r, p = rater(tmp_path, [GOOD])
    a = r.rate("cache me please")
    b = r.rate("cache me please")
    assert len(p.calls) == 1 and not a.cached and b.cached and a.scores == b.scores and a.key == b.key
    assert r.cached("cache me please").ok
    assert list(tmp_path.rglob("*.json"))


def test_cache_key_depends_on_model_and_text(tmp_path):
    r, p = rater(tmp_path, [GOOD, GOOD, GOOD])
    r.rate("text one")
    r.rate("text two")
    p.model = "fake-2"
    r.rate("text one")
    assert len(p.calls) == 3


def test_provider_exception_leaves_no_cache(tmp_path):
    class Boom(FakeProvider):
        def complete(self, s, u):
            raise ConnectionError("down")

    r = L.LLMRater(Boom([]), cache_dir=tmp_path)
    with pytest.raises(ConnectionError):
        r.rate("x y z")
    assert not list(tmp_path.rglob("*.json"))


# ---------------------------------------------------------------- providers without network

def test_anthropic_needs_spend_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("AAS_LLM_ALLOW_SPEND", raising=False)
    r = L.LLMRater(L.AnthropicProvider(), cache_dir=tmp_path)
    with pytest.raises(PermissionError):
        r.rate("hello there friend")


def test_anthropic_request_shape_with_mock_client(tmp_path):
    class Block:
        type = "text"
        text = GOOD

    class Resp:
        content = [Block()]
        stop_reason = "end_turn"

        class usage:
            input_tokens = 900
            output_tokens = 200

    class Msgs:
        def __init__(self):
            self.kw = None

        def create(self, **kw):
            self.kw = kw
            return Resp()

    class Client:
        messages = Msgs()

    c = Client()
    r = L.LLMRater(L.AnthropicProvider("claude-sonnet-5-5", client=c), cache_dir=tmp_path)
    out = r.rate("I like long walks.")
    assert out.ok
    kw = c.messages.kw
    assert kw["model"] == "claude-sonnet-5-5" and kw["thinking"] == {"type": "between_tools"}
    assert "temperature" not in kw and kw["messages"][0]["role"] == "user"


def test_unknown_anthropic_model_rejected():
    with pytest.raises(ValueError):
        L.AnthropicProvider("claude-made-up")


def test_local_provider_refuses_remote_host():
    with pytest.raises(ValueError):
        L.LocalProvider("http://203.0.113.7:11434/v1", "qwen")
    L.LocalProvider("http://127.0.0.1:11434/v1", "qwen")
    L.LocalProvider("http://localhost:8080/v1", "qwen")


def test_from_env_local(monkeypatch, tmp_path):
    monkeypatch.setenv("AAS_LLM_PROVIDER", "local")
    monkeypatch.setenv("AAS_LLM_LOCAL_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("AAS_LLM_LOCAL_MODEL", "qwen3:8b")
    r = L.LLMRater.from_env(cache_dir=tmp_path)
    assert r.provider.name == "local" and r.provider.model == "qwen3:8b"


def test_cache_dir_is_gitignored_data():
    assert L.CACHE_DIR.parts[-2:] == ("data", "llm_cache")


def test_token_estimate():
    t = "word " * 300
    assert L.estimate_tokens(t) >= 399 and L.estimate_tokens(t, "4.7+") > L.estimate_tokens(t)

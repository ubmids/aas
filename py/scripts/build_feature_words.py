"""Build the fixed word lists that aas.features ships with.

    cd py && uv run --with wordfreq python scripts/build_feature_words.py \
        --ldnoobw /path/to/LDNOOBW/en

wordfreq is NOT a project dependency: it is only needed here, once, to produce the lists. The
outputs are plain JSON and are what the extractor (and later the browser build) reads.

Writes
  src/aas/data/word_blocklist.json  the hand-made special-category / name / place blocklist, plus
                                    the LDNOOBW English list (CC BY 4.0) for profanity and slurs
  src/aas/data/common_words.json    words: the 2,000 opt-in word-rate words
                                    english_reference: wordfreq's top 5,000 alphabetic English
                                    words, used only locally to decide "is this query English?"
                                    (never written to a record)

The rule for the 2,000 (PRIVACY.md section 2, block B), applied in wordfreq frequency order:
  1. in wordfreq's top 5,000 English words ("best" list, wordfreq 3.1.1)
  2. only letters a-z, at least 2 of them (the readers' tokeniser drops 1-letter tokens)
  3. in the vocabulary of at least one trained text reader (unigrams of text-bigfive / text-mbti)
  4. not in /usr/share/dict/propernames (first names; except a few ordinary words such as "will"), not in the hand blocklist below, not in
     LDNOOBW (profanity, sexual terms, slurs)
  5. keep the first 2,000 that survive.
Every word that survived was read by a person (the author of this script) before it was frozen.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib

PY = Path(__file__).resolve().parents[1]
DATA = PY / "src" / "aas" / "data"
MODELS = PY / "models"
N_WORDS = 2000
VERSION = "1.0.0"

# Hand blocklist. Grouped so a reviewer can see why each word is out. Conservative on purpose:
# words that are usually innocent (black, white, pain, weight) are dropped when one of their common
# senses points at a special category, because the cost of losing one common word is tiny.
BLOCK = {
    "health": """health healthy healthcare medical medicine medication hospital hospitals doctor doctors
        patient patients cancer disease diseases illness ill sick sickness mental depression depressed anxiety
        anxious therapy therapist pregnant pregnancy surgery treatment treatments symptoms symptom diagnosis
        diagnosed injury injuries pain painful drug drugs alcohol drunk addiction addicted suicide suicidal
        diet weight fat obese overweight abortion disability disabled autism autistic adhd bipolar
        schizophrenia vaccine virus covid hiv aids dementia diabetes nurse clinic pill pills smoke smoking
        cigarette cigarettes weed marijuana cocaine overdose rehab psychiatric psychologist counselling""",
    "sexuality_gender_identity": """sex sexual sexuality sexy gay lesbian bisexual queer trans transgender
        gender lgbt homosexual straight porn husband wife girlfriend boyfriend""",
    "religion": """god gods lord jesus christ christian christians christianity church churches catholic
        protestant bible holy religious religion faith pray prayer prayers heaven hell muslim muslims islam
        islamic jewish jews judaism hindu buddhist atheist atheism spiritual christmas easter ramadan saint
        angel angels sin""",
    "politics_unions": """political politics politician politicians election elections elected vote votes
        voted voting voter democrat democrats democratic republican republicans liberal liberals conservative
        conservatives congress senate president presidential candidate candidates senator parliament minister labour labor union unions socialist
        communism communist fascist nazi trump obama clinton biden brexit leftist""",
    "ethnicity_nationality": """black white race racial racist racism ethnic ethnicity immigrant immigrants
        immigration native american americans british english french chinese russian german indian japanese
        australian canadian african asian spanish irish italian scottish european mexican korean jewish arab
        hispanic latino latina""",
    "crime": """crime crimes criminal criminals murder murdered arrested arrest jail prison prisoner guilty
        illegal convicted conviction trial victim victims abuse abused""",
    "finance_hardship": """debt debts loan loans bankrupt bankruptcy unemployed unemployment homeless
        eviction evicted welfare""",
    "places": """london america york california washington australia india china england europe france
        germany canada russia japan texas scotland britain ireland italy korea asia africa spain brazil mexico
        israel florida boston chicago paris ohio virginia jersey michigan pacific turkey angeles los san uk usa
        eu dc ca""",
    "names_brands": """john david james michael george paul peter william robert thomas tom jack louis chris
        mike ed richard joe mary charles frank henry martin jones lee harry scott steve ben ray sam tony bob alex
        daniel jim kim nick ryan adam andrew kevin matt smith johnson williams taylor jackson grace max bill al
        don google facebook twitter youtube amazon disney instagram microsoft iphone netflix bbc fox""",
    "foreign_or_fragments": """la de el da et le il un ma ii iii""",
}


# Ordinary words that /usr/share/dict/propernames also lists as first names (Will, Mark, ...).
# They stay: in running text they are almost always the common word.
# (This macOS file also lists "The", "No", "Think" and a few more; they stay too.)
NOT_NAMES = {"will", "mark", "son", "king", "rich", "sir", "judge", "may", "june", "art", "page", "rose",
             "the", "no", "think", "those", "part", "old", "real", "case", "trying", "per", "guy", "win",
             "major", "list", "price", "space", "miles", "vice", "leads"}


def blocklist(ldnoobw: Path) -> dict:
    hand = {k: sorted(set(v.split())) for k, v in BLOCK.items()}
    bad = sorted({w.strip().lower() for w in ldnoobw.read_text().splitlines() if w.strip()})
    return {
        "version": VERSION,
        "about": "Words never sent in the opt-in word-rate block, and (with the topic lexicon's sensitive list) "
                 "the reason an item is routed to topic 'other'. Hand groups are by special category; ldnoobw is "
                 "the List of Dirty, Naughty, Obscene, and Otherwise Bad Words (English), CC BY 4.0, "
                 "github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words",
        "hand": hand,
        "ldnoobw": bad,
    }


def main() -> None:
    import wordfreq
    from importlib.metadata import version

    ap = argparse.ArgumentParser()
    ap.add_argument("--ldnoobw", type=Path, required=True, help="LDNOOBW 'en' file")
    args = ap.parse_args()

    bl = blocklist(args.ldnoobw)
    blocked = {w for ws in bl["hand"].values() for w in ws} | set(bl["ldnoobw"])
    names = {w.strip().lower() for w in Path("/usr/share/dict/propernames").read_text().splitlines()} - NOT_NAMES

    top = wordfreq.top_n_list("en", 5000, wordlist="best")
    alpha = [w for w in top if re.fullmatch(r"[a-z]{2,}", w)]
    vocab: set[str] = set()
    for d in ("text_bigfive", "text_mbti"):
        vocab |= {k for k in joblib.load(MODELS / d / "model.joblib")["vocab"] if " " not in k}

    words = [w for w in alpha if w in vocab and w not in blocked and w not in names][:N_WORDS]
    assert len(words) == N_WORDS, len(words)

    (DATA / "word_blocklist.json").write_text(json.dumps(bl, indent=1) + "\n")
    (DATA / "common_words.json").write_text(json.dumps({
        "version": VERSION,
        "source": f"wordfreq {version('wordfreq')} top_n_list('en', 5000, wordlist='best') "
                  "(Robyn Speer, doi:10.5281/zenodo.7199437)",
        "licence": "wordfreq data: CC BY-SA 4.0. Attribution: Speer, R. (2022) rspeer/wordfreq: v3.0. Zenodo.",
        "rule": "top-5000 alphabetic English words (>= 2 letters) that are also in a trained text reader's "
                "unigram vocabulary, minus /usr/share/dict/propernames and word_blocklist.json, first 2,000 "
                "in frequency order. Hand-reviewed. See scripts/build_feature_words.py.",
        "words": words,
        "english_reference": alpha,
    }, indent=0) + "\n")
    print(f"{len(words)} words (last kept: {words[-1]!r}, rank {top.index(words[-1]) + 1} in wordfreq); "
          f"{len(alpha)} reference words; {len(blocked)} blocked")


if __name__ == "__main__":
    main()

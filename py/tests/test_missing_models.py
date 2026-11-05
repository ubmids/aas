"""Readers whose weights are not shipped fail with a message that says how to build them."""

from __future__ import annotations

import shutil

import pytest

from aas.readers import distilled as Dm
from aas.readers import text as R


@pytest.mark.parametrize("cls,folder,hint", [
    (R.TextMBTIReader, "text_mbti", "scripts/train_text_readers.py --only mbti"),
    (R.TextBigFiveReader, "text_bigfive", "scripts/train_text_readers.py --only bigfive"),
    (Dm.TextDistilledReader, "text_distilled", "scripts/distill.py --rate"),
])
def test_missing_weights_say_how_to_build(tmp_path, cls, folder, hint):
    d = tmp_path / folder
    d.mkdir()
    shutil.copy(R.MODELS_DIR / folder / "card.json", d / "card.json")
    with pytest.raises(FileNotFoundError, match=hint.replace(".", r"\.")):
        cls(model_dir=d)


def test_cards_ship_without_weights_still_describe_the_model():
    for folder in ("text_mbti", "text_bigfive", "text_distilled"):
        assert (R.MODELS_DIR / folder / "card.json").exists()

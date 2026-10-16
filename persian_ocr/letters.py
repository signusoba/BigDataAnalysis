"""The 32 isolated Persian alphabet classes used by this prototype."""
from __future__ import annotations

import unicodedata

_ALPHABET = [
    ("ا", "alef"), ("ب", "be"), ("پ", "pe"), ("ت", "te"),
    ("ث", "se"), ("ج", "jim"), ("چ", "che"), ("ح", "he-jimi"),
    ("خ", "khe"), ("د", "dal"), ("ذ", "zal"), ("ر", "re"),
    ("ز", "ze"), ("ژ", "zhe"), ("س", "sin"), ("ش", "shin"),
    ("ص", "sad"), ("ض", "zad"), ("ط", "ta"), ("ظ", "za"),
    ("ع", "eyn"), ("غ", "gheyn"), ("ف", "fe"), ("ق", "qaf"),
    ("ک", "kaf"), ("گ", "gaf"), ("ل", "lam"), ("م", "mim"),
    ("ن", "nun"), ("و", "vav"), ("ه", "he"), ("ی", "ye"),
]
LETTERS = [dict(index=i, character=char, name=name) for i, (char, name) in enumerate(_ALPHABET)]
_INDEX = {item["character"]: item["index"] for item in LETTERS}


def normalize_label(value: str) -> str:
    """Normalize compatible glyph forms and Arabic kaf/yeh; reject words/digits."""
    normalized = unicodedata.normalize("NFKC", value).strip().replace("ك", "ک").replace("ي", "ی")
    if normalized not in _INDEX:
        raise ValueError("Label must be exactly one of the 32 Persian alphabet characters.")
    return normalized


def label_index(value: str) -> int:
    return _INDEX[normalize_label(value)]

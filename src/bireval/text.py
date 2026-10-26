"""Tokenisers. Chinese has no spaces, so I compare plain character n-grams with spaCy's word segmenter."""
from __future__ import annotations

import re
from functools import lru_cache

EN_STOP = set("""a an the of to in on for and or is are was were be been by with as at from that this
which who whom what when where why how did do does it its into than then there their they he she his her
""".split())
ZH_PUNCT = re.compile(r"[\s，。、；：「」『』（）《》？！,.;:()\[\]\"'?!…—-]+")


def zh_char_ngrams(text: str) -> list[str]:
    s = ZH_PUNCT.sub("", text)
    return list(s) + [s[i:i + 2] for i in range(len(s) - 1)]


@lru_cache(maxsize=1)
def _zh_nlp():
    import spacy
    return spacy.load("zh_core_web_md", disable=["parser", "ner", "tagger", "attribute_ruler", "lemmatizer"])


def zh_words(text: str) -> list[str]:
    return [t.text for t in _zh_nlp()(text) if not ZH_PUNCT.fullmatch(t.text)]


def en_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in EN_STOP]


def en_words_bigrams(text: str) -> list[str]:
    w = en_words(text)
    return w + [w[i] + "_" + w[i + 1] for i in range(len(w) - 1)]


def tokenizer(lang: str, kind: str = "default"):
    if lang == "en":
        return en_words_bigrams if kind == "dense" else en_words
    return zh_words if kind == "words" else zh_char_ngrams

"""DRCD (Traditional Chinese) and SQuAD v1.1 (English) as retrieval benchmarks.

A question's gold passage is whichever chunk holds the start of its labelled answer,
so relevance comes from the human annotations, not from a model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
FILES = {
    ("zh", "train"): "DRCD_training.json", ("zh", "test"): "DRCD_test.json",
    ("en", "train"): "squad-train-v1.1.json", ("en", "test"): "squad-dev-v1.1.json",
}


@dataclass
class Benchmark:
    lang: str
    chunks: list[str]
    chunk_article: list[int]
    questions: list[str]
    qids: list[str]
    gold: list[set[int]]      # chunk indices


def _articles(lang: str, split: str):
    with open(DATA_DIR / FILES[(lang, split)], encoding="utf-8") as f:
        return json.load(f)["data"]


def _windows(text: str, size: int, lang: str) -> list[tuple[int, int]]:
    """Character spans of fixed-size windows with 25% overlap.

    Size counts characters for Chinese and words for English.
    """
    step, overlap = size - size // 4, size // 4
    if lang == "zh":
        return [(s, min(s + size, len(text))) for s in range(0, max(len(text) - overlap, 1), step)]
    starts, ends, pos = [], [], 0
    for tok in text.split():
        i = text.index(tok, pos)
        starts.append(i)
        ends.append(i + len(tok))
        pos = ends[-1]
    return [(starts[s], ends[min(s + size, len(starts)) - 1]) for s in range(0, max(len(starts) - overlap, 1), step)]


def load_benchmark(lang: str, split: str = "test", chunking: str = "paragraph",
                   max_questions: int | None = None, seed: int = 0, distractors: bool = True) -> Benchmark:
    """chunking is "paragraph" or "fixed-<size>".

    With distractors=True every training passage joins the corpus too (questions still come
    from `split` only), so it's a real search over 10-20k passages rather than ~1-2k.
    """
    chunks, chunk_article, questions, qids, gold = [], [], [], [], []

    def add_question(qa, hit):
        questions.append(qa["question"])
        qids.append(qa["id"])
        gold.append(hit)

    arts = [(a, True) for a in _articles(lang, split)]
    if distractors and split != "train":
        arts += [(a, False) for a in _articles(lang, "train")]
    for a_i, (art, ask) in enumerate(arts):
        paras = art["paragraphs"]
        if chunking == "paragraph":
            for p in paras:
                chunks.append(p["context"])
                chunk_article.append(a_i)
                for qa in p["qas"] if ask else []:
                    add_question(qa, {len(chunks) - 1})
            continue

        sep = "" if lang == "zh" else "\n"
        texts = [p["context"] for p in paras]
        full = sep.join(texts)
        offsets = np.cumsum([0] + [len(t) + len(sep) for t in texts[:-1]])
        spans = _windows(full, int(chunking.split("-")[1]), lang)
        base = len(chunks)
        chunks.extend(full[s:e] for s, e in spans)
        chunk_article.extend([a_i] * len(spans))
        for p_i, p in enumerate(paras):
            for qa in p["qas"] if ask else []:
                ans = qa["answers"][0]
                a0 = offsets[p_i] + ans["answer_start"]
                a1 = a0 + len(ans["text"])
                hit = {base + k for k, (s, e) in enumerate(spans) if s <= a0 and a1 <= e}
                # answer split across two windows: take any window holding its start
                hit = hit or {base + k for k, (s, e) in enumerate(spans) if s <= a0 < e}
                add_question(qa, hit)

    if max_questions and len(questions) > max_questions:
        idx = np.sort(np.random.default_rng(seed).choice(len(questions), max_questions, replace=False))
        questions, qids, gold = ([x[i] for i in idx] for x in (questions, qids, gold))
    return Benchmark(lang, chunks, chunk_article, questions, qids, gold)


def split_train_articles(lang: str, holdout_frac: float = 0.1, seed: int = 0, max_pairs: int | None = None,
                         max_heldout: int = 3000):
    """(question, paragraph) pairs from 90% of training articles for the dual encoder, and
    questions from the other 10% for the reranker, so it trains on scores the encoder hasn't overfit."""
    arts = _articles(lang, "train")
    rng = np.random.default_rng(seed)
    held = set(rng.choice(len(arts), int(len(arts) * holdout_frac), replace=False).tolist())
    pairs, heldout = [], []
    for i, art in enumerate(arts):
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                (heldout if i in held else pairs).append((qa["question"], p["context"]))
    if max_pairs and len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    if len(heldout) > max_heldout:
        heldout = [heldout[i] for i in rng.choice(len(heldout), max_heldout, replace=False)]
    return pairs, heldout

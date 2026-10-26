"""Run the full benchmark for zh (DRCD) and en (SQuAD).

Writes reports/results_<lang>.json and per_query_<lang>.parquet, and saves the
search index (BM25, dual encoder, reranker) to artifacts/ for the API.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bireval.data import load_benchmark, split_train_articles  # noqa: E402
from bireval.metrics import bootstrap_ci, gold_ranks, holm, paired_comparison, per_query  # noqa: E402
from bireval.rerank import Reranker, candidate_features  # noqa: E402
from bireval.retrievers import BM25, LSA, DualEncoder, RRFHybrid, SpacyMeanVectors, TfidfCosine  # noqa: E402
from bireval.text import tokenizer  # noqa: E402

REPORTS = ROOT / "reports"
ARTIFACTS = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
N_QUESTIONS = 3500
EPOCHS = int(os.environ.get("DE_EPOCHS", 8))


def chunking_study(lang, tok):
    """BM25 over fixed-size windows, holding the context budget (k x size) constant."""
    sizes, budget = ([128, 256, 512], 1024) if lang == "zh" else ([64, 128, 256], 512)
    out = {}
    for size in sizes:
        b = load_benchmark(lang, "test", chunking=f"fixed-{size}", max_questions=N_QUESTIONS)
        hit = (gold_ranks(BM25(tok).fit(b.chunks).score(b.questions), b.gold) <= budget // size).astype(float)
        out[f"fixed-{size}"] = {"n_chunks": len(b.chunks), "k": budget // size,
                                "recall_at_budget": float(hit.mean()), "ci95": bootstrap_ci(hit)}
    return {"budget": budget, "unit": "characters" if lang == "zh" else "words", "results": out}


def run_language(lang: str) -> dict:
    t0 = time.time()
    bench = load_benchmark(lang, "test", max_questions=N_QUESTIONS)
    tok = tokenizer(lang)
    pairs, heldout = split_train_articles(lang, max_pairs=60000)

    systems = {"bm25": BM25(tok, name="bm25")}
    if lang == "zh":
        systems["bm25_words"] = BM25(tokenizer(lang, "words"), name="bm25_words")
    systems |= {"tfidf": TfidfCosine(tok), "lsa": LSA(tok), "spacy_vectors": SpacyMeanVectors(lang)}
    for r in systems.values():
        r.fit(bench.chunks)
    de = DualEncoder(tokenizer(lang, "dense"), pairs, epochs=EPOCHS)
    t_de = time.time()
    systems["dual_encoder"] = de.fit(bench.chunks)
    de_train_sec = time.time() - t_de

    def score_sets(questions, bm25=None, dense=None, tfidf=None):
        sb = systems["bm25"].score(questions) if bm25 is None else bm25
        sd = de.score(questions) if dense is None else dense
        st = systems["tfidf"].score(questions) if tfidf is None else tfidf
        return {"bm25": sb, "dense": sd, "tfidf": st, "rrf": RRFHybrid([]).score(questions, [sb, sd])}

    scores = {name: r.score(bench.questions) for name, r in systems.items()}
    test_sets = score_sets(bench.questions, scores["bm25"], scores["dual_encoder"], scores["tfidf"])
    scores["rrf_bm25_dual"] = test_sets["rrf"]

    # The reranker learns on questions from held-out training articles, which the dual
    # encoder never saw, so its dense scores look like they will at test time.
    c_tokens = [set(tok(c)) for c in bench.chunks]
    c_len = np.array([len(c) for c in bench.chunks], float)
    pos = {}
    for i, c in enumerate(bench.chunks):
        pos.setdefault(c, set()).add(i)
    h = [(q, pos[p]) for q, p in heldout if p in pos]
    h_q, h_gold = [q for q, _ in h], [g for _, g in h]
    cand, feat = candidate_features(score_sets(h_q), [set(tok(q)) for q in h_q], c_tokens, c_len)
    rr = Reranker().fit(cand, feat, h_gold)
    cand, feat = candidate_features(test_sets, [set(tok(q)) for q in bench.questions], c_tokens, c_len)
    scores["rrf_rerank"] = rr.rescore(cand, feat, len(bench.chunks))

    # per-question metrics, CIs, and every system vs BM25 with Holm across the whole family
    per_q = {name: per_query(gold_ranks(s, bench.gold)) for name, s in scores.items()}
    summary = {name: {k: {"mean": float(v.mean()), "ci95": bootstrap_ci(v)} for k, v in m.items()}
               for name, m in per_q.items()}
    comparisons = {f"{name} vs bm25 | {metric}": paired_comparison(per_q["bm25"][metric], m[metric])
                   for name, m in per_q.items() if name != "bm25"
                   for metric in ("recall@1", "recall@5", "mrr@10")}
    for c, p in zip(comparisons.values(), holm([c["p_value"] for c in comparisons.values()])):
        c["p_holm"] = p

    pd.concat([pd.DataFrame({"qid": bench.qids, "lang": lang, "system": name, **m}) for name, m in per_q.items()]) \
        .to_parquet(REPORTS / f"per_query_{lang}.parquet", index=False)

    result = {"lang": lang, "n_questions": len(bench.questions), "n_chunks": len(bench.chunks),
              "dual_encoder": {"train_pairs": len(pairs), "epochs": EPOCHS, "train_sec": round(de_train_sec),
                               "loss_first": float(np.mean(de.losses[:20])), "loss_last": float(np.mean(de.losses[-20:]))},
              "reranker_train_questions": len(h_q),
              "metrics": summary, "comparisons_vs_bm25": comparisons,
              "chunking_equal_budget": chunking_study(lang, tok),
              "runtime_sec": round(time.time() - t0)}
    (REPORTS / f"results_{lang}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    de.pairs = None   # don't ship the training data inside the index
    with open(ARTIFACTS / f"search_{lang}.pkl", "wb") as f:
        pickle.dump({"chunks": bench.chunks, "bm25": systems["bm25"], "tfidf": systems["tfidf"],
                     "dual": de, "reranker": rr}, f)
    return result


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    for lang in sys.argv[1:] or ["zh", "en"]:
        r = run_language(lang)
        print(lang, {k: round(v["recall@1"]["mean"], 3) for k, v in r["metrics"].items()}, r["runtime_sec"], "s")

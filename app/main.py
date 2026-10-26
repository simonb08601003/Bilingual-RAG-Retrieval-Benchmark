"""Passage search API: BM25 + dual encoder, fused with RRF, reranked with LambdaRank."""
from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from bireval.rerank import candidate_features
from bireval.retrievers import RRFHybrid
from bireval.text import tokenizer

ARTIFACTS = Path(os.environ.get("ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
app = FastAPI(title="Bilingual passage search", version="1.0")


@lru_cache(maxsize=2)
def index(lang: str) -> dict:
    with open(ARTIFACTS / f"search_{lang}.pkl", "rb") as f:
        art = pickle.load(f)
    tok = tokenizer(lang)
    art["c_tokens"] = [set(tok(c)) for c in art["chunks"]]
    art["c_len"] = np.array([len(c) for c in art["chunks"]], float)
    return art


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(q: str = Query(min_length=1), lang: str = Query("zh", pattern="^(zh|en)$"),
           k: int = Query(5, ge=1, le=20), rerank: bool = True):
    try:
        art = index(lang)
    except FileNotFoundError:
        raise HTTPException(503, f"index for {lang} not built; run scripts/run_eval.py")
    sb, sd = art["bm25"].score([q]), art["dual"].score([q])
    rrf = RRFHybrid([art["bm25"], art["dual"]]).score([q], [sb, sd])
    if rerank:
        ss = {"bm25": sb, "dense": sd, "tfidf": art["tfidf"].score([q]), "rrf": rrf}
        cand, feats = candidate_features(ss, [set(tokenizer(lang)(q))], art["c_tokens"], art["c_len"])
        scores = art["reranker"].rescore(cand, feats, len(art["chunks"]))[0]
    else:
        scores = rrf[0]
    top = np.argsort(-scores)[:k]
    return {"query": q, "lang": lang, "results": [{"rank": r + 1, "chunk_id": int(i), "text": art["chunks"][i]}
                                                   for r, i in enumerate(top)]}

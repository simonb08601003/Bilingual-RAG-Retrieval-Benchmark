"""LightGBM LambdaRank over the top 30 candidates from rank fusion."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRanker

FEATURES = ["bm25", "bm25_rank", "dense", "dense_rank", "tfidf", "rrf", "coverage", "log_len"]


def _rank(s):
    return np.argsort(np.argsort(-s, axis=1), axis=1) + 1


def candidate_features(scores: dict[str, np.ndarray], q_tokens: list[set], c_tokens: list[set],
                       c_len: np.ndarray, k: int = 30):
    """Top-k candidates by RRF and their features: (ids [n_q, k], features [n_q, k, 8])."""
    cand = np.argsort(-scores["rrf"], axis=1)[:, :k]
    rows = np.arange(len(cand))[:, None]
    at = lambda m: m[rows, cand]  # noqa: E731
    coverage = np.array([[len(q & c_tokens[c]) / max(len(q), 1) for c in row] for q, row in zip(q_tokens, cand)])
    feats = np.stack([at(scores["bm25"]), np.log(at(_rank(scores["bm25"]))),
                      at(scores["dense"]), np.log(at(_rank(scores["dense"]))),
                      at(scores["tfidf"]), at(scores["rrf"]),
                      coverage, np.log1p(c_len[cand])], axis=-1).astype(np.float32)
    # raw scores aren't comparable across queries; standardise them within each query
    for f in (0, 2, 4, 5):
        x = feats[..., f]
        feats[..., f] = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
    return cand, feats


class Reranker:
    def __init__(self, seed: int = 0):
        self.model = LGBMRanker(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                random_state=seed, verbose=-1)

    def fit(self, cand, feats, gold: list[set[int]]):
        labels = np.array([[int(c in g) for c in row] for row, g in zip(cand, gold)])
        keep = labels.sum(axis=1) > 0      # a query with no gold in its top k teaches nothing
        self.model.fit(feats[keep].reshape(-1, feats.shape[-1]), labels[keep].ravel(),
                       group=[cand.shape[1]] * int(keep.sum()))
        return self

    def rescore(self, cand, feats, n_chunks: int) -> np.ndarray:
        """Full score matrix with the reranked candidates on top."""
        s = self.model.predict(feats.reshape(-1, feats.shape[-1])).reshape(cand.shape)
        out = np.full((len(cand), n_chunks), -1e9)
        np.put_along_axis(out, cand, s, axis=1)
        return out

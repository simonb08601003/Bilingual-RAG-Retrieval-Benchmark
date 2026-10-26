"""Per-question retrieval metrics and paired significance tests."""
from __future__ import annotations

import numpy as np
from scipy import stats


def gold_ranks(scores: np.ndarray, gold: list[set[int]]) -> np.ndarray:
    """Rank (from 1) of the best gold chunk per query. Ties go against us."""
    ranks = np.empty(len(gold), dtype=np.int64)
    for i, g in enumerate(gold):
        g_scores = scores[i, list(g)]
        best = g_scores.max()
        ranks[i] = int((scores[i] >= best).sum() - (g_scores >= best).sum() + 1)
    return ranks


def per_query(ranks: np.ndarray) -> dict[str, np.ndarray]:
    return {"recall@1": (ranks <= 1).astype(float), "recall@5": (ranks <= 5).astype(float),
            "recall@10": (ranks <= 10).astype(float), "mrr@10": np.where(ranks <= 10, 1.0 / ranks, 0.0)}


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparison(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """b - a on the same queries. McNemar for 0/1 metrics, sign-flip permutation otherwise."""
    d = b - a
    lo, hi = bootstrap_ci(d, n_boot, seed)
    if set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}:
        b_only, a_only = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
        p = stats.binomtest(b_only, b_only + a_only, 0.5).pvalue if b_only + a_only else 1.0
        test = {"test": "mcnemar_exact", "b_only": b_only, "a_only": a_only}
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_boot, len(d)))
        p = ((np.abs((signs * d).mean(axis=1)) >= abs(d.mean())).sum() + 1) / (n_boot + 1)
        test = {"test": "paired_permutation"}
    return {"diff": float(d.mean()), "ci95": [lo, hi], "p_value": float(p), **test}


def holm(p_values) -> list[float]:
    p = np.asarray(p_values, float)
    m = len(p)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()

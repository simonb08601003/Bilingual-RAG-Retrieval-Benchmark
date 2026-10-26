import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bireval.metrics import gold_ranks, holm, paired_comparison, per_query  # noqa: E402
from bireval.retrievers import BM25, DualEncoder, RRFHybrid  # noqa: E402
from bireval.text import en_words, zh_char_ngrams  # noqa: E402

DOCS = ["台達電子是電源管理與散熱解決方案的廠商", "台積電是全球最大的晶圓代工公司",
        "梵語是印度的古典語言", "the eiffel tower is in paris", "the golden gate bridge is in san francisco"]


def test_zh_char_ngrams():
    assert zh_char_ngrams("晶圓代工") == ["晶", "圓", "代", "工", "晶圓", "圓代", "代工"]


def test_bm25_finds_obvious_match():
    bm = BM25(lambda t: zh_char_ngrams(t) + en_words(t)).fit(DOCS)
    s = bm.score(["晶圓代工公司是哪一家", "where is the eiffel tower"])
    assert s[0].argmax() == 1 and s[1].argmax() == 3


def test_gold_rank_ties_are_pessimistic():
    scores = np.array([[0.5, 0.5, 0.1], [0.9, 0.2, 0.2]])
    assert gold_ranks(scores, [{1}, {0}]).tolist() == [2, 1]
    assert gold_ranks(scores, [{0, 1}, {1, 2}]).tolist() == [1, 2]


def test_per_query_mrr():
    m = per_query(np.array([1, 2, 11]))
    assert np.allclose(m["mrr@10"], [1, 0.5, 0])


def test_rrf_prefers_consensus():
    a = np.array([[3.0, 2.0, 1.0]])
    b = np.array([[1.0, 3.0, 2.0]])
    s = RRFHybrid([None, None], k=1).score(["q"], [a, b])
    assert s[0].argmax() == 1


def test_mcnemar_and_holm():
    a = np.array([1, 0, 0, 0, 1, 0] * 50, float)
    b = np.array([1, 1, 1, 0, 1, 1] * 50, float)
    r = paired_comparison(a, b)
    assert r["diff"] > 0 and r["p_value"] < 1e-10
    assert holm([0.01, 0.02]) == [0.02, 0.02]


def test_dual_encoder_learns_toy_task():
    pairs = [(f"question about topic {i}", f"passage on topic {i} details") for i in range(64)] * 8
    de = DualEncoder(en_words, pairs, dim=32, buckets=4096, epochs=10, batch=32, seed=0)
    de.fit([p for _, p in pairs[:64]])
    s = de.score([q for q, _ in pairs[:64]])
    assert (s.argmax(axis=1) == np.arange(64)).mean() > 0.9

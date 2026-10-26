"""Retrievers. Each has .fit(chunks) and .score(queries) -> [n_queries, n_chunks] array."""
from __future__ import annotations

import zlib

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


class BM25:
    def __init__(self, tokenize, k1: float = 1.2, b: float = 0.75, name: str = "bm25"):
        self.tok, self.k1, self.b, self.name = tokenize, k1, b, name

    def fit(self, chunks):
        self.vec = CountVectorizer(analyzer=self.tok)
        tf = self.vec.fit_transform(chunks).tocoo().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self

    def score(self, queries):
        q = self.vec.transform(queries)
        q.data[:] = 1.0     # count each query term once
        return (q @ self.W.T).toarray()


class TfidfCosine:
    def __init__(self, tokenize, name: str = "tfidf"):
        self.tok, self.name = tokenize, name

    def fit(self, chunks):
        self.vec = TfidfVectorizer(analyzer=self.tok, sublinear_tf=True)
        self.D = self.vec.fit_transform(chunks)
        return self

    def score(self, queries):
        return (self.vec.transform(queries) @ self.D.T).toarray()


class LSA:
    """TF-IDF reduced to 256 dimensions with truncated SVD."""

    def __init__(self, tokenize, dim: int = 256, name: str = "lsa"):
        self.tok, self.dim, self.name = tokenize, dim, name

    def fit(self, chunks):
        self.vec = TfidfVectorizer(analyzer=self.tok, sublinear_tf=True)
        X = self.vec.fit_transform(chunks)
        self.svd = TruncatedSVD(min(self.dim, X.shape[0] - 1), random_state=0)
        self.D = normalize(self.svd.fit_transform(X))
        return self

    def score(self, queries):
        return normalize(self.svd.transform(self.vec.transform(queries))) @ self.D.T


class SpacyMeanVectors:
    """Average of spaCy's 300-d word vectors. The cheapest dense baseline."""

    def __init__(self, lang: str, name: str = "spacy_vectors"):
        self.lang, self.name = lang, name

    def _embed(self, texts):
        out = np.zeros((len(texts), self.nlp.vocab.vectors.shape[1]), np.float32)
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=256)):
            vs = [t.vector for t in doc if t.has_vector and not t.is_punct and not t.is_stop]
            if vs:
                out[i] = np.mean(vs, axis=0)
        return normalize(out)

    def fit(self, chunks):
        import spacy
        model = "zh_core_web_md" if self.lang == "zh" else "en_core_web_md"
        self.nlp = spacy.load(model, exclude=["parser", "ner", "tagger", "attribute_ruler", "lemmatizer", "senter"])
        self.D = self._embed(chunks)
        return self

    def score(self, queries):
        return self._embed(queries) @ self.D.T


class Tower(torch.nn.Module):
    """Mean of hashed token embeddings, a residual MLP, then L2-normalise."""

    def __init__(self, buckets: int, dim: int):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(buckets, dim, mode="mean", sparse=True)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.GELU(), torch.nn.Linear(dim, dim))

    def forward(self, ids, offs):
        e = self.emb(ids, offs)
        return torch.nn.functional.normalize(e + self.mlp(e), dim=-1)


class DualEncoder:
    """Bi-encoder trained from scratch with in-batch negatives. Small enough for a CPU.

    Queries and passages share one tower; tokens are hashed into 2^18 buckets.
    """

    def __init__(self, tokenize, train_pairs, dim: int = 256, buckets: int = 2**18, epochs: int = 4,
                 batch: int = 256, lr: float = 2e-3, temperature: float = 0.05, seed: int = 0,
                 name: str = "dual_encoder"):
        self.tok, self.pairs, self.dim, self.buckets = tokenize, train_pairs, dim, buckets
        self.epochs, self.batch, self.lr, self.t, self.seed, self.name = epochs, batch, lr, temperature, seed, name

    def _token_ids(self, text):
        # crc32 rather than hash(), which changes between Python processes
        return [zlib.crc32(tok.encode()) % self.buckets for tok in self.tok(text)] or [0]

    @staticmethod
    def _batch(id_lists):
        offsets = np.cumsum([0] + [len(ids) for ids in id_lists[:-1]])
        return torch.tensor([i for ids in id_lists for i in ids]), torch.tensor(offsets)

    def train(self):
        torch.manual_seed(self.seed)
        self.net = Tower(self.buckets, self.dim)
        opt_sparse = torch.optim.SparseAdam(list(self.net.emb.parameters()), lr=self.lr)
        opt_dense = torch.optim.Adam(self.net.mlp.parameters(), lr=self.lr)
        # tokenise once up front; many questions share a paragraph
        para_index, para_ids, q_ids, p_of_q = {}, [], [], []
        for q, p in self.pairs:
            if p not in para_index:
                para_index[p] = len(para_ids)
                para_ids.append(self._token_ids(p))
            q_ids.append(self._token_ids(q))
            p_of_q.append(para_index[p])
        p_of_q = np.array(p_of_q)
        rng = np.random.default_rng(self.seed)
        self.losses = []
        for _ in range(self.epochs):
            order = rng.permutation(len(q_ids))
            for b in range(0, len(order) - self.batch + 1, self.batch):
                idx = order[b:b + self.batch]
                pidx = p_of_q[idx]
                q = self.net(*self._batch([q_ids[i] for i in idx]))
                p = self.net(*self._batch([para_ids[j] for j in pidx]))
                logits = q @ p.T / self.t
                # two questions on the same paragraph aren't negatives for each other
                same = torch.from_numpy(pidx[:, None] == pidx[None, :])
                logits = logits.masked_fill(same & ~torch.eye(len(idx), dtype=torch.bool), -1e4)
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(idx)))
                opt_sparse.zero_grad()
                opt_dense.zero_grad()
                loss.backward()
                opt_sparse.step()
                opt_dense.step()
                self.losses.append(loss.item())
        return self

    def _encode(self, texts):
        self.net.eval()
        with torch.no_grad():
            return np.vstack([self.net(*self._batch([self._token_ids(t) for t in texts[b:b + 512]])).numpy()
                              for b in range(0, len(texts), 512)])

    def fit(self, chunks):
        if not hasattr(self, "net"):
            self.train()
        self.D = self._encode(chunks)
        return self

    def score(self, queries):
        return self._encode(queries) @ self.D.T


class RRFHybrid:
    """Reciprocal rank fusion: sum of 1 / (k + rank) across retrievers."""

    def __init__(self, retrievers, k: int = 60, name: str | None = None):
        self.rs, self.k = retrievers, k
        self.name = name or "rrf(" + "+".join(getattr(r, "name", "?") for r in retrievers) + ")"

    def score(self, queries, scores: list[np.ndarray] | None = None):
        scores = scores or [r.score(queries) for r in self.rs]
        total = np.zeros_like(scores[0], dtype=np.float64)
        for s in scores:
            total += 1.0 / (self.k + np.argsort(np.argsort(-s, axis=1), axis=1) + 1)
        return total

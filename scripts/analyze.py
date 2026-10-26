"""Does the gap between systems depend on language? A GEE logistic model of hit@5
(questions as clusters), plus the README figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "reports"
SYSTEMS = ["bm25", "tfidf", "lsa", "spacy_vectors", "dual_encoder", "rrf_bm25_dual", "rrf_rerank"]
LABELS = {"bm25": "BM25", "bm25_words": "BM25 (word seg.)", "tfidf": "TF-IDF", "lsa": "LSA",
          "spacy_vectors": "spaCy vectors", "dual_encoder": "Dual encoder (PyTorch)",
          "rrf_bm25_dual": "RRF(BM25+dual)", "rrf_rerank": "RRF + LambdaRank"}

df = pd.concat([pd.read_parquet(R / f"per_query_{l}.parquet") for l in ("zh", "en")])
sub = df[df.system.isin(["bm25", "dual_encoder", "rrf_rerank"])].copy()
sub["hit5"] = sub["recall@5"]
sub["cluster"] = sub.lang + ":" + sub.qid.astype(str)
model = smf.gee("hit5 ~ C(system, Treatment('bm25')) * C(lang, Treatment('en'))", groups="cluster",
                data=sub, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
coef = pd.DataFrame({"coef": model.params, "se": model.bse, "p": model.pvalues})
coef["odds_ratio"] = np.exp(coef.coef)
coef.to_csv(R / "gee_hit5.csv")
(R / "gee_hit5.txt").write_text(str(model.summary()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, lang, title in zip(axes, ["zh", "en"], ["Traditional Chinese (DRCD)", "English (SQuAD v1.1)"]):
    res = json.loads((R / f"results_{lang}.json").read_text())["metrics"]
    names = [s for s in (["bm25", "bm25_words"] + SYSTEMS[1:]) if s in res]
    y = np.arange(len(names))
    for off, metric, color in ((-0.18, "recall@1", "#4C72B0"), (0.18, "recall@5", "#DD8452")):
        m = np.array([res[n][metric]["mean"] for n in names])
        lo = np.array([res[n][metric]["ci95"][0] for n in names])
        hi = np.array([res[n][metric]["ci95"][1] for n in names])
        ax.barh(y + off, m, 0.36, xerr=[m - lo, hi - m], color=color, label=metric.replace("recall", "Recall"))
    ax.set_yticks(y, [LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Share of questions whose gold passage is retrieved")
axes[1].legend(loc="center right")
fig.suptitle("Passage retrieval over 9k (zh) / 21k (en) passages, 95% bootstrap CIs")
fig.tight_layout()
fig.savefig(R / "figures" / "recall_by_system.png", dpi=150)
print(coef.round(4))

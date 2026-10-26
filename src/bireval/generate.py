"""Optional: answer from the retrieved passages with Claude, citing them. Not part of the benchmark.

python -m bireval.generate --lang zh "台達電子的總部在哪裡？"   (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse

SYSTEM = (
    "Answer the question using only the numbered passages. Cite every claim with the passage "
    "number in square brackets, e.g. [2]. If the passages do not contain the answer, say so. "
    "Answer in the language of the question."
)


def answer(question: str, passages: list[str], model: str = "claude-opus-5-5") -> dict:
    import anthropic

    client = anthropic.Anthropic()
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    response = client.beta.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"}],
    )
    if response.stop_reason == "refusal":
        return {"answer": None, "refused": True, "model": response.model}
    text = "".join(b.text for b in response.content if b.type == "text")
    return {"answer": text, "refused": False, "model": response.model}


def main():
    import pickle
    from pathlib import Path

    import numpy as np

    from .retrievers import RRFHybrid

    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    art = pickle.load(open(Path(__file__).resolve().parents[2] / "artifacts" / f"search_{a.lang}.pkl", "rb"))
    s = RRFHybrid([art["bm25"], art["dual"]]).score([a.question])
    top = np.argsort(-s[0])[: a.k]
    print(answer(a.question, [art["chunks"][i] for i in top]))


if __name__ == "__main__":
    main()

"""
RAG pipeline on the self-authored Aetherline corpus.

Pipeline stages (each one inspectable):
    1. Chunk: split documents into 1-3-sentence chunks.
    2. Embed: TF-IDF vectorize chunks (avoids any external embedding API).
    3. Retrieve: cosine similarity → top-k chunks per query.
    4. Generate: rule-based — pick the highest-scoring sentence in the
       top-k chunks; if max similarity is below a threshold, refuse.

Evaluation:
    - For in-corpus questions: retrieval recall@k (was the correct doc
      retrieved among top-k?) and answer correctness (does the chosen
      sentence come from the right doc?).
    - For out-of-corpus questions: refusal rate (did the threshold
      correctly say "I don't know"?).

We *deliberately* don't use a real LLM here. The point is to make every
RAG stage visible. With a good retriever and refusal threshold the
"generation" step is mostly stitching — the work is in retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from generate_data import DOCUMENTS, QUESTIONS

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"
COLOR_LIGHT_BLUE = "#9EB7D6"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 11,
})

CMAP_BLUE = LinearSegmentedColormap.from_list("blue_only", ["#FFFFFF", COLOR_BLUE])


# ---------------------------------------------------------- chunking ----
def split_sentences(text: str) -> list[str]:
    # Quick splitter — sufficient for our small corpus.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    title: str
    text: str


def make_chunks(documents: list[dict], sentences_per_chunk: int = 2) -> list[Chunk]:
    chunks: list[Chunk] = []
    cid = 0
    for d in documents:
        sents = split_sentences(d["text"])
        for i in range(0, len(sents), sentences_per_chunk):
            block = " ".join(sents[i:i + sentences_per_chunk])
            chunks.append(Chunk(chunk_id=cid, doc_id=d["doc_id"],
                                title=d["title"], text=block))
            cid += 1
    return chunks


# ------------------------------------------------------------- retrieval --
class Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english")
        self.X = self.vec.fit_transform([c.text for c in chunks])

    def query(self, q: str, k: int = 3) -> list[tuple[Chunk, float]]:
        v = self.vec.transform([q])
        scores = cosine_similarity(v, self.X)[0]
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


# ------------------------------------------------------------ generation --
def best_sentence_from_chunks(q: str, retrieved: list[tuple[Chunk, float]],
                              vec: TfidfVectorizer) -> tuple[str, str, float]:
    """
    Pick the single sentence from the top-k chunks that has the highest
    TF-IDF cosine to the question. Returns (sentence, source_doc_id, score).
    """
    candidates: list[tuple[str, str]] = []
    for chunk, _ in retrieved:
        for sent in split_sentences(chunk.text):
            candidates.append((sent, chunk.doc_id))
    if not candidates:
        return "", "", 0.0
    qv = vec.transform([q])
    sv = vec.transform([s for s, _ in candidates])
    scores = cosine_similarity(qv, sv)[0]
    best = int(np.argmax(scores))
    return candidates[best][0], candidates[best][1], float(scores[best])


def answer(q: str, retriever: Retriever, k: int = 3,
           refuse_threshold: float = 0.20) -> dict:
    retrieved = retriever.query(q, k=k)
    sent, src_doc, sent_score = best_sentence_from_chunks(q, retrieved, retriever.vec)
    top_score = retrieved[0][1] if retrieved else 0.0

    refused = top_score < refuse_threshold
    return {
        "question": q,
        "retrieved": [(c.doc_id, score) for c, score in retrieved],
        "top_score": top_score,
        "answer_sentence": sent,
        "answer_source_doc": src_doc,
        "answer_score": sent_score,
        "refused": refused,
    }


# ---------------------------------------------------------------- figures --
def fig_corpus_overview(chunks: list[Chunk], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.axis("off")
    rows = [["Doc id", "Title", "Sentences", "Sample chunk"]]
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for doc_id, cs in by_doc.items():
        sample = cs[0].text[:90] + ("…" if len(cs[0].text) > 90 else "")
        rows.append([doc_id, cs[0].title, str(len(cs)), sample])
    table = ax.table(cellText=rows, loc="center",
                     colWidths=[0.18, 0.20, 0.07, 0.55], cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.0, 1.5)
    for c in range(4):
        cell = table[0, c]
        cell.set_facecolor("#E5EAF2")
        cell.set_text_props(weight="bold", color=COLOR_TEXT)
    fig.suptitle(f"Self-authored corpus — {len(by_doc)} documents, "
                 f"{sum(len(v) for v in by_doc.values())} chunks",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_embedding_2d(chunks: list[Chunk], retriever: Retriever, out_path: Path) -> None:
    """Project the TF-IDF chunk vectors to 2-D with TruncatedSVD."""
    svd = TruncatedSVD(n_components=2, random_state=42)
    Z = svd.fit_transform(retriever.X)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    docs = sorted(set(c.doc_id for c in chunks))
    palette = [COLOR_BLUE, COLOR_RED, COLOR_GRAY, COLOR_LIGHT_BLUE,
               "#5A8FCC", "#C9A0A0", "#8FA8B5", "#A8B0BB"]
    for d, c in zip(docs, palette):
        idx = [i for i, ch in enumerate(chunks) if ch.doc_id == d]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=80, color=c, alpha=0.85,
                   edgecolor="white", linewidth=0.6, label=d)
    ax.set_xlabel("SVD axis 1"); ax.set_ylabel("SVD axis 2")
    ax.set_title("TF-IDF chunk embeddings projected to 2-D\n"
                 "(chunks from the same doc cluster together — what we want)")
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_retrieval_evaluation(results: list[dict], out_path: Path) -> tuple[dict, dict]:
    """Plot recall@1, recall@3, refusal-rate, hallucination-rate."""
    in_q = [r for r in results if not r["truth_out_of_corpus"]]
    out_q = [r for r in results if r["truth_out_of_corpus"]]

    recall_at_1 = float(np.mean([r["correct_top1_doc"] for r in in_q]))
    recall_at_3 = float(np.mean([r["correct_in_top3_docs"] for r in in_q]))
    in_correct = float(np.mean([r["answer_correct"] for r in in_q]))
    out_refusal = float(np.mean([r["refused"] for r in out_q])) if out_q else 0.0
    out_halluc = 1 - out_refusal

    metrics = {
        "Recall@1\n(in-corpus)": recall_at_1,
        "Recall@3\n(in-corpus)": recall_at_3,
        "Answer correctness\n(in-corpus)": in_correct,
        "Refusal rate\n(out-of-corpus)": out_refusal,
        "Hallucination rate\n(out-of-corpus)": out_halluc,
    }
    palette = [COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_GRAY, COLOR_RED]

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=palette, edgecolor=COLOR_LIGHT_GRAY, linewidth=0.8)
    ax.set_ylim(0, 1.10); ax.set_ylabel("Rate")
    ax.set_title("RAG pipeline evaluation")
    for bar, v in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=11,
                color=COLOR_TEXT, weight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "answer_correctness_in_corpus": in_correct,
        "refusal_rate_out_of_corpus": out_refusal,
        "hallucination_rate_out_of_corpus": out_halluc,
    }
    return summary, metrics


def fig_score_histogram(results: list[dict], threshold: float, out_path: Path) -> None:
    in_scores = [r["top_score"] for r in results if not r["truth_out_of_corpus"]]
    out_scores = [r["top_score"] for r in results if r["truth_out_of_corpus"]]

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    bins = np.linspace(0, max(max(in_scores, default=0), max(out_scores, default=0)) + 0.05, 18)
    ax.hist(in_scores, bins=bins, color=COLOR_BLUE, alpha=0.75,
            edgecolor="white", linewidth=0.6, label="In-corpus questions")
    ax.hist(out_scores, bins=bins, color=COLOR_RED, alpha=0.75,
            edgecolor="white", linewidth=0.6, label="Out-of-corpus questions")
    ax.axvline(threshold, color=COLOR_GRAY, linewidth=1.5, linestyle="--",
               label=f"refusal threshold = {threshold}")
    ax.set_xlabel("Top-1 retrieval cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution — the threshold needs to *separate* the two")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    chunks = make_chunks(DOCUMENTS)
    retriever = Retriever(chunks)

    REFUSE_THRESHOLD = 0.20
    K = 3

    results = []
    for q in QUESTIONS:
        out = answer(q["q"], retriever, k=K, refuse_threshold=REFUSE_THRESHOLD)
        retrieved_docs = [doc for doc, _ in out["retrieved"]]
        truth_doc = q["doc_id"]
        out["truth_doc_id"] = truth_doc
        out["truth_out_of_corpus"] = q["out_of_corpus"]
        out["correct_top1_doc"] = (truth_doc is not None
                                   and len(retrieved_docs) > 0
                                   and retrieved_docs[0] == truth_doc)
        out["correct_in_top3_docs"] = (truth_doc is not None
                                       and truth_doc in retrieved_docs)
        out["answer_correct"] = (truth_doc is not None
                                 and out["answer_source_doc"] == truth_doc
                                 and not out["refused"])
        results.append(out)

    print("\nPer-question results:")
    print(f"  {'Q':<70} {'expected':<22} {'retrieved-top1':<22} {'top_score':>9}  {'refused':>7}")
    for r in results:
        exp = r["truth_doc_id"] if r["truth_doc_id"] else "(out-of-corpus)"
        top1 = r["retrieved"][0][0] if r["retrieved"] else "—"
        print(f"  {r['question'][:68]:<70} {exp:<22} {top1:<22} "
              f"{r['top_score']:>9.3f}  {str(r['refused']):>7}")

    Path("results").mkdir(exist_ok=True)
    assets = Path("assets"); assets.mkdir(exist_ok=True)

    fig_corpus_overview(chunks, assets / "01_corpus.png")
    fig_embedding_2d(chunks, retriever, assets / "02_embeddings.png")
    summary, _ = fig_retrieval_evaluation(results, assets / "03_eval.png")
    fig_score_histogram(results, REFUSE_THRESHOLD, assets / "04_scores.png")

    with open("results/metrics.json", "w") as f:
        json.dump({
            "refuse_threshold": REFUSE_THRESHOLD,
            "k": K,
            **summary,
            "n_in_corpus_questions": int(sum(1 for r in results if not r["truth_out_of_corpus"])),
            "n_out_of_corpus_questions": int(sum(1 for r in results if r["truth_out_of_corpus"])),
        }, f, indent=2)

    print(f"\nSummary:")
    for k, v in summary.items():
        print(f"  {k:<35} {v:.3f}")
    print(f"\nFigures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()

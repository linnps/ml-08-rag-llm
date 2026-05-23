<div align="center">

# RAG — Retrieval-Augmented Generation on a Self-Authored Corpus

**A complete RAG pipeline (chunk → embed → retrieve → generate) over a fictional company's internal handbook, with adversarial out-of-corpus questions to measure hallucination.**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![data](https://img.shields.io/badge/data-self--authored-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

<p align="center">
  <a href="https://linnps.github.io/ml-08-rag-llm/"><img src="https://img.shields.io/badge/View_Live_Dashboard-0E2841?style=for-the-badge&amp;logo=githubpages&amp;logoColor=white" alt="View live dashboard"></a>
</p>

## At a glance

> Build a complete RAG pipeline over an invented corpus (the handbook of a fictional logistics company called *Aetherline*) and test it with **two kinds of questions**: ones whose answers are in the corpus, and ones whose answers *aren't*. The first set measures retrieval quality. The second set measures the system's willingness to refuse rather than hallucinate.

<p align="center">
  <img src="https://img.shields.io/badge/Recall%401-1.00-3B6EA8?style=for-the-badge" alt="Recall@1 1.00">
  <img src="https://img.shields.io/badge/Recall%403-1.00-3B6EA8?style=for-the-badge" alt="Recall@3 1.00">
  <img src="https://img.shields.io/badge/Out--of--corpus_refusal_rate-0.50-C04040?style=for-the-badge" alt="Out-of-corpus refusal rate 0.50">
</p>
<p align="center"><sub>Recall%40k &rarr; in-corpus retrieval (k = 3)&nbsp;&middot;&nbsp;refusal rate &rarr; fraction of unanswerable queries correctly refused&nbsp;&middot;&nbsp;50 % hallucination = the honest weak spot</sub></p>

<sub>**Headline finding:** the retriever is *perfect* on questions whose answers exist in the corpus. The interesting (and honest) result is the 50% hallucination rate on questions whose answers *don't* exist — TF-IDF cosine similarity isn't a sharp enough signal to reliably distinguish "no relevant content" from "marginally relevant content." This is the single most important production-RAG problem, and it's visible right here on a 14-question evaluation.</sub>

---

## Experimental setup

Everything below is fully deterministic: the corpus is hard-coded in `generate_data.py` (no random draws), so the pipeline produces identical numbers on every run with the pinned library versions.

### Corpus & question design

The corpus is the invented internal handbook of *Aetherline Logistics*, written entirely from scratch. Eight short documents cover distinct topics:

| Doc ID | Topic |
|---|---|
| `policy-vacation` | Vacation policy |
| `policy-remote` | Remote-work policy |
| `product-relayframe` | RelayFrame fleet-management product |
| `product-cargolens` | CargoLens shipment-visibility product |
| `people-leadership` | Leadership team |
| `procedure-incident` | Incident response |
| `procedure-onboarding` | New-hire onboarding |
| `policy-expense` | Expense reimbursement |

The 14 test questions are hand-authored in two distinct slices:

- **10 in-corpus questions** — each paired with an explicit `doc_id` whose text contains the answer. Ground-truth relevance is exact: one canonical document per question, no ambiguity.
- **4 out-of-corpus questions** — marked `out_of_corpus: True`; the correct system behaviour is to refuse. These are deliberately adversarial: their vocabulary overlaps with the corpus ("Aetherline", "CargoLens", "educational", "employees") even though no document contains the answer. This is what exercises the refusal threshold.

There is no random seed: the corpus and questions are static literals in the source, so the only stochasticity in the whole pipeline is TF-IDF weighting, which is fully determined by the (also static) text.

### Chunking

| Parameter | Value | Why |
|---|---|---|
| Strategy | Sentence-boundary splitting (`re.split` on `[.!?]\s+`) then grouped | Keeps sentences semantically coherent; avoids mid-sentence cuts |
| `sentences_per_chunk` | 2 | Gives chunks long enough to contain a full fact but short enough to score sharply against a short question |
| Overlap | 0 (no sliding window) | Sufficient for a small, well-structured corpus; production pipelines add overlap to handle cross-boundary facts |
| Resulting chunks | 17 | Across 8 documents (2–3 chunks per doc) |

### Embedding & retrieval

| Parameter | Value | Why |
|---|---|---|
| Vectorizer | `sklearn.feature_extraction.text.TfidfVectorizer` | Zero external API dependency; fast and interpretable; competitive on keyword-rich corpora |
| Vocabulary | Derived from the 17 chunks (fit on corpus) | Query vectors are projected into the same space via `vec.transform` |
| Stop words | English (sklearn default list) | Suppresses high-frequency function words that would inflate irrelevant similarity |
| Similarity metric | Cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) | Length-normalised; appropriate for sparse TF-IDF vectors |
| Index type | Dense matrix (no ANN index) | Brute-force is trivial for 17 chunks; an ANN index such as FAISS would only matter at ≥ 10 k+ chunks |
| Top-k (`K`) | 3 | Retrieves the top-3 chunks per query for recall@3 measurement; top-1 is the primary retrieval metric |

### Generation step (rule-based, not LLM)

There is **no LLM call**. Generation is fully deterministic: the best single sentence from the top-k retrieved chunks is selected by a second cosine-similarity pass (`best_sentence_from_chunks`), comparing each candidate sentence against the question. If `max(top-k similarity) < 0.20`, the system refuses instead of returning an answer.

| Parameter | Value | Why |
|---|---|---|
| Refusal threshold | 0.20 (TF-IDF cosine) | Empirically chosen; see Dashboard §4 for why it is insufficient to cleanly separate the two score distributions |
| Generation model | None — rule-based sentence extraction | Makes every retrieval failure directly visible; an LLM at this stage would paper over retrieval errors with fluent prose |

### Environment

`python ≥ 3.10` · `numpy ≥ 1.24` · `scikit-learn ≥ 1.3` · `matplotlib ≥ 3.7`

---

## Dashboard

### Retrieval & honesty scorecard

<table>
<tr>
  <th align="left">Group</th>
  <th>Metric</th>
  <th align="center">Value</th>
</tr>
<tr>
  <td rowspan="3"><b>In-corpus</b></td>
  <td>Recall@1</td>
  <td align="center"><img src="https://img.shields.io/badge/1.00-3B6EA8?style=flat-square" alt="1.00"></td>
</tr>
<tr>
  <td>Recall@3</td>
  <td align="center"><img src="https://img.shields.io/badge/1.00-3B6EA8?style=flat-square" alt="1.00"></td>
</tr>
<tr>
  <td>Answer correctness</td>
  <td align="center"><img src="https://img.shields.io/badge/1.00-3B6EA8?style=flat-square" alt="1.00"></td>
</tr>
<tr>
  <td rowspan="2"><b>Out-of-corpus</b></td>
  <td>Refusal rate</td>
  <td align="center"><img src="https://img.shields.io/badge/0.50-C04040?style=flat-square" alt="0.50"></td>
</tr>
<tr>
  <td>Hallucination rate</td>
  <td align="center"><img src="https://img.shields.io/badge/0.50-C04040?style=flat-square" alt="0.50"></td>
</tr>
</table>

<sub>Blue = good &middot; Red = attention &middot; In-corpus metrics: higher is better &middot; Hallucination rate: lower is better &middot; values from <code>results/metrics.json</code></sub>

### 1. The corpus

![corpus](assets/01_corpus.png)

8 short documents about the invented company *Aetherline Logistics* — vacation policy, remote-work policy, two products, leadership team, incident response, onboarding, expense policy. Together they yield ~17 chunks of 1–3 sentences each. Every fact is something we wrote ourselves, so we have **complete ground truth**: we know which doc answers each test question, and we know which test questions have *no* answer in the corpus at all.

### 2. TF-IDF chunk embeddings

![embeddings](assets/02_embeddings.png)

A 2-D SVD projection of the TF-IDF vectors. Chunks from the same document tend to land near each other, which is exactly what a good chunk-level embedding should produce. With only 17 chunks the structure is loose, but the qualitative groupings (`product-cargolens` and `product-relayframe` cluster on the right; `procedure-incident` and `people-leadership` cluster on the left) are visible.

### 3. RAG pipeline evaluation

![evaluation](assets/03_eval.png)

Five metrics on the same plot:

- **Recall@1 / Recall@3 (in-corpus)**: was the right document in the top-1 / top-3 retrieved? **100% / 100%**.
- **Answer correctness (in-corpus)**: did the extracted sentence come from the right doc? **100%**.
- **Refusal rate (out-of-corpus)**: did the system correctly say "I don't know" on questions whose answers aren't in the corpus? **50%**.
- **Hallucination rate (out-of-corpus)** = 1 − refusal rate. **50%**.

The bar chart frames the problem clearly: this RAG pipeline is excellent at *answering* but mediocre at *refusing*.

### 4. The score histogram — why hallucination happens

![scores](assets/04_scores.png)

This is the diagnostic that explains figure 3. Each top-1 retrieval similarity score is plotted, colored by whether the question was actually in the corpus or not. The vertical dashed line at 0.20 is the refusal threshold.

The key thing to notice: **the two distributions overlap**. Some out-of-corpus questions ("How many employees does Aetherline have?", "What's the discount for educational customers?") share enough vocabulary with the corpus (`Aetherline`, `educational`, `customers`, `CargoLens`) to score above the threshold. TF-IDF, by definition, can't tell that the *meaning* of those questions has no answer in the corpus.

The fix is **semantic embeddings** (sentence-transformers, OpenAI embeddings, etc.) — they map "How many employees" and "How many vehicles can RelayFrame support" further apart in vector space than TF-IDF does, because they encode meaning, not tokens.

---

## Validation methodology

This project has a synthetic-data superpower: because every question is hand-authored and every ground-truth answer document is explicitly labelled, we can measure **retrieval quality exactly** — no annotation approximation needed.

### Retrieval metrics

All metrics are computed over the 10 in-corpus questions (where a ground-truth relevant doc exists):

| Metric | Definition | Reads as |
|---|---|---|
| **Recall@1** | $\frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{top-1 doc} = \text{truth doc}_i]$ | Fraction of queries where the single best-retrieved chunk comes from the correct document. |
| **Recall@3** | $\frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{truth doc}_i \in \text{top-3 docs}]$ | Fraction of queries where the correct document appears anywhere in the top-3. Always ≥ Recall@1. |
| **Answer correctness** | $\frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{answer\_source\_doc} = \text{truth doc}_i \wedge \neg\text{refused}]$ | Did the extracted answer sentence come from the right document? A stricter check than Recall@1: the sentence-selection pass must also route to the correct source. |

And over the 4 out-of-corpus questions (where refusal is the correct action):

| Metric | Definition | Reads as |
|---|---|---|
| **Refusal rate** | $\frac{1}{M}\sum_{j=1}^{M} \mathbf{1}[\text{top\_score}_j < \tau]$ | Fraction of out-of-corpus questions correctly refused (top-1 cosine < threshold $\tau = 0.20$). |
| **Hallucination rate** | $1 - \text{refusal rate}$ | Fraction of out-of-corpus questions where the system answered instead of refusing — the "bad" outcome. |

There is no MRR or nDCG: with a single canonical answer document per question and a corpus of only 17 chunks, Recall@k is the appropriate and sufficient metric. MRR/nDCG add meaningful signal only when ground-truth relevance has graded values or multiple relevant documents.

### Full results

All numbers pulled directly from `results/metrics.json`:

| Slice | Metric | Value |
|---|---:|---:|
| In-corpus (10 questions) | Recall@1 | **1.000** |
| In-corpus (10 questions) | Recall@3 | **1.000** |
| In-corpus (10 questions) | Answer correctness | **1.000** |
| Out-of-corpus (4 questions) | Refusal rate | 0.500 |
| Out-of-corpus (4 questions) | Hallucination rate | **0.500** |

Pipeline parameters at which these numbers were produced: `k = 3`, `refuse_threshold = 0.20`.

### Reproducibility & determinism

The corpus and questions are static literals in `generate_data.py` — there are no random draws. TF-IDF weighting is fully determined by the corpus text. Running `python generate_data.py && python train.py` produces identical `results/metrics.json` on every machine with the pinned library versions: no seed is needed because there is no randomness.

---

## What's actually happening

### Pipeline stages (each one inspectable in the code)

```
Documents
  │  split_sentences + group into ~2-sentence chunks
  ▼
Chunks  ── TF-IDF vectorize ──▶  X (chunks × terms)
                                  │
Question  ── TF-IDF transform ──▶ q (1 × terms)
                                  │
                       cosine similarity → top-k chunks
                                  │
       Best sentence in top-k by question similarity → answer
       If max(top-k similarity) < refusal_threshold  → refuse
```

The "generation" step is intentionally rule-based. A real RAG would call an LLM at this stage with the retrieved chunks as context. We deliberately don't, for two reasons: (a) the cost / dependency footprint, and (b) the LLM step *masks* retrieval problems — when retrieval fails, the LLM gracefully bullshits over it. By using a deterministic generator, we make every retrieval failure visible.

### Why TF-IDF is a reasonable starting point

For small corpora and clear vocabulary overlap between questions and answers, TF-IDF is competitive with sentence-transformer embeddings on retrieval quality. It's deterministic, fast, has no external dependency, and produces interpretable scores. The point at which TF-IDF starts to hurt is exactly where this project shows it: **detecting questions whose answers aren't in the corpus**. There, semantic embeddings win.

### The refusal-threshold dilemma

A higher threshold → more refusals → lower hallucination but missed in-corpus answers.
A lower threshold → more attempted answers → higher recall but more hallucination.

Looking at the histogram: there is **no clean threshold** that separates the two distributions on TF-IDF scores. That's *not* a bug — it's the empirical evidence that this signal is too weak to be the *sole* basis of a refusal decision. Production RAG systems combine retrieval score with semantic similarity, perplexity of the generated answer, and explicit uncertainty estimation.

### Mental model

| Stage | This project | Production upgrade |
|---|---|---|
| Chunking | Fixed 2-sentence | Semantic chunking (paragraph or topic boundaries) |
| Embedding | TF-IDF | Sentence-transformers / OpenAI / Voyage embeddings |
| Retrieval | Cosine similarity | Same + MMR for diversity, hybrid with BM25 |
| Generation | Pick best sentence | LLM with retrieved chunks as context |
| Refusal | Single threshold on score | Multi-signal: retrieval score + answer perplexity + LLM-judge |

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py    # writes the corpus + questions to data/
python train.py            # build retriever, run evaluation, render figures
```

Wall-time: ~3 seconds. There's no model training — this is pure information retrieval.

### Tweak the difficulty

Edit the corpus or question list directly in [`generate_data.py`](generate_data.py):

```python
QUESTIONS = [
    {"q": "How many vacation days do full-time staff get per year?",
     "doc_id": "policy-vacation", "out_of_corpus": False},
    ...
    {"q": "What is Aetherline's stock ticker?",
     "doc_id": None, "out_of_corpus": True},   # adversarial: no answer in corpus
]
```

Add more out-of-corpus questions whose vocabulary heavily overlaps with the corpus (e.g., "How many CargoLens shipments are tracked monthly?" — *CargoLens shipments* is in the corpus, but the *number* isn't) and watch the hallucination rate climb above 50%.

---

## Project layout

```
08-rag-llm/
├── README.md              ← this dashboard
├── requirements.txt
├── generate_data.py       ← self-authored corpus + question set
├── train.py               ← chunking, retrieval, generation, evaluation, figures
├── assets/                ← 4 dashboard PNGs
└── results/metrics.json
```

---

## Notes on methodology & limitations

Stated plainly so a reader can judge what these numbers do and don't support:

- **Synthetic corpus makes retrieval artificially easy.** Every document covers exactly one topic; questions are authored to have a single canonical answer. Real corpora contain overlapping facts, ambiguous phrasings, and partial answers spread across multiple documents — all of which would degrade Recall@1 below 100% even with a much more powerful retriever.
- **TF-IDF dominates the results, not "RAG design".** Every metric is a direct consequence of whether keyword overlap is sufficient to distinguish the relevant chunk. Swapping to sentence-transformer embeddings (e.g. `sentence-transformers/all-MiniLM-L6-v2`) would likely push recall to 100% and refusal rate well above 50% — without changing any other part of the pipeline. This makes the architecture comparison between TF-IDF and semantic embeddings the most meaningful follow-on experiment.
- **No re-ranking.** The pipeline selects the top-k chunks purely by similarity score and then picks the best sentence from them by a second cosine pass. A production system would apply a cross-encoder re-ranker (e.g. a bi-encoder shortlist → cross-encoder rerank) to improve ordering within the top-k, especially for longer or more ambiguous queries.
- **Chunking is naive (non-overlapping fixed-length by sentence count).** The `sentences_per_chunk = 2` strategy with zero overlap can split a single fact across two chunks. For a 17-chunk corpus the risk is low, but on longer documents overlap (e.g. 1-sentence stride) is standard practice to avoid boundary-straddling facts falling between chunks.
- **Metrics measure retrieval quality, not factual correctness.** "Answer correctness" checks whether the extracted sentence's source document matches the labelled truth document — it does not verify that the sentence actually contains the right answer. A sentence from the correct document that does not address the question would still score 1. This is a labelling artefact of using `doc_id` as a proxy for answer correctness rather than annotating at the sentence level.

---

## What I learned

- **The corpus is the project.** Spending an hour writing a careful synthetic corpus — with deliberately adversarial out-of-corpus questions whose vocabulary overlaps with in-corpus content — produced a much sharper diagnostic than throwing a generic Q&A benchmark at the same pipeline.
- **Hallucination on out-of-corpus questions is the headline metric.** In-corpus retrieval being 100% looks great until you discover the system happily *also* answers questions whose answers don't exist. A RAG system that refuses appropriately is harder to build than one that retrieves well.
- **TF-IDF gets you 80% of the way there for free.** It's a real production-grade retriever for any corpus where keyword overlap is meaningful. The 20% it misses is exactly the cases where you need semantic embeddings — and the score histogram visualizes which 20% with one chart.
- **Building a "fake LLM" forces honesty.** Replacing the LLM with a rule that just picks the best-matching sentence reveals retrieval failures that an LLM would otherwise paper over with fluent-sounding prose. For evaluation purposes, the deterministic generator is more useful than a real LLM.

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Corpus is fully synthetic and self-authored.</sub>
</div>

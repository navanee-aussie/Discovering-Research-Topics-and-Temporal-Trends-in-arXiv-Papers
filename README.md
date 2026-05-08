# Discovering Research Topics and Temporal Trends in arXiv Papers


---

## What This Project Does

Applies unsupervised machine learning to 2.8 million arXiv research papers to automatically discover research topics and track how they have grown or declined over time.

Two topic modelling approaches are compared:
- **LDA** (baseline) — bag-of-words, K=60 topics, u_mass coherence = -4.49
- **BERTopic** — transformer embeddings + HDBSCAN clustering, 50 topics, u_mass coherence = -2.60 (~42% improvement over LDA), topic diversity = 0.784

Temporal trends are measured per topic using Spearman correlation across yearly publication share, identifying statistically significant rising and declining research areas.

---

## Key Results

**Declining topics (all p < 0.001):** Quantum Chromodynamics, High-Energy Particle Physics, Condensed Matter & Magnetism, Theoretical Physics & String Theory, Galaxy Formation & Cosmology

These reflect a measurable shift in research volume away from classical physics domains toward newer computational and ML-adjacent fields over the arXiv publication history.

---

## Pipeline Overview

```
Raw arXiv JSON (2.8M papers)
        ↓
Stratified sampling by year (~10,000 records)
        ↓
Text cleaning — LaTeX, HTML, math symbol removal + language detection
        ↓
Quality filtering — year validation, abstract length, withdrawn papers
        ↓
LDA baseline — CountVectorizer + Gensim, coherence sweep over K
        ↓
BERTopic — SentenceTransformer embeddings → UMAP → HDBSCAN → KeyBERT labels
        ↓
Temporal trend analysis — per-topic yearly share + Spearman correlation
        ↓
Visualisations — rising/declining topic bar charts, coherence comparison
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Topic modelling | `bertopic==0.16.0`, `gensim` |
| Embeddings | `sentence-transformers` |
| Dimensionality reduction | `umap-learn` |
| Clustering | `hdbscan==0.8.38` |
| ML utilities | `scikit-learn==1.4.2` |
| Data processing | `pandas`, `numpy` |
| Visualisation | `matplotlib` |
| Language detection | `langdetect` |

---

## Getting Started

### 1. Get the Data

Download the arXiv metadata snapshot from Kaggle:  
[https://www.kaggle.com/datasets/Cornell-University/arxiv/data](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)

File needed: `arxiv-metadata-oai-snapshot.json` (~4GB, not included in this repo)

Place it in the same directory as the notebook, then update this line in the notebook:

```python
# Change this to your local path
file = "arxiv-metadata-oai-snapshot.json"
```

### 2. Install Dependencies

```bash
pip install bertopic==0.16.0 hdbscan sentence-transformers umap-learn gensim scikit-learn pandas numpy matplotlib langdetect
```

### 3. Run

Open `group_29.ipynb` in Jupyter and run cells sequentially.

> Note: BERTopic with SentenceTransformer embeddings on 10,000 records requires ~4–8GB RAM. GPU not required but speeds up embedding generation.

---

## Repository Structure

```
├── group_29.ipynb          # Full pipeline — data loading to trend analysis
├── README.md
└── (data not included — see Getting Started above)
```

---

## Notes

- The notebook uses stratified sampling (300 papers/year) from the full 2.8M record dataset to keep compute manageable while maintaining temporal coverage from 1991 to present.
- LDA coherence sweep tested K = 10 to 100; K=60 gave best u_mass score.
- BERTopic topic count (50) was determined by HDBSCAN's density-based clustering, not set manually.

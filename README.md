# Netflix catalogue clustering

**Question:** can 7,777 Netflix titles be sorted into meaningful groups from
their text metadata alone — description, genre, cast, director — with no labels
to learn from?

**Answer: not cleanly, and that is the finding.** The best K-Means silhouette
across K=2…10 is **0.077**, where 0 means no cluster structure at all. The
notebooks are kept here as the evidence for that, rather than tuned until the
number flattered the method.

## Method

- 7,787 titles from the Netflix Movies and TV Shows dataset (the standard
  Kaggle set — `show_id`, `type`, `title`, `director`, `cast`, `country`,
  `date_added`, `release_year`, `rating`, `duration`, `listed_in`,
  `description`). 7,777 remain after cleaning.
- `description` + `listed_in` + `cast` + `director` collapsed into one text
  field, lowercased, stopworded and lemmatised (WordNet — lemmatisation over
  stemming, to keep real words for interpretability).
- **TF-IDF**, `max_features=5000`, `min_df=5`, `max_df=0.95` → a 7777 × 5000
  matrix that is **99.4% sparse** (228,723 non-zero entries).
- **TruncatedSVD** → 200 components, chosen over PCA because it operates
  directly on the sparse matrix. Those 200 components retain **27.5%** of
  variance.
- **K-Means**, **Agglomerative** and **DBSCAN**, compared on silhouette,
  Calinski–Harabasz and Davies–Bouldin.

## Results

| Model | Silhouette | Shape of the result |
| --- | --- | --- |
| K-Means, K=2 | **0.077** | best K over 2–10; split 6214 / 1563 |
| K-Means, grid searched | 0.077 | no configuration moved it at all |
| Agglomerative, Ward, K=2 | 0.068 | split 6132 / 1645 |
| Agglomerative, average linkage, K=2 | 0.256 | see the caveat below |
| DBSCAN, eps=0.52, min_samples=5 | 0.163 | 6 clusters — but 7,497 of 7,777 in one of them |
| DBSCAN, grid searched, eps=0.63 | 0.211 | collapses to 2 clusters, 0% noise |

Supporting metrics for K-Means at K=2: Calinski–Harabasz 386.06,
Davies–Bouldin 3.58. A Davies–Bouldin near 3.6 is itself a statement that the
clusters overlap heavily.

### The caveat is the interesting part

The two highest silhouettes in that table — average-linkage Agglomerative at
0.256 and grid-searched DBSCAN at 0.211 — **are not better clusterings.** Both
are the same artefact: on sparse, high-dimensional text, average/complete
linkage and a generous DBSCAN `eps` both tend to put nearly every point in one
giant cluster and leave a handful of specks outside it. Silhouette rewards that
heavily, because almost every point is comfortably inside its own enormous
cluster and far from the tiny ones.

DBSCAN makes the mechanism visible: at eps=0.52 it reports 6 clusters with a
silhouette of 0.163, but the distribution is 7497 / 224 noise / 7 / 13 / 14 /
15 / 7. That is not six content categories. It is one blob and five rounding
errors.

So the honest ranking is that **K-Means at K=2 is the only result that is
actually a partition of the data**, and it scores 0.077 — near zero. Chasing
the higher silhouette would have meant reporting a one-cluster model as a
success.

### What surprised me

That K=2 won the silhouette sweep at all. I expected genre structure to produce
something like 8–12 clusters. Instead the strongest split the text supports is
essentially Movies vs TV Shows — which the dataset already tells you in a
column, without any of this.

The 27.5% explained variance at 200 components points the same way: Netflix
descriptions are short, formulaic marketing copy, and TF-IDF over them carries
much less separable signal than the genre labels imply.

## What I would do differently

- Cluster on **embeddings** (sentence-transformers) rather than TF-IDF —
  descriptions are short, and bag-of-words throws away exactly the semantics
  that would separate them.
- Report cluster **stability** across seeds and subsamples, not just silhouette
  on one fit. A near-zero silhouette with unstable membership is a much clearer
  negative result than either number alone.
- Treat `listed_in` as a weak label and measure cluster purity against it,
  instead of evaluating in a vacuum.
- Drop the 80/20 split. It is in the notebook, but for unsupervised clustering
  on the full dataset it does no work.

## Known issues in the notebooks

Kept deliberately, so the record matches what was actually run:

- Several of the templated commentary cells in `02-clustering.ipynb` state
  conclusions that the cell outputs directly contradict — that K-Means scored
  highest, that `k-means++` beat random initialisation, that Ward linkage
  performed best. None of those hold against the printed numbers. The tables
  above supersede them; the cells are left in place rather than quietly edited,
  since the outputs are the evidence.
- The final comparison chart omits DBSCAN and both tuned variants.

## Running it

```bash
git clone https://github.com/Deadsunx/Netflix-project.git
cd Netflix-project
pip install pandas numpy matplotlib seaborn scikit-learn nltk scipy jupyter
jupyter notebook
```

Both notebooks fetch the dataset over HTTP at run time, so there is no local
CSV to supply — `01-eda.ipynb` first, then `02-clustering.ipynb`.

| File | What |
| --- | --- |
| `01-eda.ipynb` | data wrangling and exploratory analysis |
| `02-clustering.ipynb` | text preprocessing, TF-IDF, SVD, the three models |
| `tfidf_vectorizer.pkl` | fitted vectoriser |
| `svd_model.pkl` | fitted 200-component TruncatedSVD |
| `kmeans_model.pkl` | fitted K-Means, K=2 |

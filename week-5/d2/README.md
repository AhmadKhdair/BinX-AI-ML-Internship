# DBSCAN and Hierarchical Clustering

Day 2 of Week 5 extends the clustering work from K-Means to methods that handle different data structures and assumptions.

The notebook compares **K-Means, DBSCAN, and Agglomerative Hierarchical Clustering** on the same standardized dataset. The main focus is understanding why the methods produce different partitions, how DBSCAN handles density and noise, and how hierarchical clustering builds a merge structure through linkage and a dendrogram.

## Notebook

`02_dbscan_hierarchical_clustering.ipynb`

## Core Idea

K-Means assigns every observation to the nearest centroid. This is effective when groups are compact and can be represented well by a center, but it is less suitable for non-convex structures and does not provide a noise label.

DBSCAN approaches the problem differently. It searches for **dense connected regions** using a neighborhood radius and a minimum density requirement. Agglomerative clustering builds a hierarchy instead, starting with one cluster per observation and repeatedly merging the closest clusters according to a linkage rule.

The notebook uses all three methods so the final recommendation is based on the geometry of the data rather than on one clustering algorithm in isolation.

## What I Covered

The notebook follows this workflow:

1. generate a two-moons dataset and add a small number of isolated points;
2. inspect the dataset and confirm that no values are missing;
3. standardize the features with `StandardScaler`;
4. fit K-Means with two clusters as a baseline;
5. explain why centroid-based clustering is limited on curved groups and isolated points;
6. define DBSCAN core, border, and noise points;
7. inspect a k-distance plot before selecting `eps`;
8. compare several `eps` values using cluster count, noise count, and silhouette score;
9. fit the final DBSCAN model and inspect its core, border, and noise samples;
10. build an agglomerative hierarchy using Ward linkage;
11. visualize the hierarchy with a dendrogram;
12. select a cut height and obtain the final hierarchical clusters;
13. compare K-Means, DBSCAN, and hierarchical clustering on the same data;
14. recommend the method that best matches the dataset structure.

## Dataset

The experiment uses a synthetic **two-moons** dataset generated with Scikit-learn:

```python
make_moons(n_samples=400, noise=0.07, random_state=42)
```

Twelve additional points are sampled around the main data region to introduce sparse observations.

Final dataset size:

| Property | Value |
|---|---:|
| Samples | 412 |
| Features | 2 |
| Missing values | 0 |

The curved shape is intentional. It makes the difference between centroid-based and density-based clustering easier to inspect directly.

## Scaling

The two features are standardized before clustering:

```text
Original features
        ↓
StandardScaler
        ↓
Scaled feature space
        ↓
K-Means / DBSCAN / Hierarchical Clustering
```

This keeps the distance calculations from being dominated by feature scale and ensures that the three clustering methods are compared on the same representation.

## K-Means Baseline

K-Means is fitted with:

```python
n_clusters = 2
random_state = 42
n_init = 10
```

The result provides a useful baseline, but the partition is determined by distance to the two centroids. The two moons are curved and non-convex, so a nearest-centroid split does not follow their natural connected structure.

K-Means also assigns every point to one of the two clusters, including sparse points that may not naturally belong to either group.

## DBSCAN

DBSCAN defines clusters through local density instead of centroids.

The two main parameters are:

- `eps`: maximum radius used to define a neighborhood;
- `min_samples`: minimum number of samples required in that neighborhood for a point to be considered a core point.

The point types used in the notebook are:

- **Core point** — satisfies the density condition and can expand a cluster;
- **Border point** — belongs to the neighborhood of a core point but cannot expand the cluster itself;
- **Noise point** — is not reached from any dense region and receives label `-1`.

A cluster expands through connected core points. Border points can join the cluster, while noise points remain outside the discovered dense regions.

## Choosing `eps`

I use `min_samples = 5` and inspect the sorted distance to each sample's fifth nearest neighbor.

The k-distance plot is used as a practical starting point for the `eps` search rather than selecting a value arbitrarily. I then compare a small range of candidate values:

| `eps` | Clusters | Noise Points | Silhouette* |
|---:|---:|---:|---:|
| 0.18 | 4 | 14 | -0.0131 |
| 0.20 | 3 | 9 | 0.0673 |
| 0.22 | 2 | 9 | 0.3776 |
| 0.25 | 2 | 8 | 0.3777 |
| 0.30 | 2 | 8 | 0.3777 |

\*For DBSCAN, the silhouette score is calculated after excluding samples labeled as noise.

The result becomes stable from approximately `eps=0.22` onward. I use:

```python
eps = 0.25
min_samples = 5
```

This produces two dense clusters without forcing all sparse observations into a group.

## Final DBSCAN Result

The selected DBSCAN model produces:

| Point Type | Count |
|---|---:|
| Core | 397 |
| Border | 7 |
| Noise | 8 |

Final number of clusters: **2**.

The important difference from K-Means is not only the cluster count. DBSCAN follows the connected curved regions and explicitly separates sparse observations as noise.

## Agglomerative Hierarchical Clustering

Agglomerative clustering starts with every observation as its own cluster and repeatedly merges the closest clusters until one hierarchy remains.

After a merge, the new cluster is not treated as a new Euclidean point. A **linkage rule** defines the distance between clusters:

- **Single linkage** — minimum pairwise distance;
- **Complete linkage** — maximum pairwise distance;
- **Average linkage** — average cross-cluster pairwise distance;
- **Ward linkage** — chooses the merge that causes the smallest increase in within-cluster variance.

For example, if `P3` and `P6` are merged, single linkage would define the distance from the new cluster to `P1` as:

```text
d({P3, P6}, P1) = min(d(P3, P1), d(P6, P1))
```

This is why the distance matrix must be updated after every merge: the meaning of distance between clusters depends on the selected linkage rule.

## Dendrogram

The notebook uses:

```python
linkage(X_scaled, method="ward")
```

The dendrogram records the hierarchy of merges and the linkage distance at which each merge occurs.

To obtain two final clusters, the notebook places the cut between the last two merge levels:

| Quantity | Value |
|---|---:|
| Previous merge height | 15.407 |
| Final merge height | 28.533 |
| Selected cut height | 21.970 |
| Resulting clusters | 2 |

The dendrogram is truncated to its upper structure so the important merge pattern remains readable with more than 400 observations.

## Method Comparison

The final comparison is:

| Method | Clusters | Noise Points | Silhouette |
|---|---:|---:|---:|
| K-Means | 2 | 0 | 0.4840 |
| DBSCAN | 2 | 8 | 0.3777 |
| Hierarchical (Ward) | 2 | 0 | 0.4328 |

The silhouette score is useful, but it is not enough to select the method by itself.

K-Means has the highest silhouette score in this experiment, yet its partition does not follow the two curved groups as naturally. This is an important limitation of relying on one internal clustering metric: the metric measures compactness and separation under its distance definition, not whether the discovered structure matches the geometry that matters for the problem.

## Method Recommendation

For this dataset, **DBSCAN is the most suitable method**.

The decision is based mainly on the structure of the data:

- the two groups are curved and non-convex;
- the clusters are connected through local density rather than around one center;
- several observations are sparse and should not be forced into a cluster;
- DBSCAN keeps the two main dense regions separate while identifying eight samples as noise.

K-Means remains a simple and efficient baseline when clusters are compact and `k` is approximately known.

Hierarchical clustering is useful when the merge hierarchy itself is important or when the data should be inspected at multiple clustering levels. With Ward linkage, however, the method still favors compact variance-based groups, which is less suitable for the moon-shaped structure used here.

## Main Takeaway

The main comparison can be summarized as:

```text
K-Means
    → nearest centroid
    → compact / roughly spherical groups
    → requires k
    → assigns every sample

DBSCAN
    → local density
    → irregular connected shapes
    → no predefined cluster count
    → can identify noise

Hierarchical Clustering
    → repeated cluster merges
    → linkage defines cluster distance
    → dendrogram exposes nested structure
    → final clusters depend on the selected cut
```

The key lesson from Day 2 is that clustering method selection should follow the **shape, density, noise level, and structure of the data**, not only the value of a single evaluation metric.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- Jupyter Notebook or JupyterLab

Install the required packages with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

## Running the Notebook

From the directory containing the notebook, start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
02_dbscan_hierarchical_clustering.ipynb
```

Run the notebook from top to bottom so the scaling, parameter comparison, final clustering, dendrogram, and method comparison are produced in order.

# K-Means Clustering and Choosing k

Day 1 of Week 5 focuses on unsupervised learning through K-Means clustering and on choosing a reasonable number of clusters instead of setting `k` arbitrarily.

## Notebook

`01_kmeans_clustering.ipynb`

## Core Idea

Supervised learning learns against a known target. K-Means is used here as an unsupervised method: it receives only the feature matrix and searches for groups in the feature space.

For a fixed `k`, the algorithm repeatedly assigns each observation to its nearest centroid and then moves each centroid to the mean of the observations assigned to it.

## What I Covered

The notebook follows this workflow:

1. load a numeric dataset without using its class target;
2. inspect the feature ranges and missing values;
3. standardize all input features with `StandardScaler`;
4. fit K-Means for `k=1` through `k=10`;
5. plot inertia and inspect the elbow;
6. compare `k=2` and `k=3` using the silhouette score;
7. fit the final K-Means model with the selected `k`;
8. inspect cluster sizes and centroids;
9. summarize the main feature profile of each cluster;
10. visualize the final clusters on a two-dimensional scatter plot.

## Dataset

The experiment uses the Wine dataset available directly through Scikit-learn.

Only the 13 numeric feature columns are used for clustering. The class target included with the dataset is intentionally excluded from the K-Means workflow.

This keeps the experiment unsupervised while avoiding an additional external dataset dependency.

## Scaling

The original features have very different numeric ranges. Since K-Means assigns observations using distance, the dataset is standardized before clustering:

```text
Original numeric features
        ↓
StandardScaler
        ↓
Scaled feature space
        ↓
K-Means
```

The scaled columns have means close to zero and standard deviations close to one.

## Choosing the Number of Clusters

### Elbow Method

K-Means was fitted for `k=1` through `k=10`.

The inertia drops quickly for the first few values of `k`, then the rate of improvement becomes smaller. The elbow region is around `k=3`, with `k=2` also kept as a reasonable nearby candidate.

### Silhouette Score

The two candidates were compared directly:

| k | Silhouette Score |
|---:|---:|
| 2 | 0.2593 |
| 3 | 0.2849 |

`k=3` produced the stronger silhouette score and was consistent with the elbow region, so it was selected for the final model.

The silhouette score is not an accuracy percentage. It measures how well observations fit within their assigned clusters relative to nearby clusters.

## Final Clustering

The final K-Means model uses:

```text
n_clusters = 3
random_state = 42
n_init = 10
```

Final cluster sizes:

| Cluster | Samples |
|---:|---:|
| 0 | 65 |
| 1 | 51 |
| 2 | 62 |

The final inertia is approximately `1277.93`, and the silhouette score is approximately `0.2849`.

## Cluster Interpretation

The centroids were transformed back to the original feature units before interpretation.

A smaller set of features makes the main differences easier to see:

| Cluster | Alcohol | Malic Acid | Flavanoids | Color Intensity | Hue | Proline |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.25 | 1.90 | 2.05 | 2.97 | 1.06 | 510.17 |
| 1 | 13.13 | 3.31 | 0.82 | 7.23 | 0.69 | 619.06 |
| 2 | 13.68 | 2.00 | 3.00 | 5.45 | 1.07 | 1100.23 |

The main patterns are:

- Cluster 0 has the lowest average alcohol, color intensity, and proline.
- Cluster 1 has the highest average malic acid and color intensity, with the lowest flavanoids and hue.
- Cluster 2 has the highest average alcohol, flavanoids, and proline.

The cluster numbers are identifiers only; they do not represent an ordered ranking.

## Visualization

The final scatter plot uses `flavanoids` and `proline` as the two displayed axes.

K-Means itself is fitted on all 13 standardized features. The two-dimensional plot is only used to inspect part of the resulting structure visually.

## Main Takeaway

The main result is not just fitting K-Means. The important part is the selection process:

```text
Scale the features
        ↓
Compare inertia across k
        ↓
Find the elbow region
        ↓
Compare candidate k values with silhouette score
        ↓
Fit the final model
        ↓
Interpret the clusters
```

For this experiment, both the elbow analysis and the silhouette comparison support using three clusters.

## Requirements

- Python 3
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook or JupyterLab

Install the required packages with:

```bash
pip install pandas matplotlib scikit-learn jupyter
```

## Running the Notebook

From the repository environment:

```bash
jupyter notebook
```

Then open:

```text
week-5/d1/01_kmeans_clustering.ipynb
```

Run the notebook from top to bottom so the scaling, candidate comparison, final model, and interpretation are produced in order.

# Week 5 — Day 2: DBSCAN & Hierarchical Clustering

This notebook completes the Day 2 clustering requirements by comparing K-Means, DBSCAN, and agglomerative hierarchical clustering on the same scaled dataset.

## Covered

- K-Means limitations
- DBSCAN intuition
- core, border, and noise points
- `eps` and `min_samples`
- k-distance plot
- DBSCAN parameter sensitivity
- DBSCAN cluster and noise counts
- agglomerative hierarchical clustering
- distance-matrix updates after a merge
- single, complete, average, and Ward linkage
- dendrogram construction
- cut-height selection
- cluster comparison with silhouette scores
- final method recommendation

## Dataset

The notebook uses a two-moons dataset with a few added outlier points.  
The curved structure makes the behavior of the three clustering methods easy to compare.

## Main Result

DBSCAN is the strongest fit for this dataset because it follows the curved dense regions and can leave isolated samples as noise.

K-Means is used as the Day 1 baseline, while hierarchical clustering is used to inspect the merge structure through a dendrogram.

## Tools

- Python
- NumPy
- pandas
- Matplotlib
- Scikit-learn
- SciPy
- Jupyter Notebook

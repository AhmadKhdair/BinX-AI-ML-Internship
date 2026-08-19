# t-SNE Visualization and Anomaly Detection

Day 4 of Week 5 focuses on visualizing high-dimensional data with t-SNE and detecting unusual observations with Isolation Forest.

The notebook continues with the same Breast Cancer Wisconsin dataset used in Day 3, which makes it possible to compare PCA and t-SNE on exactly the same samples and feature space.

## Notebook

`04_tsne_anomaly_detection.ipynb`

## What I Covered

The notebook follows this workflow:

1. load the Breast Cancer Wisconsin dataset with 30 numeric features;
2. keep the diagnosis target separate from the unsupervised workflow;
3. standardize the feature matrix with `StandardScaler`;
4. recreate the two-component PCA representation from Day 3;
5. apply t-SNE with different perplexity values;
6. select a final t-SNE configuration and visualize the data in 2D;
7. compare the PCA and t-SNE representations;
8. fit Isolation Forest on the full feature representation;
9. report the number of observations flagged as anomalies;
10. inspect the two most unusual samples using their standardized feature values;
11. visualize the anomaly flags on the t-SNE representation;
12. distinguish Isolation Forest anomalies from DBSCAN noise.

## Dataset

The experiment uses Scikit-learn's Breast Cancer Wisconsin dataset.

| Property | Value |
| --- | --- |
| Samples | 569 |
| Numeric features | 30 |
| Missing values | 0 |
| Target classes | malignant, benign |

The target is not passed to PCA, t-SNE, or Isolation Forest. It is used only as a visual reference when comparing how the known groups appear in the low-dimensional plots.

The clustering notebooks from Days 1 and 2 used different datasets, so their cluster labels cannot be transferred to these observations.

## Scaling

The 30 input features have different numeric ranges, so the feature matrix is standardized before dimensionality reduction:

```text
30 original features
        ↓
StandardScaler
        ↓
standardized feature space
        ↓
PCA / t-SNE / Isolation Forest
```

Scaling is particularly important for PCA and t-SNE because their results can otherwise be influenced by the original feature scales.

## PCA Reference

A two-component PCA transformation is recreated as a reference for the t-SNE comparison.

The first two components explain:

| Component | Explained Variance |
| --- | --- |
| PC1 | 44.27% |
| PC2 | 18.97% |
| Total | 63.24% |

This gives a useful linear view of the dataset, but it does not preserve all of the information contained in the original 30-dimensional representation.

## t-SNE

t-SNE is used to create a two-dimensional embedding while focusing on local neighborhood relationships.

Three perplexity values are compared:

```text
5
30
50
```

`perplexity` represents the effective neighborhood scale considered by t-SNE. It is related to the number of nearby observations, but it is not a fixed rule that selects exactly that many nearest neighbors.

The final visualization uses:

```python
n_components=2
perplexity=30
init="pca"
learning_rate="auto"
random_state=42
```

`random_state` keeps the stochastic transformation reproducible.

The resulting transformation is:

```text
569 samples × 30 features
            ↓
          t-SNE
            ↓
569 samples × 2 coordinates
```

The two t-SNE axes are embedding coordinates rather than interpretable features. Visible groups can provide evidence of local structure, but t-SNE itself does not create cluster labels.

## PCA vs. t-SNE

PCA and t-SNE reduce the same dataset to two dimensions, but they preserve different properties.

**PCA**
- is a linear transformation;
- preserves directions with high variance;
- can be used for compression as well as visualization;
- produces components that can be inspected through their feature loadings.

**t-SNE**
- is nonlinear;
- focuses on local neighborhood structure;
- is mainly used here for visualization;
- produces axes without direct feature meaning.

The t-SNE plot makes local structure easier to inspect, while the PCA plot provides a more direct view of the major linear variance directions.

## Isolation Forest

Isolation Forest is used to detect observations with unusual feature profiles.

The method builds multiple random isolation trees. Observations that are repeatedly isolated after relatively few splits are treated as more unusual.

The model uses:

```python
n_estimators=200
contamination=0.05
random_state=42
```

The `contamination` value is an assumption used to determine the final decision threshold. It does not mean the model independently discovered that exactly 5% of the dataset contains true anomalies.

### Result

| Status | Samples |
| --- | --- |
| Normal | 540 |
| Flagged anomalies | 29 |
| Flagged fraction | 5.10% |

Scikit-learn represents inliers with `1` and flagged anomalies with `-1`.

## Inspecting Flagged Samples

The two observations with the lowest Isolation Forest decision scores are inspected in more detail.

Their standardized feature values show several large deviations from the dataset mean, particularly in measurements related to area, perimeter, radius, and their error values.

For example, the most unusual sample has an `area error` more than 10 standard deviations above the dataset mean.

This helps explain why these observations can be isolated quickly by the random trees.

An anomaly flag does not mean that the observation is automatically incorrect or that it belongs to a particular diagnosis class. It only indicates that its feature pattern is unusual relative to the rest of the dataset.

## Isolation Forest vs. DBSCAN Noise

Both Isolation Forest and DBSCAN can identify unusual observations, but they do so for different reasons.

**DBSCAN** is primarily a clustering algorithm. A point receives the noise label `-1` when it is not connected to a sufficiently dense region according to `eps` and `min_samples`.

**Isolation Forest** is specifically an anomaly detection algorithm. A point is considered unusual when it can be isolated unusually quickly across the random trees.

The same observation may be identified by both methods, but DBSCAN noise and Isolation Forest anomalies are not equivalent definitions.

## Main Takeaway

The same high-dimensional dataset can be examined from different unsupervised-learning perspectives:

```text
30-dimensional data
        ↓
        ├── PCA
        │     → preserve major variance directions
        │
        ├── t-SNE
        │     → visualize local neighborhood structure
        │
        └── Isolation Forest
              → detect unusual observations
```

PCA and t-SNE both reduce dimensionality, but they should not be interpreted in the same way. t-SNE is especially useful for visual exploration, while Isolation Forest addresses a separate problem by identifying samples with unusual feature patterns.

For this experiment, Isolation Forest flagged 29 of the 569 observations for further inspection.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook or JupyterLab

Install the required packages with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Running the Notebook

From the repository environment:

```bash
jupyter notebook
```

Then open:

```text
week-5/d4/04_tsne_anomaly_detection.ipynb
```

Run the notebook from top to bottom so the preprocessing, PCA reference, t-SNE comparison, anomaly detection, and sample inspection are produced in order.
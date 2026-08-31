# Week 5 - Day 5: Heart Disease Unsupervised Learning Extension

This folder contains the Day 5 continuation of the Heart Disease Classification project.

The original project was built as a supervised classification workflow using `HeartDisease` as the target. This version extends the same dataset with Week 5 unsupervised learning methods to explore hidden structure, patient subgroups, density behavior, low-dimensional visualizations, and unusual records.

The unsupervised section does not use `HeartDisease` during fitting. The target is added back only after clustering or anomaly detection for interpretation.

## Files

| File | Description |
|---|---|
| `Heart_Disease_Classification_Final.ipynb` | Main notebook with supervised workflow and Week 5 unsupervised extension |
| `data/heart.csv` | Heart disease dataset |
| `data/data_dictionary.csv` | Column descriptions |
| `outputs/figures/` | Generated plots |
| `outputs/metrics/` | Generated metrics and profiling tables |

## Dataset

The dataset is a tabular clinical dataset with 918 patient records.

| Item | Value |
|---|---:|
| Rows | 918 |
| Original features | 11 |
| Target | `HeartDisease` |
| Target type | Binary classification |

The target values are:

| Value | Meaning |
|---:|---|
| 0 | No heart disease |
| 1 | Heart disease |

Main feature groups:

- Numeric: `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`
- Categorical: `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`

This project is an ML classification and exploratory analysis project. It is not a medical diagnosis system.

## Preprocessing

The same preprocessing logic is reused from the supervised workflow:

- `RestingBP = 0` is treated as invalid and converted to missing.
- `Cholesterol = 0` is treated as missing/unknown.
- Numeric features use median imputation and standard scaling.
- Categorical features use most-frequent imputation and one-hot encoding.
- Feature engineering adds `MaxHRRatio`, `OldpeakPositive`, `ExerciseOldpeak`, and `AgeGroup`.

The unsupervised feature matrix is built from the training split only:

| Split | Rows |
|---|---:|
| Training | 734 |
| Test | 184 |

After preprocessing and encoding, the unsupervised matrix has:

```text
734 rows x 27 features
```

The increase from 11 to 27 features comes from engineered features and one-hot encoding of categorical columns.

## Supervised Baseline Context

The supervised part compares Logistic Regression, Decision Tree, and Random Forest using 5-fold stratified cross-validation. The final selected supervised model is a tuned Random Forest.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8696 |
| Precision | 0.8679 |
| Recall | 0.9020 |
| F1-score | 0.8846 |
| ROC-AUC | 0.9173 |

This baseline is kept as context. The Week 5 extension is exploratory and unsupervised.

## Week 5 Unsupervised Methods

### PCA

PCA is used for 2D visualization of the preprocessed feature space.

| Component | Explained Variance |
|---|---:|
| PC1 | 27.43% |
| PC2 | 14.55% |

Together, the first two principal components explain about 42% of the variance. The PCA plot is useful for visual inspection, but it does not represent the full feature space.

### KMeans Clustering

KMeans is tested with `k` values from 2 to 8 using inertia and silhouette score.

The best selected value is:

```text
k = 2
silhouette_score = 0.1902
```

Cluster profile:

| Cluster | Size | Heart Disease % | Avg Age | Avg MaxHR | Avg Oldpeak | Exercise Angina % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 325 | 86.46 | 57.84 | 121.50 | 1.63 | 78.15 |
| 1 | 409 | 30.56 | 50.68 | 148.20 | 0.26 | 11.25 |

KMeans gives the clearest patient segmentation in this notebook. Cluster 0 looks like a higher-risk profile, while Cluster 1 looks like a lower-risk profile.

### Hierarchical Clustering

Agglomerative hierarchical clustering is applied with Ward linkage. Candidate cluster counts are compared using silhouette score.

The best selected value is:

```text
k = 2
silhouette_score = 0.1630
```

Cluster profile:

| Cluster | Size | Heart Disease % | Avg Age | Avg MaxHR | Avg Oldpeak | Exercise Angina % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 318 | 85.85 | 56.86 | 116.80 | 1.34 | 79.56 |
| 1 | 416 | 31.97 | 51.55 | 151.34 | 0.50 | 11.30 |

The hierarchical result supports the same general interpretation found by KMeans: one higher-risk group and one lower-risk group.

### DBSCAN

DBSCAN is used as a density-based clustering comparison. It does not require choosing `k`, but it is sensitive to `eps` and `min_samples`.

Selected setting:

```text
eps = 2.25
min_samples = 10
clusters = 2
noise points = 288
noise percentage = 39.24%
```

DBSCAN profile:

| Label | Size | Heart Disease % | Avg Age | Avg MaxHR | Avg Oldpeak | Exercise Angina % |
|---:|---:|---:|---:|---:|---:|---:|
| -1 Noise | 288 | 61.46 | 56.62 | 134.69 | 1.03 | 33.33 |
| 0 | 412 | 47.57 | 51.46 | 138.53 | 0.70 | 41.26 |
| 1 | 34 | 97.06 | 59.35 | 124.53 | 1.42 | 100.00 |

DBSCAN is useful as a density and noise check, but it is not the best segmentation method for this dataset because it labels a large portion of the records as noise.

### t-SNE

t-SNE is used only for visualization. Unlike PCA, which preserves global variance, t-SNE focuses more on local neighborhoods.

The notebook visualizes t-SNE in two ways:

- Colored by KMeans cluster labels
- Colored by the true `HeartDisease` label for interpretation only

The t-SNE output is not used for model training or final evaluation.

### Isolation Forest

Isolation Forest is used for unsupervised anomaly detection.

Configuration:

```text
contamination = 0.05
```

Results:

| Item | Value |
|---|---:|
| Anomaly count | 37 |
| Anomaly percentage | 5.04% |

Anomaly profile:

| Group | Size | Heart Disease % | Avg Age | Avg RestingBP | Avg Cholesterol | Avg MaxHR | Avg Oldpeak | Exercise Angina % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 697 | 55.67 | 53.60 | 132.42 | 245.70 | 136.47 | 0.83 | 40.60 |
| Anomaly | 37 | 48.65 | 58.51 | 141.51 | 262.55 | 134.57 | 1.50 | 45.95 |

Anomalies are not automatically heart-disease cases. They are records with unusual combinations of features compared with the rest of the training data.

## Main Findings

- KMeans is the strongest clustering method for this project because it gives the clearest and most interpretable segmentation.
- Hierarchical clustering supports the KMeans interpretation with a similar two-group structure.
- DBSCAN shows that the dataset does not have very clean density-based clusters in the encoded feature space.
- PCA and t-SNE are useful for visualization, but not for final prediction.
- Isolation Forest identifies unusual patient records, but anomalies should be reviewed carefully and not treated as automatic disease cases.
- `HeartDisease` is never used during unsupervised fitting, only after fitting for interpretation.

## How to Run

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn scipy notebook
```

Open the notebook:

```bash
jupyter notebook Heart_Disease_Classification_Final.ipynb
```

Run all cells from top to bottom. The notebook will create output folders for metrics and figures when needed.

## Conclusion

This notebook extends the supervised heart disease classification project with a complete Week 5 unsupervised learning analysis. The best practical clustering result comes from KMeans, supported by Hierarchical Clustering. DBSCAN, PCA, t-SNE, and Isolation Forest add useful exploratory views, but the final interpretation remains exploratory and should not be treated as medical diagnosis.
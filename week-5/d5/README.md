# Heart Disease Classification

This project builds a complete machine learning workflow for classifying heart disease presence from clinical and cardiac examination features.

The main task is supervised binary classification:

- `0`: no heart disease
- `1`: heart disease

The project also includes a Phase 2 unsupervised learning extension using `PCA` and `KMeans` to explore patient groups without using the target during clustering.

This project is educational. It is not a medical diagnosis system and it does not predict future heart disease risk.

## Project Structure

```text
heart-disease-classification/
├── README.md
├── requirements.txt
├── data/
│   ├── heart.csv
│   └── data_dictionary.csv
├── notebooks/
│   └── Heart_Disease_Classification_Final.ipynb
└── outputs/
    ├── figures/
    └── metrics/
```

## Dataset

The dataset contains `918` records and `12` columns:

- `11` input features
- `1` target column: `HeartDisease`

The features describe patient age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG result, maximum heart rate, exercise-induced angina, oldpeak, and ST slope.

## Main Workflow

The notebook follows a top-to-bottom machine learning workflow:

1. Load the dataset.
2. Validate the expected columns.
3. Audit data types, missing values, duplicates, and target balance.
4. Clean medically invalid zero values:
   - `RestingBP = 0` is treated as invalid.
   - `Cholesterol = 0` is treated as missing/unknown.
5. Split the data into training and held-out test sets.
6. Run EDA and descriptive statistics on the training set only.
7. Build preprocessing and feature engineering inside scikit-learn pipelines.
8. Compare `Logistic Regression`, `Decision Tree`, and `Random Forest` using 5-fold stratified cross-validation.
9. Tune models using `GridSearchCV` on the training set only.
10. Evaluate the selected final model once on the held-out test set.
11. Add Phase 2 unsupervised analysis with `PCA` and `KMeans`.

## Cleaning Strategy

The original dataset does not contain visible missing values or duplicated rows.

The main cleaning issue is invalid zero values:

- `RestingBP = 0`: not valid for resting blood pressure.
- `Cholesterol = 0`: not realistic as a serum cholesterol value.

These values are converted to `NaN`, then handled inside the preprocessing pipeline using median imputation. The cholesterol-zero rows are not dropped because they represent a large part of the dataset, and dropping them could bias the training sample.

## Feature Engineering

The notebook adds simple and explainable engineered features:

- `MaxHRRatio`: maximum heart rate divided by age-based expected maximum heart rate.
- `OldpeakPositive`: whether `Oldpeak` is greater than zero.
- `ExerciseOldpeak`: interaction between exercise-induced angina and oldpeak.
- `AgeGroup`: broad age group.

These features are created inside the pipeline to avoid leakage during cross-validation and tuning.

## Models

The supervised section uses classical machine learning models:

- `Logistic Regression`
- `Decision Tree`
- `Random Forest`

The selected final model is `Random Forest`.

## Supervised Results

Final held-out test results:

| Metric | Value |
|---|---:|
| Accuracy | 0.870 |
| Precision | 0.868 |
| Recall | 0.902 |
| F1-score | 0.885 |
| ROC-AUC | 0.917 |

Confusion matrix on the held-out test set:

| | Predicted No Heart Disease | Predicted Heart Disease |
|---|---:|---:|
| Actual No Heart Disease | 68 | 14 |
| Actual Heart Disease | 10 | 92 |

The model achieved strong recall for the positive class, which is important in medical-style classification problems because false negatives are usually more costly than false positives.

## Phase 2: Unsupervised Learning Extension

The unsupervised section is added as a separate extension after the supervised model is complete.

The goal is not to improve the final classifier. The goal is to explore whether the feature space contains patient groups with different clinical patterns.

Important rule:

`HeartDisease` is not used to fit `PCA` or `KMeans`. It is used only after clustering to interpret the discovered groups.

The extension includes:

- Preparing the training features without the target.
- Applying the same preprocessing logic used in the supervised pipeline.
- Using `PCA` to reduce the preprocessed feature matrix to two components for visualization.
- Using `KMeans` to cluster the training records.
- Choosing `k` using `Elbow Method` and `Silhouette Score`.
- Profiling clusters using clinical features and heart-disease rate.

## Unsupervised Results

The preprocessed unsupervised matrix has shape:

```text
(734, 27)
```

This means:

- `734` training records
- `27` features after preprocessing, one-hot encoding, and feature engineering

The best cluster count from the tested range was:

```text
k = 2
```

Cluster profile:

| Cluster | Size | Heart Disease % | Avg Age | Avg MaxHR | Avg Oldpeak | Exercise Angina % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 325 | 86.46 | 57.84 | 121.50 | 1.63 | 78.15 |
| 1 | 409 | 30.56 | 50.68 | 148.20 | 0.26 | 11.25 |

Interpretation:

- `Cluster 0` looks like the higher-risk group in this dataset. It has higher heart-disease rate, higher oldpeak, lower maximum heart rate, and much higher exercise-induced angina rate.
- `Cluster 1` looks like the lower-risk group. It has lower heart-disease rate, higher maximum heart rate, lower oldpeak, and much lower exercise-induced angina rate.

The silhouette score for `k = 2` is about `0.19`, so the clusters should be treated as exploratory groups, not clinically validated patient segments.

## Output Files

Generated figures include:

- `target_distribution_train.png`
- `numeric_distributions_by_target.png`
- `categorical_positive_rates_train.png`
- `correlation_matrix_train.png`
- `final_confusion_matrix.png`
- `final_roc_curve.png`
- `phase2_pca_training_view.png`
- `phase2_kmeans_k_selection.png`
- `phase2_kmeans_clusters_pca.png`

Generated metric files include:

- `cross_validation_model_comparison.csv`
- `tuned_model_results.csv`
- `final_test_metrics.json`
- `final_test_predictions.csv`
- `project_summary.csv`
- `phase2_kmeans_k_selection.csv`
- `phase2_cluster_profile.csv`

## How To Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/Heart_Disease_Classification_Final.ipynb
```

The notebook should be run from the project folder so it can find:

```text
data/heart.csv
```

## Limitations

- The dataset is a merged heart disease dataset, and the original source of each row is not available as a feature.
- The target represents heart disease presence in the available record, not future disease risk.
- The project is not a medical diagnosis system.
- The final supervised result is based on one held-out test split, not external clinical validation.
- The unsupervised clusters are exploratory and should not be interpreted as medical categories.

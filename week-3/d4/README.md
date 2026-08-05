# Classification Model Comparison

Week 3, Day 4 of the BinX AI/ML Internship focuses on comparing supervised classification algorithms on an imbalanced real-world dataset.

This project predicts whether a COVID-19 test result is **positive** or **negative** using recorded symptoms, age group, gender, and the reason for testing.

## Project Objective

The main goal is to train several classifiers under the same conditions and compare how they handle the positive class.

The comparison includes:

- Decision Tree
- Random Forest
- k-Nearest Neighbors
- Linear Support Vector Machine

Because positive results form a small part of the dataset, the models are evaluated with more than accuracy alone.

## Dataset

The notebook uses:

```text
corona_tested_individuals_ver_006.english.csv
```

The original dataset contains 278,848 rows and 10 columns.

### Available columns

| Column | Description |
|---|---|
| `test_date` | Date of the recorded test |
| `cough` | Indicates whether cough was recorded |
| `fever` | Indicates whether fever was recorded |
| `sore_throat` | Indicates whether sore throat was recorded |
| `shortness_of_breath` | Indicates whether shortness of breath was recorded |
| `head_ache` | Indicates whether headache was recorded |
| `corona_result` | Recorded test result |
| `age_60_and_above` | Whether the tested individual is aged 60 or above |
| `gender` | Recorded gender |
| `test_indication` | Reason for testing |

The target is encoded as:

- `0` — negative
- `1` — positive

Records labeled `other` are excluded because the notebook treats the task as binary classification.

## Workflow

The notebook follows this sequence:

1. Load and inspect the dataset.
2. Review data types, missing values, and repeated rows.
3. Standardize text values and column names.
4. Convert symptom columns to binary integers.
5. Replace missing demographic values with `unknown`.
6. Keep only positive and negative test results.
7. Compare symptom rates across both target classes.
8. Select the modeling features.
9. Create a reproducible stratified sample.
10. Split the data into training and test sets.
11. Build preprocessing and model pipelines.
12. Train four classification algorithms.
13. Compare the models with a majority-class baseline.
14. Evaluate performance with multiple metrics.
15. Inspect confusion matrices, ROC curves, and precision-recall curves.
16. Review the Decision Tree structure and Random Forest feature importance.
17. Select the final model using F1-score as the main criterion.

## Data Preparation

The five symptom indicators are treated as numeric features.

The following columns are treated as categorical features and transformed with one-hot encoding:

- `age_60_and_above`
- `gender`
- `test_indication`

Preprocessing is placed inside each Scikit-learn pipeline. This ensures that the transformations are fitted on the training data rather than the full dataset.

The notebook uses a reproducible stratified sample of 30,000 records so all four classifiers can be compared on the same data without making k-NN impractical to run on a standard laptop.

The sample contains:

| Class | Records | Percentage |
|---|---:|---:|
| Negative | 28,393 | 94.64% |
| Positive | 1,607 | 5.36% |

An 80/20 stratified split produces 24,000 training records and 6,000 test records.

## Evaluation Metrics

The models are compared using:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

A majority-class baseline is included to show why accuracy alone is not suitable for this dataset.

The baseline reaches 94.65% accuracy by predicting every record as negative, but its positive-class recall and F1-score are both 0.

## Results

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-NN | 0.9692 | 0.7927 | 0.7764 | 0.5950 | 0.9903 | **0.6737** | 0.9081 | 0.6758 |
| Linear SVM | 0.9255 | 0.8637 | 0.4009 | 0.7944 | 0.9329 | 0.5329 | 0.9239 | 0.6741 |
| Random Forest | 0.9192 | **0.8882** | 0.3848 | **0.8536** | 0.9229 | 0.5305 | **0.9249** | **0.7092** |
| Decision Tree | 0.8872 | 0.8699 | 0.3027 | 0.8505 | 0.8892 | 0.4464 | 0.9194 | 0.6733 |

## Main Findings

k-NN achieved the highest F1-score at 0.6737 and was selected as the final model under the main evaluation criterion. It also produced the highest precision and specificity, which means its positive predictions were more reliable and it generated fewer false positives.

Random Forest detected the largest proportion of positive cases, reaching a recall of 0.8536. It also achieved the highest balanced accuracy, ROC-AUC, and PR-AUC. This stronger positive-case detection came with lower precision and more false positive predictions.

The comparison therefore does not identify one classifier as the best in every respect:

- k-NN provides the strongest balance between precision and recall.
- Random Forest is more suitable when detecting as many positive cases as possible is the main priority.
- Accuracy alone is misleading because negative results represent 94.64% of the binary target.

## Feature Importance

The Random Forest relied most heavily on:

1. `test_indication_other`
2. `test_indication_contact with confirmed`
3. `fever`
4. `cough`

The feature-importance values describe how the fitted model used the predictors. They do not establish a causal relationship with a positive test result.

## Repository Contents

```text
d4/
├── 04_classifier_comparison.ipynb
├── corona_tested_individuals_ver_006.english.csv
└── README.md
```

## Running the Notebook

The notebook expects the CSV file to be stored in the same folder.

### Requirements

- Python 3.10 or later
- Jupyter Notebook or JupyterLab
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install the required packages with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Start Jupyter from the repository directory:

```bash
jupyter notebook
```

Open `week-3/d4/04_classifier_comparison.ipynb`, then run all cells from top to bottom.

## Reproducibility

The notebook uses:

```python
RANDOM_STATE = 42
```

The same random state is applied to the modeling sample, train-test split, and supported classifiers. The preprocessing steps and models are kept together in Scikit-learn pipelines.

## Limitations

- The model comparison uses a stratified sample of 30,000 records rather than the full cleaned dataset.
- The target is strongly imbalanced.
- The dataset does not provide a unique individual or test identifier, so identical rows cannot be confirmed as accidental duplicates.
- `test_date` is excluded from the predictors.
- The reported results apply to the selected sample, features, split, and model settings.
- This work is a machine-learning exercise and is not a clinical diagnosis system.

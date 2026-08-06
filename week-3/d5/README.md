# COVID-19 Test Result Classification

A supervised machine learning mini-project that predicts whether an individual received a positive or negative COVID-19 test result using reported symptoms and basic patient information.

This project was completed for **Week 3, Day 5** of the BinX Tech AI/ML Internship and demonstrates a complete binary classification workflow using Scikit-learn.

## Project Overview

The dataset contains individual COVID-19 testing records. Each row represents one tested person and includes:

- Five symptom indicators
- Age group
- Gender
- Test indication
- Recorded COVID-19 result

The original dataset contains **278,848 records**. After excluding the `other` target category, **274,956 records** remain for binary classification:

| Class | Records | Percentage |
|---|---:|---:|
| Negative | 260,227 | 94.64% |
| Positive | 14,729 | 5.36% |

The strong class imbalance makes accuracy insufficient on its own, so the models are also evaluated with precision, recall, F1-score, and ROC-AUC.

## Objective

Build and compare supervised classification models that predict:

- `0` — Negative COVID-19 test result
- `1` — Positive COVID-19 test result

The final model is selected using the highest test-set F1-score, which balances precision and recall for the minority positive class.

## Dataset Features

| Feature | Description |
|---|---|
| `cough` | Cough symptom indicator |
| `fever` | Fever symptom indicator |
| `sore_throat` | Sore throat symptom indicator |
| `shortness_of_breath` | Shortness of breath symptom indicator |
| `head_ache` | Headache symptom indicator |
| `age_60_and_above` | Whether the individual is aged 60 or above |
| `gender` | Recorded gender |
| `test_indication` | Reason for taking the test |
| `corona_result` | Target variable: positive or negative test result |

The `test_date` column is excluded from modeling because the prediction is based on symptoms and individual information rather than the calendar date.

## Project Workflow

1. Load and inspect the dataset.
2. Review data types, missing values, and recorded categories.
3. Convert the target into a binary classification variable.
4. Explore class balance and symptom rates.
5. Create a reproducible stratified modeling sample.
6. Split the data into training and testing sets.
7. Build preprocessing pipelines for numeric and categorical features.
8. Train a dummy baseline and five classification models.
9. Compare models using multiple evaluation metrics.
10. Examine the selected model with a classification report, confusion matrix, and ROC curve.

## Preprocessing

Preprocessing is included inside Scikit-learn pipelines and is fitted using the training data only.

### Numeric features

- Missing values are filled using the most frequent training value.
- Symptom columns are standardized with `StandardScaler`.

### Categorical features

- Missing values are replaced with `Unknown`.
- Categories are transformed using one-hot encoding.
- Previously unseen categories are ignored during prediction.

A stratified sample of **40,000 records** is used to make the comparison with k-nearest neighbors practical while preserving the original class distribution.

The modeling sample is divided into:

- **32,000 training records**
- **8,000 testing records**

## Models

The following classifiers are compared:

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Linear Support Vector Machine
- k-Nearest Neighbors

The dummy classifier provides a minimum baseline by always predicting the majority class.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted positive cases that are truly positive |
| Recall | Proportion of actual positive cases detected by the model |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to rank positive cases above negative cases across thresholds |

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| k-Nearest Neighbors | **0.9694** | **0.7788** | 0.5991 | **0.6772** | 0.8654 |
| Decision Tree | 0.9299 | 0.4165 | 0.7669 | 0.5398 | 0.9034 |
| Linear SVM | 0.9282 | 0.4081 | 0.7506 | 0.5287 | 0.9044 |
| Random Forest | 0.9165 | 0.3716 | **0.8065** | 0.5088 | **0.9099** |
| Logistic Regression | 0.9166 | 0.3672 | 0.7669 | 0.4966 | 0.9059 |
| Dummy Baseline | 0.9464 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

## Selected Model

**k-Nearest Neighbors** is selected because it achieved the highest test-set F1-score:

- Accuracy: **0.9694**
- Precision: **0.7788**
- Recall: **0.5991**
- F1-score: **0.6772**
- ROC-AUC: **0.8654**

### Confusion Matrix Summary

| Outcome | Count |
|---|---:|
| True Negatives | 7,498 |
| False Positives | 73 |
| False Negatives | 172 |
| True Positives | 257 |

The model correctly identified **257 of 429 positive cases**. Its precision shows that most positive predictions were correct, but its recall indicates that it missed 172 positive cases.

Random Forest produced the highest recall and ROC-AUC. It may be preferred when detecting the largest possible share of positive cases is more important than limiting false-positive predictions.

## Key Findings

- The dummy baseline achieved 94.64% accuracy while detecting no positive cases, demonstrating why accuracy alone is misleading for this dataset.
- Every recorded symptom appeared more frequently among positive cases.
- Cough appeared in 44.80% of positive cases and 13.46% of negative cases.
- Fever appeared in 37.83% of positive cases and 6.08% of negative cases.
- k-Nearest Neighbors provided the strongest precision–recall balance.
- Random Forest detected the largest proportion of positive cases.

## Project Structure

```text
week-3/
├── corona dataset/
│   └── corona_tested_individuals_ver_006.english.csv
└── d5/
    ├── 05_supervised_learning_classification_mini_project_final.ipynb
    └── README.md
```

The notebook uses the following fixed relative path:

```python
DATA_PATH = "../corona dataset/corona_tested_individuals_ver_006.english.csv"
```

Keep the dataset and notebook in the structure shown above.

## Requirements

- Python 3.10 or later
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

Install the required packages with:

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Running the Notebook

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

Open:

```text
week-3/d5/05_supervised_learning_classification_mini_project_final.ipynb
```

Run all cells from top to bottom.

## Reproducibility

A fixed random seed is used:

```python
RANDOM_STATE = 42
```

The modeling sample and train/test split are stratified so that the positive and negative class proportions remain consistent.

## Limitations

- The predictors are limited to five symptoms and a small number of demographic fields.
- Age and gender contain missing values.
- A 40,000-record stratified sample is used for model comparison.
- Results are based on one train/test split and have not yet been confirmed with cross-validation.
- The dataset reflects a specific testing period and process.
- The project is an educational machine learning exercise and is not a medical diagnostic system.

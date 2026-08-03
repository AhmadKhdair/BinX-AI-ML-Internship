# Week 3 — Day 2: Logistic Regression and Binary Classification

## Overview

This notebook develops a complete binary classification workflow using **Logistic Regression** and **Scikit-learn**.

The task is to predict whether a historical COVID-19 test result is:

* `0` — Negative
* `1` — Positive

The work covers the full supervised machine learning process, including dataset inspection, target cleaning, feature selection, preprocessing, model training, evaluation, class-imbalance handling, and coefficient interpretation.

The notebook compares a standard Logistic Regression model with a class-balanced version to study the trade-off between precision and recall.

---

## Learning Objectives

By completing this notebook, the following concepts and skills were practised:

* Distinguishing classification from regression
* Understanding the feature matrix `X` and target vector `y`
* Cleaning and encoding a binary target variable
* Detecting class imbalance
* Selecting appropriate model features
* Preventing target leakage
* Creating a stratified train-test split
* Handling missing numerical and categorical values
* Applying one-hot encoding
* Building a `ColumnTransformer`
* Combining preprocessing and modelling in a Scikit-learn `Pipeline`
* Training Logistic Regression classifiers
* Generating class predictions and probabilities
* Evaluating classification performance
* Interpreting a confusion matrix
* Comparing precision and recall
* Applying `class_weight="balanced"`
* Interpreting model coefficients and odds ratios

---

## Problem Definition

The project is a supervised binary classification problem.

The model receives recorded symptoms and patient-related information and predicts the corresponding COVID-19 test class.

### Features

The model uses eight original input features.

#### Numerical symptom features

* `cough`
* `fever`
* `sore_throat`
* `shortness_of_breath`
* `head_ache`

#### Categorical features

* `age_60_and_above`
* `gender`
* `test_indication`

The raw `test_date` column was excluded from the initial model.

### Target

The original target column is:

```text
corona_result
```

After filtering and encoding:

```text
negative → 0
positive → 1
```

Observations labelled as `other` were excluded because they do not belong clearly to either class in the binary classification task.

---

## Dataset

The notebook uses a historical anonymized COVID-19 testing dataset.

### Original dataset

| Property                |                           Value |
| ----------------------- | ------------------------------: |
| Observations            |                         278,848 |
| Columns                 |                              10 |
| Original target classes | `negative`, `positive`, `other` |

### Binary classification dataset

After removing observations labelled as `other`:

| Property              |   Value |
| --------------------- | ------: |
| Observations          | 274,956 |
| Negative observations | 260,227 |
| Positive observations |  14,729 |
| Negative percentage   |  94.64% |
| Positive percentage   |   5.36% |

The target distribution is strongly imbalanced, so accuracy alone is not sufficient for evaluating model quality.

### Dataset file

```text
../corona dataset/corona_tested_individuals_ver_006.english.csv
```

[View the dataset file](../corona%20dataset/corona_tested_individuals_ver_006.english.csv)

---

## Workflow

The notebook follows this end-to-end workflow:

1. Load the historical dataset
2. Inspect its shape, columns, data types, and missing values
3. Examine the target distribution
4. Remove the `other` target category
5. Encode `negative` as `0` and `positive` as `1`
6. Select the model features
7. Inspect feature values and missing-data percentages
8. Create the feature matrix `X` and target vector `y`
9. Perform an 80/20 stratified train-test split
10. Build numerical and categorical preprocessing pipelines
11. Train a baseline Logistic Regression model
12. Evaluate the baseline model
13. Train a class-balanced Logistic Regression model
14. Evaluate the balanced model
15. Compare both models
16. Interpret coefficients, odds ratios, and reference categories

---

## Train-Test Split

The filtered data was divided into:

| Dataset      | Observations | Percentage |
| ------------ | -----------: | ---------: |
| Training set |      219,964 |        80% |
| Testing set  |       54,992 |        20% |

The split used:

```python
random_state=42
stratify=y
```

Stratification preserved approximately the same positive and negative class distribution in both sets.

The testing set remained separate from model training and was used only for final evaluation.

---

## Preprocessing

Preprocessing was implemented inside a Scikit-learn pipeline to ensure that all transformations were learned from the training data only.

### Numerical symptom features

Missing symptom values were replaced using the most frequently recorded value in each training column:

```python
SimpleImputer(strategy="most_frequent")
```

### Categorical features

Missing categorical values were represented explicitly as:

```text
Missing
```

using:

```python
SimpleImputer(
    strategy="constant",
    fill_value="Missing"
)
```

The text categories were converted into numerical indicator columns using:

```python
OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)
```

After preprocessing, the eight original features were transformed into eleven numerical model features.

---

## Models

Two Logistic Regression models were trained.

### Baseline Logistic Regression

The baseline model used the default class weighting:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### Class-Balanced Logistic Regression

The second model gave greater training importance to the minority positive class:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
```

Both models used the same training data, testing data, feature set, and preprocessing strategy.

---

## Evaluation Metrics

Because the target is imbalanced, the models were evaluated using multiple complementary metrics:

* **Accuracy** — overall proportion of correct predictions
* **Balanced Accuracy** — average recall across both classes
* **Precision** — reliability of positive predictions
* **Recall** — percentage of actual positive observations identified
* **F1-score** — balance between precision and recall
* **ROC-AUC** — ability to rank positive observations above negative observations
* **Confusion Matrix** — detailed counts of correct and incorrect predictions

---

## Results

### Model comparison

| Metric            | Baseline Model | Balanced Model | Difference |
| ----------------- | -------------: | -------------: | ---------: |
| Accuracy          |         0.9638 |         0.9276 |    -0.0362 |
| Balanced Accuracy |         0.7468 |         0.8388 |    +0.0920 |
| Precision         |         0.7379 |         0.4039 |    -0.3340 |
| Recall            |         0.5037 |         0.7393 |    +0.2356 |
| F1-score          |         0.5987 |         0.5224 |    -0.0763 |
| ROC-AUC           |         0.8903 |         0.8935 |    +0.0032 |

### Baseline confusion matrix

|                 | Predicted Negative | Predicted Positive |
| --------------- | -----------------: | -----------------: |
| Actual Negative |             51,519 |                527 |
| Actual Positive |              1,462 |              1,484 |

The baseline model produced higher precision and fewer false positive predictions, but it identified only 50.37% of the actual positive observations.

### Balanced-model confusion matrix

|                 | Predicted Negative | Predicted Positive |
| --------------- | -----------------: | -----------------: |
| Actual Negative |             48,831 |              3,215 |
| Actual Positive |                768 |              2,178 |

The balanced model:

* Increased positive-class recall from 50.37% to 73.93%
* Reduced false negatives from 1,462 to 768
* Correctly identified 694 additional positive observations
* Increased false positives from 527 to 3,215
* Reduced positive-class precision from 73.79% to 40.39%

---

## Model Selection

The appropriate model depends on the practical cost of each type of classification error.

The baseline model is more suitable when false positive predictions are especially costly and higher precision is required.

The class-balanced model is more suitable for a screening-oriented objective where failing to identify a positive observation is considered more costly than producing additional false alerts.

For this educational screening task, the class-balanced model was selected because it substantially improved positive-class recall and reduced the number of missed positive observations.

---

## Model Interpretation

The baseline Logistic Regression coefficients were inspected to understand how the transformed features were associated with the positive class.

* Positive coefficients are associated with higher estimated log-odds of class `1`.
* Negative coefficients are associated with lower estimated log-odds of class `1`.
* Exponentiating a coefficient produces its odds ratio.
* One-hot encoded categorical coefficients are interpreted relative to their removed reference categories.

The reference categories created by `drop="first"` were:

| Feature            | Reference category |
| ------------------ | ------------------ |
| `age_60_and_above` | `Missing`          |
| `gender`           | `Missing`          |
| `test_indication`  | `Abroad`           |

The coefficients represent statistical associations learned by the model rather than causal relationships.

---

## Repository Contents

```text
week-3/
├── corona dataset/
│   └── corona_tested_individuals_ver_006.english.csv
└── d2/
    ├── README.md
    └── 02_logistic_regression_and_binary_classification.ipynb
```

| File                                                                                                                 | Description                             |
| -------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| [`02_logistic_regression_and_binary_classification.ipynb`](./02_logistic_regression_and_binary_classification.ipynb) | Complete Day 2 notebook                 |
| [`README.md`](./README.md)                                                                                           | Documentation for the Day 2 work        |
| [`corona_tested_individuals_ver_006.english.csv`](../corona%20dataset/corona_tested_individuals_ver_006.english.csv) | Historical dataset used by the notebook |

---

## Technologies Used

* Python 3.10+
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Git and GitHub

---

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd BinX-AI-ML-Internship
```

### 2. Activate the project environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the required libraries

```powershell
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```powershell
jupyter notebook
```

### 5. Open the notebook

Navigate to:

```text
week-3/d2/02_logistic_regression_and_binary_classification.ipynb
```

Run the notebook cells from top to bottom.

The dataset must remain at:

```text
week-3/corona dataset/corona_tested_individuals_ver_006.english.csv
```

so that the relative dataset path used by the notebook resolves correctly.

---

## Key Takeaways

* Binary classification predicts one of two target classes.
* Target labels must be cleaned and numerically encoded.
* Preprocessing must be learned from training data only.
* Scikit-learn pipelines help prevent data leakage.
* Accuracy can be misleading when one class is much larger than the other.
* Precision and recall describe different types of classification performance.
* Class weighting can improve minority-class recall while increasing false positives.
* The best model depends on the relative cost of false negatives and false positives.
* Logistic Regression coefficients can be interpreted through log-odds and odds ratios.

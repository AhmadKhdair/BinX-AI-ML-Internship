# Train, Validation, and Test Splits

A focused machine-learning evaluation notebook demonstrating how to separate **model fitting**, **development decisions**, and **final evaluation** using train, validation, and test sets.

The notebook uses an imbalanced COVID-19 classification dataset from the previous internship week and shows why the test set must remain outside the development feedback loop until the model configuration is finalized.

## Notebook

[`01_train_validation_test_split.ipynb`](./01_train_validation_test_split.ipynb)

## Learning Objectives

This notebook is designed to demonstrate how to:

- create a stratified **60% train / 20% validation / 20% test** split;
- assign a distinct role to each dataset split;
- avoid tuning model choices against the final test set;
- choose an evaluation metric before comparing models;
- recognize why accuracy can be misleading on imbalanced data;
- establish a simple majority-class baseline;
- tune a Decision Tree using validation data only;
- compare multiple supervised-learning model families fairly;
- tune a Logistic Regression classification threshold on validation data;
- freeze the selected configuration before final testing;
- interpret precision, recall, F1-score, and the confusion matrix;
- compare validation and test performance without reopening model selection.

## Problem Setup

The original target contains three values:

- `negative`
- `positive`
- `other`

For this experiment, the task is converted into binary classification:

- `negative` → `0`
- `positive` → `1`
- `other` → excluded from modeling

After filtering the target and removing rows with missing values in the selected symptom features, the modeling dataset contains **274,702 observations**.

The positive class represents approximately **5.35%** of the modeling data, making this an imbalanced classification problem.

## Features

The model uses seven binary input features:

- `cough`
- `fever`
- `sore_throat`
- `shortness_of_breath`
- `head_ache`
- `contact_with_confirmed`
- `abroad`

The last two features are derived from the categorical `test_indication` column. The remaining `Other` indication is represented when both derived indicators are zero.

## Evaluation Strategy

The data is split with a fixed `random_state=42` and stratification in both splitting stages:

| Split | Purpose | Share |
|---|---|---:|
| Training | Fit model parameters | 60% |
| Validation | Compare models, hyperparameters, and thresholds | 20% |
| Test | Final held-out evaluation | 20% |

The test set is held out before development decisions begin and is evaluated only after the model family and classification threshold are fixed.

### Primary Metric

Because the target is strongly imbalanced, **F1-score** is used as the primary model-selection metric.

Accuracy is retained as secondary context, while precision and recall are used to understand the final error profile.

A majority-class classifier illustrates the problem clearly:

| Model | Validation Accuracy | Validation F1 |
|---|---:|---:|
| Majority-class baseline | 0.9465 | 0.0000 |

Despite high accuracy, the baseline never identifies a positive case. This demonstrates why accuracy alone is not sufficient for this experiment.

## Development Workflow

The notebook follows an incremental development process rather than changing many variables at once.

### 1. Initial Decision Tree

A small Decision Tree with `max_depth=2` establishes the first real model baseline.

- Validation accuracy: **0.9653**
- Validation F1: **0.6363**

### 2. Decision Tree Depth Tuning

Several `max_depth` values are compared using validation F1-score only.

The best tested Decision Tree configuration is:

- `max_depth=3`
- Validation F1: **0.6558**

### 3. Model Family Comparison

The tuned Decision Tree is compared with Logistic Regression and Random Forest using the same training and validation data.

| Model | Validation Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9665 | 0.7270 | 0.5988 | **0.6567** |
| Tuned Decision Tree | 0.9663 | 0.7231 | 0.5999 | 0.6558 |
| Random Forest | **0.9678** | **0.7962** | 0.5356 | 0.6404 |

Random Forest achieves the highest validation accuracy and precision, but its lower recall reduces its F1-score. Logistic Regression is therefore selected as the leading model according to the metric defined before comparison.

### 4. Classification Threshold Tuning

The Logistic Regression probability threshold is then tuned using validation data only.

Among the tested thresholds (`0.10`, `0.20`, `0.30`, `0.40`, `0.50`), the best validation F1-score is obtained at:

- threshold: **0.20**
- validation precision: **0.7157**
- validation recall: **0.6237**
- validation F1: **0.6665**

The selected model family and threshold are frozen before the test set is evaluated.

## Final Held-Out Test Results

The finalized configuration is:

- **Model:** Logistic Regression
- **Classification threshold:** `0.20`

Final test metrics:

| Metric | Test Result |
|---|---:|
| Accuracy | **0.9664** |
| Precision | **0.7122** |
| Recall | **0.6247** |
| F1-score | **0.6656** |

Confusion matrix:

```text
[[51260   742]
 [ 1103  1836]]
```

The validation F1-score (`0.6665`) and test F1-score (`0.6656`) are very close for this particular split, suggesting that the validation result was reasonably representative of the held-out evaluation.

## Key Engineering Lesson

The main result of this notebook is not that Logistic Regression happened to achieve the highest score.

The important lesson is the **separation of responsibilities**:

- training data teaches the model parameters;
- validation data guides development decisions;
- test data estimates final performance after those decisions are complete.

Repeatedly changing the model after observing test performance would allow information from the test set to influence the final system indirectly. The test set would then behave like another validation set, and its score would no longer be an independent final estimate.

## Limitations

This notebook intentionally focuses on train/validation/test discipline rather than building a production-ready medical classifier.

Important limitations include:

- the positive class is strongly imbalanced;
- model comparison relies on a single validation split;
- only a small set of models and hyperparameters is explored;
- threshold tuning evaluates only a small predefined set of thresholds;
- false negatives remain substantial in the final model;
- the experiment does not establish clinical validity or deployment readiness.

A single validation split can still be unusually easy or difficult. The natural next step is **cross-validation**, which provides a more stable estimate by evaluating the model across multiple training/validation partitions.

## Dataset Location

The notebook expects the Week 3 dataset at the following relative path:

```text
../../week-3/corona dataset/corona_tested_individuals_ver_006.english.csv
```

A compatible repository structure is therefore:

```text
BinX-AI-ML-Internship/
├── week-3/
│   └── corona dataset/
│       └── corona_tested_individuals_ver_006.english.csv
└── week-4/
    └── d1/
        ├── README.md
        └── 01_train_validation_test_split.ipynb
```

## Requirements

The notebook uses:

- Python
- Jupyter Notebook or JupyterLab
- Pandas
- Scikit-learn

Install the required Python packages with:

```bash
pip install pandas scikit-learn jupyter
```

If the repository already contains a maintained `requirements.txt`, prefer installing from it to reproduce the project environment:

```bash
pip install -r requirements.txt
```

## Running the Notebook

From the repository environment:

```bash
jupyter notebook
```

Then open:

```text
week-4/d1/01_train_validation_test_split.ipynb
```

Run the notebook from top to bottom so that the split, model-selection decisions, and final evaluation occur in the intended order.

## Reproducibility Notes

The experiment uses `random_state=42` for dataset splitting and supported model initialization. This makes the demonstrated split and model comparison reproducible when the same data and compatible library versions are used.

The notebook also keeps preprocessing intentionally simple and performs model-selection decisions only on the training and validation sets.

## Next Step

The next evaluation improvement is **k-fold cross-validation**, which reduces dependence on one validation split and provides both an average performance estimate and a measure of variability across folds.

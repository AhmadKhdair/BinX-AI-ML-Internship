# Week 6 - Day 1: Sprint 1 Baseline

This folder contains the Day 1 baseline notebook for the Week 6 neural-network sprint.

The work continues from the existing Heart Disease Classification project. The goal here is not to train the neural network yet. The goal is to reproduce the classical ML baseline clearly, keep the test set fixed, and define the score that later neural-network experiments must be compared against.

## Files

| File | Description |
|---|---|
| `Week6_Day1_Sprint1_Baseline.ipynb` | Main Day 1 notebook |
| `outputs/day1_baseline_cv_results.csv` | Cross-validation results created when the notebook is run |
| `outputs/day1_strong_benchmark.csv` | Tuned Random Forest benchmark created when the notebook is run |

## Notebook Scope

The notebook covers:

- Sprint 1 goal and Day 1 backlog
- loading and verifying `heart.csv`
- handling invalid zero values in `RestingBP` and `Cholesterol`
- reproducing the same stratified 80/20 train/test split with `random_state=42`
- brief EDA on the training split only
- preprocessing with imputation, scaling, one-hot encoding, and engineered features
- baseline comparison between Logistic Regression, Decision Tree, and Random Forest
- tuned Random Forest reference score on the frozen test set
- Day 1 neural-network architecture notes

## Baseline Results

The baseline models were evaluated with 5-fold stratified cross-validation on the training set.

| Model | Mean CV F1 | Mean CV ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.8610 | 0.9255 |
| Decision Tree | 0.8320 | 0.8153 |
| Random Forest | 0.8663 | 0.9287 |

The tuned Random Forest remains the strongest classical reference from the previous project.

| Model | Test F1 | Test ROC-AUC | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Tuned Random Forest | 0.8846 | 0.9173 | 0.8696 | 0.8679 | 0.9020 |

For the later Week 6 comparison, the main reference score is:

```text
F1 = 0.8846
ROC-AUC = 0.9173
```

## Evaluation Rule

The test set is kept frozen after this baseline step.

Neural-network architecture choices, training settings, and hyperparameter tuning should use the training/validation data only. The held-out test set should be used again only for the final comparison against the classical baseline.

## How to Run

Place `heart.csv` in one of these locations:

```text
heart.csv
data/heart.csv
final-project/heart-disease-classification/data/heart.csv
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

Open and run the notebook:

```bash
jupyter notebook Week6_Day1_Sprint1_Baseline.ipynb
```

Run all cells from top to bottom. The notebook will create the `outputs/` folder if it does not already exist.


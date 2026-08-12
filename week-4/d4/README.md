# Week 4 — Day 4: Feature Engineering and Hyperparameter Tuning

This notebook focuses on two parts of the Week 4 workflow: creating useful features from existing data and tuning a model systematically with `GridSearchCV`.

The work starts with the COVID-19 classification dataset used in the earlier internship exercises, then applies the same process to the Credit Card Default dataset. The second dataset is included because its repeated monthly payment and billing history provides more room for meaningful feature engineering than the mostly binary COVID-19 symptom indicators.

The main workflow in both experiments is:

**baseline → feature engineering → 5-fold cross-validation → GridSearchCV → comparison**

## Objectives

The notebook demonstrates how to:

- create and justify new features,
- compare engineered features against an untuned baseline,
- distinguish model parameters from hyperparameters,
- use `StratifiedKFold` for imbalanced classification,
- tune Random Forest hyperparameters with `GridSearchCV`,
- report cross-validation mean and standard deviation,
- compare baseline, engineered, and tuned models,
- identify which engineered feature had the largest observed effect,
- identify which tested hyperparameter had the largest observed effect,
- keep the held-out test set outside feature selection and tuning.

## Notebook

`04_feature_engineering_hyperparameter_tuning.ipynb`

The notebook is divided into two experiments:

1. **Part A — COVID-19**
   - continues the classification work from the earlier internship notebooks,
   - creates two additional features,
   - tunes a Random Forest using 5-fold cross-validation.

2. **Part B — Credit Card Default**
   - uses a richer tabular dataset with six months of financial history,
   - creates eight behavior-based features,
   - tests the features individually and together,
   - tunes a Random Forest while accounting for class imbalance.

---

## Part A — COVID-19 Experiment

### Dataset

The notebook loads the COVID-19 dataset already stored in Week 3:

```text
../../week-3/corona dataset/corona_tested_individuals_ver_006.english.csv
```

Raw dataset:

| Item | Value |
|---|---:|
| Rows | 278,848 |
| Columns | 10 |
| Recorded dates | 51 |
| Date range | 2020-03-11 to 2020-04-30 |

Only `negative` and `positive` test results are used for binary classification. Rows with missing values in the five symptom columns are removed.

After preparation:

| Item | Value |
|---|---:|
| Modeling samples | 274,702 |
| Positive-class rate | 5.35% |
| Development samples | 219,761 |
| Held-out test samples | 54,941 |
| Working sample used for Day 4 experiments | 40,000 |

The working sample is stratified and fixed with `random_state=42` so all feature and tuning comparisons use the same data distribution.

### Original Features

The baseline Random Forest uses seven predictors:

- `cough`
- `fever`
- `sore_throat`
- `shortness_of_breath`
- `head_ache`
- `contact_with_confirmed`
- `abroad`

The last two variables are derived from `test_indication`.

### Engineered Features

Two additional features are created:

| Feature | Description |
|---|---|
| `symptom_count` | Number of recorded symptoms present for each individual |
| `days_since_start` | Number of days since 2020-03-11 |

`days_since_start` is useful in this experiment because the data was collected over a changing testing period. Its effect should still be interpreted carefully because a time-based pattern from this collection period may not generalize to another period.

### Cross-Validation

All comparisons use:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

with:

```python
scoring="f1"
```

F1-score is used because the positive class represents only about 5.35% of the prepared dataset.

### Feature Engineering Results

| Feature Set | Mean CV F1 | CV Std |
|---|---:|---:|
| Original features | 0.6475 | 0.0135 |
| Original + `symptom_count` | 0.6475 | 0.0135 |
| Original + `days_since_start` | 0.6684 | 0.0076 |
| Original + both engineered features | **0.6695** | **0.0078** |

The strongest individual engineered feature is `days_since_start`, which improves mean CV F1 by about **0.0210** over the original feature set.

### Hyperparameter Search

The COVID-19 Random Forest grid tests:

```python
{
    "n_estimators": [100, 200],
    "max_depth": [10, None],
    "min_samples_leaf": [3, 5],
}
```

This produces:

- **8 parameter combinations**
- **5 folds**
- **40 total CV fits**

Best configuration:

```text
n_estimators = 200
max_depth = None
min_samples_leaf = 5
```

Best cross-validated result:

```text
Mean CV F1 = 0.6888 ± 0.0098
```

Among the tested values, `min_samples_leaf` has the largest observed effect on mean CV F1.

### Part A Comparison

| Stage | Mean CV F1 | CV Std |
|---|---:|---:|
| Baseline — original features | 0.6475 | 0.0135 |
| Untuned — engineered features | 0.6695 | 0.0078 |
| Tuned — engineered features | **0.6888** | **0.0098** |

Baseline-to-tuned improvement:

- Absolute F1 gain: **+0.0413**
- Relative improvement: **6.38%**

---

## Part B — Credit Card Default Experiment

### Why a Second Dataset?

The COVID-19 dataset is useful for continuing the earlier classification work, but most of its predictors are binary indicators. That limits the number of meaningful transformations that can be created.

The Credit Card Default dataset contains repeated monthly information about repayment status, bill amounts, and payment amounts. This makes it more suitable for demonstrating feature engineering based on behavior over time.

### Dataset

The notebook reads:

```text
data/default_credit_card_clients.csv
```

The data is based on the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

Official dataset page:

https://archive.ics.uci.edu/ml/datasets/default%2Bof%2Bcredit%2Bcard%2Bclients

Dataset summary:

| Item | Value |
|---|---:|
| Rows | 30,000 |
| Columns in the CSV | 25 |
| Original model predictors after removing `ID` and target | 23 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Default class | 6,636 |
| No-default class | 23,364 |
| Default rate | 22.12% |
| Development samples | 24,000 |
| Held-out test samples | 6,000 |
| Working sample used for Day 4 experiments | 12,000 |

The target is:

```text
default payment next month
```

with:

```text
0 = no default
1 = default
```

Because the default class represents only **22.12%** of the data, accuracy alone would be misleading. Predicting the majority class for every record would already produce 77.88% accuracy while detecting no default cases. For this reason, the notebook uses **F1-score for class 1** as the primary tuning metric.

### Engineered Credit Features

Eight features are created from the six-month payment history:

| Feature | Description |
|---|---|
| `months_with_delay` | Number of months with a positive repayment delay |
| `max_delay` | Worst repayment delay observed across the six months |
| `avg_delay` | Average non-negative repayment delay |
| `avg_bill_amount` | Mean billed amount across the six months |
| `avg_credit_utilization` | Average bill amount divided by the credit limit |
| `payment_to_bill_ratio` | Total payments divided by total non-negative billed amount |
| `bill_change_ratio` | Difference between most recent and oldest bill, scaled by credit limit |
| `payment_change_ratio` | Difference between most recent and oldest payment, scaled by credit limit |

The feature count increases from **23** original predictors to **31** predictors after adding all engineered features.

### Feature Engineering Results

The same untuned Random Forest and the same five stratified folds are used for every feature comparison.

| Feature Set | Mean CV F1 | Change vs. Baseline |
|---|---:|---:|
| Original features | 0.4640 | — |
| Original + `max_delay` | 0.4730 | +0.0090 |
| Original + `avg_bill_amount` | 0.4693 | +0.0054 |
| Original + `months_with_delay` | 0.4693 | +0.0053 |
| Original + all engineered features | **0.4734** | **+0.0094** |

`max_delay` is the strongest individual engineered feature in this experiment.

Not every reasonable feature improves the model. For example, `payment_to_bill_ratio` reduces mean CV F1 slightly when added by itself. The notebook keeps this result because feature engineering is evaluated from measured performance rather than assumption.

### Hyperparameter Search

The Credit Card Random Forest grid tests:

```python
{
    "max_depth": [8, None],
    "min_samples_leaf": [1, 5],
    "class_weight": [None, "balanced"],
}
```

This again produces:

- **8 parameter combinations**
- **5 folds**
- **40 total CV fits**

Best configuration:

```text
class_weight = balanced
max_depth = 8
min_samples_leaf = 1
```

Best cross-validated result:

```text
Mean CV F1 = 0.5393 ± 0.0094
```

The value `0.5393` is an **F1-score**, not an accuracy score.

Among the tested hyperparameters, `class_weight` has the largest observed effect on mean CV F1. This is consistent with the class imbalance in the target.

### Part B Comparison

| Stage | Mean CV F1 | CV Std |
|---|---:|---:|
| Baseline — original features | 0.4640 | 0.0146 |
| Untuned — all engineered features | 0.4734 | 0.0192 |
| Tuned — engineered features | **0.5393** | **0.0094** |

Observed gains:

| Comparison | Absolute F1 Gain | Relative Improvement |
|---|---:|---:|
| Feature engineering vs. baseline | +0.0094 | 2.03% |
| Tuning vs. engineered model | +0.0659 | 13.92% |
| Tuned model vs. baseline | **+0.0753** | **16.24%** |

The larger improvement in Part B comes from hyperparameter tuning, especially the use of balanced class weights.

---

## Overall Results

| Experiment | Baseline CV F1 | Engineered CV F1 | Tuned CV F1 | Absolute Gain | Relative Gain |
|---|---:|---:|---:|---:|---:|
| COVID-19 | 0.6475 | 0.6695 | **0.6888** | +0.0413 | 6.38% |
| Credit Card Default | 0.4640 | 0.4734 | **0.5393** | +0.0753 | 16.24% |

The raw F1-scores should not be used to decide which dataset is "better." The two problems have different class distributions, features, and prediction difficulty. The useful comparison is the improvement from each experiment's own baseline.

## Main Findings

1. Feature engineering should be measured rather than assumed to help.
2. `days_since_start` is the strongest individual engineered feature in the COVID-19 experiment.
3. `max_delay` is the strongest individual engineered feature in the Credit Card Default experiment.
4. Hyperparameter tuning improves the Random Forest in both experiments.
5. `min_samples_leaf` has the largest observed hyperparameter effect in the tested COVID-19 grid.
6. `class_weight` has the largest observed effect in the tested Credit Card grid.
7. F1-score is more useful than raw accuracy for the imbalanced targets used here.
8. Cross-validation mean and standard deviation provide more information than a single validation score.
9. The held-out test sets are intentionally kept outside Day 4 feature selection and hyperparameter tuning.

## Project Structure

Expected Day 4 structure:

```text
week-4/
└── d4/
    ├── 04_feature_engineering_hyperparameter_tuning.ipynb
    ├── README.md
    └── data/
        └── default_credit_card_clients.csv
```

The notebook also depends on the existing Week 3 COVID-19 dataset:

```text
week-3/
└── corona dataset/
    └── corona_tested_individuals_ver_006.english.csv
```

Keeping this structure unchanged allows both relative dataset paths in the notebook to work correctly.

## Requirements

The notebook uses:

- Python
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- scikit-learn

The executed notebook metadata records Python **3.14.4**. Exact package versions are not pinned in the notebook.

A minimal installation is:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Run

From the repository:

```bash
cd week-4/d4
jupyter notebook
```

Open:

```text
04_feature_engineering_hyperparameter_tuning.ipynb
```

Then run the notebook from top to bottom.

Before running, confirm that these files exist:

```text
../../week-3/corona dataset/corona_tested_individuals_ver_006.english.csv
data/default_credit_card_clients.csv
```

## Reproducibility

The notebook uses fixed random seeds where applicable:

```python
random_state=42
```

It also keeps the comparison design fixed within each experiment:

- the same working sample,
- the same 5-fold `StratifiedKFold`,
- the same F1 scoring rule,
- the same baseline model when testing engineered features.

This makes the before-and-after comparisons easier to interpret.

## Evaluation Boundary

The held-out test sets are created before feature comparison and hyperparameter tuning and remain outside the Day 4 model-development process.

The reported results are therefore **cross-validation development results**, not final held-out test estimates.

Final held-out evaluation belongs after the complete preprocessing and modeling workflow has been selected.

## Scope and Limitations

- The COVID-19 `days_since_start` feature may capture changes specific to the data-collection period and should not automatically be treated as a general medical predictor.
- The Credit Card Default experiment uses a fixed 12,000-row development sample to keep repeated Random Forest fitting practical.
- The parameter grids are intentionally small and interpretable. A larger search space could be explored with `RandomizedSearchCV`.
- Random Forest is tree-based, so numeric feature scaling is not required for these experiments.
- The notebook is an educational model-development exercise and is not intended as a production medical or credit-risk decision system.

## Context

**BinX Tech AI & Machine Learning Internship — Week 4, Day 4**

Topic: **Feature Engineering and Hyperparameter Tuning**

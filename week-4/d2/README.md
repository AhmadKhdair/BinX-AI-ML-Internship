# Week 4 — Day 2: Cross-Validation

This notebook evaluates whether the Logistic Regression result from Day 1 remains stable when the validation data changes.

Instead of relying on a single validation split, it applies **5-fold cross-validation** to the development data, compares regular `KFold` with `StratifiedKFold`, measures fold-to-fold variation, and re-evaluates the probability-threshold adjustment introduced on Day 1.

The held-out test set is kept outside the cross-validation and threshold-search process.

## Overview

The main question in this notebook is not whether a new model can beat the previous one, but whether the previous evaluation was **stable and representative**.

The notebook therefore keeps the same:

- COVID-19 dataset,
- feature set,
- target definition,
- Logistic Regression configuration,
- F1-score evaluation metric,

and changes the **evaluation strategy** from one validation split to repeated validation across five stratified folds.

## Learning Objectives

By the end of the notebook, the workflow demonstrates how to:

- understand why a single validation split can be insufficient,
- apply 5-fold cross-validation to development data,
- distinguish `KFold` from `StratifiedKFold`,
- preserve class proportions in an imbalanced classification problem,
- interpret per-fold scores, mean cross-validation performance, and standard deviation,
- compare a single validation estimate with a cross-validation estimate,
- evaluate a custom probability threshold across multiple folds,
- search nearby thresholds without touching the held-out test set,
- avoid selecting the model from the “best” fold,
- refit the selected configuration on the full development set after tuning.

## Dataset and Modeling Setup

The notebook loads the same COVID-19 dataset used in the previous work.

```text
../../week-3/corona dataset/corona_tested_individuals_ver_006.english.csv
```

### Dataset Summary

| Item | Value |
|---|---:|
| Raw dataset rows | 278,848 |
| Raw dataset columns | 10 |
| Modeling samples | 274,702 |
| Number of model features | 7 |
| Negative samples | 260,008 |
| Positive samples | 14,694 |
| Positive-class rate | 5.35% |
| Development samples | 219,761 |
| Held-out test samples | 54,941 |

Only rows whose target is `negative` or `positive` are used for modeling, and rows missing any of the five symptom features are removed.

### Features

The model uses seven binary features:

1. `cough`
2. `fever`
3. `sore_throat`
4. `shortness_of_breath`
5. `head_ache`
6. `contact_with_confirmed`
7. `abroad`

The last two variables are derived from `test_indication`:

- `contact_with_confirmed = 1` when the indication is `Contact with confirmed`,
- `abroad = 1` when the indication is `Abroad`.

The target is encoded as:

```text
negative -> 0
positive -> 1
```

The feature set is intentionally kept unchanged from Day 1 so that the comparison isolates the effect of the **evaluation method** rather than mixing it with feature changes.

## Why F1-Score?

Only about **5.35%** of the modeling samples belong to the positive class, so the target is strongly imbalanced.

For this notebook, **F1-score** remains the main evaluation metric because the experiment is focused on balancing performance on the positive class rather than relying on overall accuracy alone.

## Evaluation Design

The full modeling dataset is first separated into development and test data using stratification:

```text
Full modeling data: 274,702 samples
│
├── 80% Development: 219,761 samples
│   └── used for validation, cross-validation, and threshold selection
│
└── 20% Test: 54,941 samples
    └── kept outside the folds and threshold search
```

The split uses:

```python
random_state=42
stratify=y
```

For comparison with Day 1, the notebook also recreates the previous single train/validation split inside the development set:

| Split | Samples | Approx. share of full modeling data |
|---|---:|---:|
| Training | 164,820 | 60% |
| Validation | 54,941 | 20% |
| Test | 54,941 | 20% |

## Model

The notebook uses the same Logistic Regression configuration as Day 1:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
)
```

The goal is to evaluate the stability of the existing setup, not to introduce a new model family.

## K-Fold vs. Stratified K-Fold

Both methods rotate which subset is used for validation, but they differ in how the folds are constructed.

- `KFold` splits samples into folds without explicitly preserving the target distribution.
- `StratifiedKFold` preserves the class proportions as closely as possible in each fold.

Because the positive class is only about 5% of the dataset, the notebook uses **Stratified K-Fold** for model evaluation.

### Observed Validation-Fold Class Balance

| Fold | KFold positive rate | StratifiedKFold positive rate |
|---:|---:|---:|
| 1 | 5.331% | 5.349% |
| 2 | 5.258% | 5.349% |
| 3 | 5.479% | 5.349% |
| 4 | 5.253% | 5.349% |
| 5 | 5.424% | 5.349% |

The ordinary K-Fold percentages are already fairly close because the dataset is large, but stratification keeps the target ratio essentially identical across all five folds.

## Cross-Validation Configuration

The main cross-validation experiment uses:

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

Each round trains a fresh Logistic Regression model on four folds and evaluates it on the remaining fold.

With five folds, each round uses approximately:

- **80% of the development set for training**, and
- **20% of the development set for validation**.

These percentages are within the development data, not the full dataset.

## Default Logistic Regression Results

### Per-Fold F1-Scores

| Fold | F1 |
|---:|---:|
| 1 | 0.6536 |
| 2 | 0.6657 |
| 3 | 0.6617 |
| 4 | 0.6586 |
| 5 | 0.6496 |

The fold scores stay within a relatively narrow range.

### Cross-Validation Summary

```text
Mean CV F1: 0.6579
CV F1:      0.6579 ± 0.0057
```

The small standard deviation indicates that the Logistic Regression result is fairly stable across the five development splits.

## Comparison with the Day 1 Validation Split

| Evaluation method | F1 estimate | Score standard deviation |
|---|---:|---:|
| Day 1 — Single Validation Split | 0.6567 | — |
| Day 2 — 5-Fold Cross-Validation | 0.6579 | 0.0057 |

The difference between `0.6567` and `0.6579` is very small.

This is not treated as a model improvement. Instead, the cross-validation result provides stronger evidence that the Day 1 validation score was reasonably representative rather than being caused by one unusually favorable or difficult validation split.

## Threshold Evaluation Across Folds

Day 1 found that lowering the Logistic Regression probability threshold to `0.20` improved the single-validation F1.

This notebook checks whether that improvement remains when evaluated across all five development folds.

### Threshold 0.20

| Configuration | Mean CV F1 | CV Std |
|---|---:|---:|
| Default Logistic Regression | 0.6579 | 0.0057 |
| Logistic Regression, threshold = 0.20 | 0.6675 | 0.0060 |

Using the `0.20` threshold improves mean CV F1 by approximately:

```text
+0.0096
```

The standard deviation changes only slightly, suggesting that the higher F1 does not come with a large loss of fold-to-fold stability.

## Nearby Threshold Search

The notebook then evaluates thresholds from `0.12` through `0.30` using the same five stratified folds.

For efficiency, Logistic Regression is fitted once per fold, validation probabilities are stored, and F1 is then calculated for each candidate threshold without retraining the model for every threshold value.

The threshold is selected by **mean cross-validation F1**, not by the best score from an individual fold.

### Best Observed Development-CV Result

```text
Best threshold:              0.12
Best mean CV F1:             0.6679 ± 0.0057
Improvement over default:    0.0100
Improvement over threshold .20: 0.0004
```

Several thresholds from approximately `0.12` to `0.19` produce nearly identical mean F1 values.

Therefore, the useful conclusion is not that `0.12` is clearly superior to `0.20`. The difference is only about `0.0004`, which is much smaller than the fold-to-fold variation. Instead, the experiment identifies a **broad good threshold region** below the default decision threshold.

## Final Refit

After the threshold comparison is complete, the notebook:

1. selects the threshold with the highest mean development-CV F1,
2. creates a fresh Logistic Regression model,
3. fits that model on all `219,761` development samples.

The selected threshold in the executed notebook is:

```text
0.12
```

This refit uses all available development data after the tuning process is finished.

## Important Test-Set Boundary

The held-out test set is **not used** in:

- fold construction,
- cross-validation scoring,
- mean or standard-deviation calculation,
- threshold comparison,
- nearby-threshold selection.

The test set had already been evaluated in Day 1, so this notebook does not reuse it to select the new threshold or claim a new final test estimate.

The best-threshold cross-validation result should therefore be interpreted as a **development/tuning result**, not as a fresh unbiased test-set performance estimate.

## Notebook Roadmap

The notebook is organized into five main stages:

### 1. Setup and baseline reproduction

Sections 1–6:

- import libraries,
- load the dataset,
- recreate the modeling data,
- rebuild the development/test split,
- reproduce the Day 1 validation result,
- reuse the same Logistic Regression configuration.

### 2. Cross-validation design

Sections 7–9:

- introduce 5-fold cross-validation,
- compare `KFold` and `StratifiedKFold`,
- inspect class balance and fold sizes.

### 3. Default model stability

Sections 10–15:

- run 5-fold CV,
- inspect individual fold scores,
- calculate mean F1,
- calculate standard deviation,
- compare CV with the Day 1 validation result,
- explain why the best fold model is not retained.

### 4. Threshold validation and tuning

Sections 16–18:

- evaluate the Day 1 threshold of `0.20` across all folds,
- search nearby thresholds,
- compare mean F1 and stability,
- refit the highest-mean configuration on the full development set.

### 5. Evaluation boundary and conclusions

Sections 19–20:

- clarify the role of the held-out test set,
- summarize the engineering conclusions from the experiment.

## Key Findings

The main results are:

1. **The Day 1 Logistic Regression validation result is stable.**  
   The single-validation F1 of `0.6567` is close to the 5-fold mean of `0.6579`.

2. **Fold-to-fold variation is small.**  
   The default model has a CV standard deviation of only `0.0057`.

3. **Stratification is appropriate for this imbalanced target.**  
   It keeps the positive-class rate essentially constant across folds.

4. **Lowering the probability threshold consistently improves F1.**  
   Threshold `0.20` increases mean CV F1 to `0.6675`.

5. **The best threshold is not sharply defined.**  
   A broad region around `0.12–0.19` performs almost identically, so the result should be interpreted as evidence for a useful lower-threshold region rather than proof that one exact value is uniquely optimal.

6. **Cross-validation is used for evaluation and tuning, not for selecting the luckiest fold model.**

7. **The test set remains outside the Day 2 tuning process.**  
   No new final test estimate is reported in this notebook.

## Requirements

The notebook uses:

- Python
- Jupyter Notebook or JupyterLab
- pandas
- scikit-learn

The notebook metadata records:

```text
Python 3.14.4
```

Exact package versions are not stored in the notebook metadata.

A minimal installation is:

```bash
pip install pandas scikit-learn jupyter
```

## How to Run

1. Ensure the dataset exists at the relative path expected by the notebook:

   ```text
   ../../week-3/corona dataset/corona_tested_individuals_ver_006.english.csv
   ```

2. Install the required Python packages:

   ```bash
   pip install pandas scikit-learn jupyter
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

   or:

   ```bash
   jupyter lab
   ```

4. Open the cross-validation notebook and run the cells from top to bottom.

Running the notebook in order is important because later cells depend on variables, splits, trained models, and cross-validation objects created earlier.

## Technical Notes

- The random seed is fixed with `random_state=42` where applicable for reproducibility.
- The development/test split is stratified by the target.
- The five cross-validation folds are shuffled and stratified.
- F1-score is used consistently as the main metric.
- Threshold tuning uses validation probabilities from the development folds only.
- The nearby-threshold loop avoids unnecessary model retraining by reusing each fold's predicted probabilities.
- The cross-validation mean is used instead of selecting the highest-scoring individual fold.

## Scope and Limitations

This notebook is intentionally focused on **evaluation methodology**, not a broad model-search experiment.

It does not:

- compare new model families,
- change the Day 1 feature set,
- perform a new unbiased final test evaluation,
- claim that threshold `0.12` is meaningfully better than every nearby threshold,
- pin exact pandas or scikit-learn package versions.

These boundaries keep the experiment focused on the central question: **does the model's validation performance remain stable when the validation data changes?**

## Conclusion

The 5-fold stratified cross-validation experiment supports the Day 1 result: the default Logistic Regression performance is stable across different validation subsets.

The larger practical finding is that lowering the decision threshold improves F1 consistently across the folds, while performance remains similarly stable. The experiment therefore provides stronger evidence for using a lower-threshold Logistic Regression configuration than a single validation split alone could provide.

At the same time, the nearby-threshold results are almost flat across several values. The correct interpretation is therefore a robust **good threshold region**, not an overconfident claim that one exact threshold is uniquely best.

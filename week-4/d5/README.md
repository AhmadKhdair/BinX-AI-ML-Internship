# Day 5 — Scikit-learn Pipelines & Tuned Mini-Project

This notebook completes Week 4 by putting the preprocessing and model training steps into a single Scikit-learn pipeline.

I continued using the **Credit Card Default** dataset from Day 4 so I could focus on the pipeline workflow instead of starting again with a new dataset.

## Dataset

The dataset contains **30,000 customer records** and the target column is:

`default payment next month`

* `0` → No default
* `1` → Default

The `ID` column is removed before training.

The dataset is reused from:

```text
week-4/d4/data/default_credit_card_clients.csv
```

## Train / Test Split

I first separated the data into:

* 80% development data
* 20% held-out test data

The split was stratified to keep the same class distribution in both sets.

The test set was kept outside cross-validation and hyperparameter tuning and was only used after the final pipeline settings were selected.

## Feature Engineering

I reused the same eight engineered features from Day 4:

* `months_with_delay`
* `max_delay`
* `avg_delay`
* `avg_bill_amount`
* `avg_credit_utilization`
* `payment_to_bill_ratio`
* `bill_change_ratio`
* `payment_change_ratio`

These features summarize repayment delays, bill behavior, credit utilization, and payment behavior.

After feature engineering, the model uses **31 predictors**.

## Preprocessing

The dataset contains both numeric and categorical features, so I used a `ColumnTransformer`.

### Numeric features

Numeric columns are processed with:

```python
StandardScaler()
```

### Categorical features

The following columns are treated as categorical:

```text
SEX
EDUCATION
MARRIAGE
```

They are processed with:

```python
OneHotEncoder(handle_unknown="ignore")
```

The numeric and categorical preprocessing steps are then combined with the model inside one `Pipeline`.

## Pipeline

The final structure is:

```text
Input features
      ↓
ColumnTransformer
 ├── Numeric → StandardScaler
 └── Categorical → OneHotEncoder
      ↓
RandomForestClassifier
```

Keeping preprocessing inside the pipeline is important because scaling and encoding are fitted separately inside each training fold during cross-validation instead of being fitted on the full dataset.

## Baseline Model

The baseline model is a Random Forest with:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

It was evaluated using **5-fold Stratified Cross-Validation** with F1-score as the main metric.

Baseline result:

```text
CV F1 = 0.4697 ± 0.0127
```

## Hyperparameter Tuning

I used `GridSearchCV` to tune the Random Forest inside the pipeline.

The search included:

```python
param_grid = {
    "model__max_depth": [8, None],
    "model__min_samples_leaf": [1, 5],
    "model__class_weight": [None, "balanced"],
}
```

The search used:

* 5-fold `StratifiedKFold`
* F1-score
* the same development data used for the baseline comparison

The best parameters were:

```text
class_weight = balanced
max_depth = None
min_samples_leaf = 5
```

Best cross-validation result:

```text
CV F1 = 0.5407 ± 0.0070
```

Compared with the baseline:

```text
Baseline CV F1 = 0.4697
Tuned CV F1    = 0.5407

Improvement    = +0.0709
```

## Final Test Evaluation

After selecting the best hyperparameters, both pipelines were fitted using the full development set.

The held-out test set was then evaluated once.

| Model             | Accuracy | Precision | Recall |         F1 |
| ----------------- | -------: | --------: | -----: | ---------: |
| Baseline Pipeline |   0.8155 |    0.6414 | 0.3760 |     0.4741 |
| Tuned Pipeline    |   0.8015 |    0.5542 | 0.5237 | **0.5386** |

The tuned pipeline improved test F1 by:

```text
+0.0644
```

The main change was in recall.

```text
Baseline Recall = 0.3760
Tuned Recall    = 0.5237
```

The tuned model detects more customers who actually default, while losing some precision and overall accuracy.

Since F1-score was chosen as the main metric before tuning, the tuned pipeline is the better model for this experiment.

## Confusion Matrix

The tuned model also reduced the number of false negatives:

```text
Baseline false negatives = 828
Tuned false negatives    = 632
```

This matches the improvement seen in recall.

## Why the Pipeline Matters

The main goal of Day 5 was not to introduce a new model or a new metric.

The goal was to organize the complete workflow correctly.

Using `Pipeline` and `ColumnTransformer` means that preprocessing is part of the model training process itself.

During cross-validation:

1. the training fold is used to fit the scaler and encoder,
2. the same transformations are applied to the validation fold,
3. the model is trained,
4. F1 is calculated on the validation fold.

This prevents preprocessing information from leaking from the validation or test data into training.

## Final Workflow

```text
Train / Test Split
        ↓
Day 4 Feature Engineering
        ↓
ColumnTransformer
        ↓
Pipeline
        ↓
5-Fold Stratified CV
        ↓
GridSearchCV
        ↓
Best Parameters
        ↓
Refit on Development Data
        ↓
Final Held-Out Test Evaluation
```

## Files

```text
week-4/
├── d4/
│   └── data/
│       └── default_credit_card_clients.csv
│
└── d5/
    ├── 05_scikit_learn_pipelines_tuned_mini_project.ipynb
    └── README.md
```

## Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Git & GitHub

Main Scikit-learn tools used:

```text
train_test_split
StratifiedKFold
cross_val_score
StandardScaler
OneHotEncoder
ColumnTransformer
Pipeline
RandomForestClassifier
GridSearchCV
```

## Conclusion

Day 5 combined the work from the previous days into one complete workflow.

The main improvement was moving preprocessing and model training into a single pipeline and tuning that pipeline using cross-validation.

The baseline achieved an F1-score of **0.4741** on the held-out test set, while the tuned pipeline reached **0.5386**.

More importantly, the final model was selected without using the test set during tuning, and preprocessing remained inside the cross-validation workflow.

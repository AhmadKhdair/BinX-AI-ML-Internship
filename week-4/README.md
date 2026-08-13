# Week 4 — Model Evaluation, Tuning & Pipelines

Week 4 focused on making model evaluation more reliable.

In the previous week, the main goal was getting supervised learning models to work. This week went further: how to split data correctly, how to get a more reliable estimate with cross-validation, how to recognize overfitting and underfitting, how to improve the input features, and finally how to put preprocessing and modeling into one pipeline.

The work is split into five days, with each day building on the previous one.

## Week Structure

| Day   | Topic                                        |
| ----- | -------------------------------------------- |
| Day 1 | Train / Validation / Test Splits             |
| Day 2 | Cross-Validation                             |
| Day 3 | Bias-Variance and Model Fit                  |
| Day 4 | Feature Engineering and GridSearchCV         |
| Day 5 | Scikit-learn Pipelines and Final Tuned Model |

---

## Day 1 — Train / Validation / Test Splits

The first day was about separating model development from final evaluation.

Instead of using the test set while making decisions, the data was divided into:

* training data for fitting the model
* validation data for comparing choices
* test data for the final evaluation

The important rule was to keep the test set out of the tuning process.

Several model choices were compared using the development data, while F1-score was used as an important metric for the imbalanced classification problem.

The final test result was checked only after the model choice was finished.

**Main idea:** the test set should measure the final model, not help build it.

---

## Day 2 — Cross-Validation

A single validation split can give a score that depends too much on which rows happened to be selected.

Day 2 replaced that single estimate with **5-fold cross-validation**.

I used `StratifiedKFold` so the class distribution stayed similar across folds and reported both:

```text
Mean CV score
Standard deviation
```

This gave a better view of both performance and stability.

The cross-validation result was also compared with the single-split result from Day 1.

**Main idea:** one split gives one estimate; cross-validation gives a more reliable picture.

---

## Day 3 — Bias, Variance and Model Fit

Day 3 focused on understanding why a model performs badly instead of immediately changing models or tuning parameters.

I created examples of both:

### Underfitting

The model was too simple and performed poorly on both the training and validation data.

### Overfitting

The model performed very well on training data but noticeably worse on unseen data.

The train-vs-validation gap was used to identify the problem, then model complexity was adjusted to improve generalization.

This day made the connection between:

```text
Model complexity
      ↓
Bias / Variance
      ↓
Train vs validation performance
```

**Main idea:** before tuning a model, understand what kind of problem it has.

---

## Day 4 — Feature Engineering & Hyperparameter Tuning

Day 4 moved from diagnosing models to improving them.

### Feature Engineering

New features were created from existing columns to give the model more useful information.

For the Credit Card Default experiment, the engineered features summarized customer behavior across several months, including:

```text
months_with_delay
max_delay
avg_delay
avg_bill_amount
avg_credit_utilization
payment_to_bill_ratio
bill_change_ratio
payment_change_ratio
```

The goal was not to assume that every new feature would improve the model. The engineered version still had to be compared with the baseline.

### GridSearchCV

Instead of manually trying different Random Forest settings, I used `GridSearchCV`.

The search evaluated different hyperparameter combinations with cross-validation and selected them using F1-score.

This made the tuning process systematic and kept model selection based on development data rather than the final test set.

**Main idea:** feature engineering changes what the model learns from, while hyperparameter tuning changes how the model learns.

---

## Day 5 — Pipelines

The last day connected the previous work into one workflow.

The Credit Card Default experiment from Day 4 was continued using the same engineered features.

The data contains both numeric and categorical columns, so preprocessing was handled with a `ColumnTransformer`.

Numeric columns:

```text
StandardScaler
```

Categorical columns:

```text
SEX
EDUCATION
MARRIAGE
        ↓
OneHotEncoder
```

The preprocessing step and Random Forest were then combined into one Scikit-learn `Pipeline`.

```text
Features
   ↓
ColumnTransformer
   ├── Numeric → StandardScaler
   └── Categorical → OneHotEncoder
   ↓
RandomForestClassifier
```

This matters during cross-validation because the scaler and encoder are fitted only on the training part of each fold.

The pipeline itself was then passed to `GridSearchCV`.

### Baseline

The untuned pipeline produced:

```text
5-Fold CV F1 = 0.4697 ± 0.0127
```

### Tuned Pipeline

The best parameters were:

```text
class_weight     = balanced
max_depth        = None
min_samples_leaf = 5
```

Cross-validation improved to:

```text
5-Fold CV F1 = 0.5407 ± 0.0070
```

After tuning was complete, the final pipelines were fitted on the full development set and evaluated on the held-out test set.

| Model             | Accuracy | Precision | Recall |         F1 |
| ----------------- | -------: | --------: | -----: | ---------: |
| Baseline Pipeline |   0.8155 |    0.6414 | 0.3760 |     0.4741 |
| Tuned Pipeline    |   0.8015 |    0.5542 | 0.5237 | **0.5386** |

The tuned pipeline improved F1 from **0.4741 to 0.5386**.

It also increased recall from **0.3760 to 0.5237**, meaning more of the actual default cases were detected.

Since F1 was selected as the main metric before tuning, the tuned pipeline was kept as the final model.

---

## Week 4 Workflow

By the end of the week, the separate topics came together into one process:

```text
Raw Data
   ↓
Train / Validation / Test Discipline
   ↓
Cross-Validation
   ↓
Bias / Variance Diagnosis
   ↓
Feature Engineering
   ↓
Hyperparameter Tuning
   ↓
ColumnTransformer
   ↓
Pipeline
   ↓
Final Held-Out Test
```

The biggest difference from the beginning of the week is that model performance is no longer based on one convenient score.

The final workflow keeps model selection inside the development data, uses cross-validation during tuning, and leaves the held-out test set for the end.

---

## Repository Structure

```text
week-4/
├── d1/
│   └── Train / Validation / Test Split
│
├── d2/
│   └── Cross-Validation
│
├── d3/
│   └── Bias-Variance & Model Fit
│
├── d4/
│   ├── Feature Engineering
│   ├── GridSearchCV
│   └── data/
│       └── default_credit_card_clients.csv
│
└── d5/
    └── Scikit-learn Pipeline & Tuned Mini-Project
```

Each day contains its own notebook and notes for the experiment.

---

## Main Tools Used

```text
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Jupyter Notebook
Git
GitHub
```

Scikit-learn tools used during the week include:

```text
train_test_split
cross_val_score
StratifiedKFold
DecisionTree
RandomForestClassifier
GridSearchCV
StandardScaler
OneHotEncoder
ColumnTransformer
Pipeline
```

---

## Week 4 Summary

Week 4 was less about adding more models and more about making the modeling process trustworthy.

The main lessons were:

* keep the test set out of model selection
* use cross-validation instead of trusting one split
* compare training and validation results before deciding how to fix a model
* measure whether engineered features actually help
* tune hyperparameters systematically
* keep preprocessing inside the pipeline during cross-validation

The final Day 5 notebook brings these ideas together in one tuned workflow, with the final evaluation performed only after the model and preprocessing choices were fixed.

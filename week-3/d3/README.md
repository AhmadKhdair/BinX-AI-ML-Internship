# Week 3 — Day 3: Logistic Regression Classification

## Overview

This notebook continues the Week 3 supervised learning work by applying Logistic Regression to a binary classification problem.

The same COVID-19 dataset prepared during the previous day is used so that the modeling stage remains connected to the earlier data-cleaning and target-encoding work.

## Objectives

The main goals of this notebook are to:

* Understand how Logistic Regression converts a linear score into a probability.
* Prepare the feature matrix and encoded target for modeling.
* Split the dataset into training and testing sets.
* Preserve the target-class distribution during the split.
* Build a simple baseline model for comparison.
* Train Logistic Regression using a preprocessing pipeline.
* Evaluate the model with several classification metrics.
* Examine prediction errors using a confusion matrix.
* Interpret the direction and relative strength of the learned coefficients.

## Work Completed

### Data Validation

The prepared features and target were checked before training to confirm that:

* The feature columns are numeric.
* No missing values remain in the modeling data.
* The target contains two classes.
* The feature matrix and target contain the same number of observations.

### Class Distribution

The target distribution was inspected to determine whether the classes were balanced.

This step is important because accuracy can be misleading when one class appears much more frequently than the other.

### Train-Test Split

The dataset was divided into training and testing subsets.

A stratified split was used to preserve approximately the same class proportions in both subsets. A fixed random state was also used to keep the results reproducible.

### Baseline Model

A `DummyClassifier` was trained using the most frequent class strategy.

This model provides a basic reference point. The Logistic Regression model should perform better than a classifier that ignores all input features.

### Logistic Regression Pipeline

The model was built using a Scikit-learn pipeline containing:

1. `StandardScaler`
2. `LogisticRegression`

The scaler standardizes the numeric features before training. Keeping the scaler inside the pipeline ensures that it learns only from the training data and helps prevent data leakage.

### Model Evaluation

The trained model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

These metrics provide a more complete view of performance than accuracy alone.

### Error Analysis

A confusion matrix was used to separate predictions into:

* True negatives
* False positives
* False negatives
* True positives

This made it possible to identify the exact types of mistakes made by the model.

### Coefficient Interpretation

The learned Logistic Regression coefficients were inspected after feature scaling.

Positive coefficients move the prediction toward class `1`, while negative coefficients move it toward class `0`. Features with larger absolute coefficient values have a stronger effect on the model score.

The coefficients describe relationships learned from the dataset and should not be treated as proof of causation.

## Main Concepts

* Binary classification
* Logistic Regression
* Sigmoid function
* Probability threshold
* Stratified train-test split
* Baseline comparison
* Feature scaling
* Data leakage prevention
* Classification metrics
* Confusion matrix
* ROC curve
* Model coefficients

## Tools and Libraries

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Notebook

```text
03_logistic_regression_classification.ipynb
```

## Running the Notebook

Activate the project environment and start Jupyter Notebook:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

Open the Day 3 notebook and run the cells in order from top to bottom.

## Result

By the end of this notebook, a complete binary classification workflow was created, starting from prepared data and ending with model evaluation and interpretation.

The Logistic Regression model was compared with a simple baseline, evaluated on unseen test data, and examined using both numerical metrics and visual diagnostics.

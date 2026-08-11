# Bias, Variance, and Model Fit

A practical machine learning experiment using a Decision Tree classifier to understand **underfitting, overfitting, model complexity, and generalization** through training and cross-validation performance.

## Overview

This project investigates how model complexity affects performance on unseen data.

Rather than focusing only on achieving a high training score, the experiment compares training and cross-validation performance to diagnose:

* Underfitting and high bias
* Overfitting and high variance
* The effect of hyperparameters on model complexity
* The trade-off between training performance and generalization
* Why the train-validation gap should be treated as a diagnostic signal rather than the optimization target

The experiment is implemented in a Jupyter Notebook using a Decision Tree classifier and the Breast Cancer Wisconsin dataset provided by scikit-learn.

## Dataset

The experiment uses the **Breast Cancer Wisconsin Diagnostic dataset** available through `scikit-learn`.

* **Samples:** 569
* **Features:** 30 numeric features
* **Task:** Binary classification
* **Classes:** Malignant and benign

The original scikit-learn target is remapped so that:

* `1 = malignant`
* `0 = benign`

This makes the malignant class the positive class used for the F1-score evaluation.

The dataset contains 357 benign samples and 212 malignant samples.

## Experimental Setup

The data is divided into:

* **80% development set:** used for model evaluation and hyperparameter selection
* **20% held-out test set:** reserved and intentionally not evaluated in this notebook

The development set is evaluated using **5-fold Stratified Cross-Validation** with a fixed random seed of `42`.

### Evaluation Metrics

The main metric is **F1-score for the malignant class**.

For each model configuration, the experiment records:

* Mean training F1
* Mean cross-validation F1
* Cross-validation standard deviation
* Train-CV performance gap

A majority-class `DummyClassifier` is also used as a baseline.

## Experiments

### 1. Baseline

A `DummyClassifier` using the most-frequent strategy provides a simple reference point.

The baseline achieves:

* Accuracy: **0.626**
* Malignant F1: **0.000**

This demonstrates why accuracy alone is not sufficient for evaluating this classification task.

### 2. Deliberately Underfit Model

A highly constrained Decision Tree is created using:

```python
DecisionTreeClassifier(
    max_depth=1,
    max_features=1,
    random_state=42
)
```

Results:

| Model            | Train F1 | CV F1 | CV Std |   Gap |
| ---------------- | -------: | ----: | -----: | ----: |
| Constrained tree |    0.481 | 0.450 |  0.103 | 0.031 |

Both training and cross-validation performance are weak, indicating that the model is too restricted to capture the useful patterns in the data.

### 3. Deliberately Overfit Model

An unrestricted Decision Tree is then evaluated:

```python
DecisionTreeClassifier(
    random_state=42
)
```

Results:

| Model         | Train F1 | CV F1 | CV Std |   Gap |
| ------------- | -------: | ----: | -----: | ----: |
| Unpruned tree |    1.000 | 0.896 |  0.045 | 0.104 |

The tree achieves perfect training performance but performs worse on unseen validation folds. This provides a clear example of high variance and overfitting.

### 4. Controlling Model Complexity

The experiment uses `min_samples_leaf` to control tree complexity.

The tested values are:

```text
1, 2, 3, 5, 10, 20, 30, 50
```

The best configuration based on mean cross-validation F1 is:

```text
min_samples_leaf = 3
```

Results:

* Train F1: **0.974**
* Mean CV F1: **0.913**
* CV standard deviation: **0.028**
* Train-CV gap: **0.061**

Compared with the unrestricted tree, the selected model improves cross-validation performance from **0.896 to 0.913** while reducing the train-CV gap.

## Model Comparison

| Model                                | Train F1 |     CV F1 |    CV Std |   Gap |
| ------------------------------------ | -------: | --------: | --------: | ----: |
| Deliberately constrained tree        |    0.481 |     0.450 |     0.103 | 0.031 |
| Unpruned tree                        |    1.000 |     0.896 |     0.045 | 0.104 |
| Selected tree (`min_samples_leaf=3`) |    0.974 | **0.913** | **0.028** | 0.061 |

The results illustrate the bias-variance trade-off:

* The constrained tree has insufficient capacity and underfits.
* The unrestricted tree has enough capacity to fit the training data perfectly but shows stronger signs of overfitting.
* The selected tree provides the strongest cross-validation performance among the tested configurations.

## Key Takeaways

1. Training performance alone is not sufficient for judging a model.
2. Cross-validation provides a better estimate of how a model behaves on unseen data during development.
3. A small train-CV gap does not automatically indicate a better model.
4. High bias is associated with a model that is too constrained.
5. High variance occurs when a model becomes overly sensitive to training-specific details.
6. Hyperparameters such as `max_depth`, `max_features`, and `min_samples_leaf` can substantially affect model complexity.
7. Reducing model complexity is useful when it improves or preserves validation performance.
8. The best configuration should be selected based on reliable generalization performance rather than training performance or gap size alone.
9. The held-out test set remains untouched because cross-validation results were used during model development.
10. Regularization is another general approach for controlling effective model complexity, although Ridge and Lasso are discussed conceptually rather than fitted in this experiment.

## Project Structure

```text
.
├── 03_bias_variance_model_fit.ipynb
└── README.md
```

If additional environment or dependency files are added, they can be included here as the project evolves.

## Requirements

The notebook uses Python and the following libraries:

* NumPy
* pandas
* Matplotlib
* scikit-learn
* Jupyter Notebook / JupyterLab

## Running the Notebook

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
```

Then install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
03_bias_variance_model_fit.ipynb
```

and run the notebook from top to bottom.

## Reproducibility

The experiment uses fixed random seeds where applicable, including:

```python
random_state=42
```

The held-out test set is intentionally not used for model selection or evaluation in this experiment.

Because cross-validation is used repeatedly to compare hyperparameter configurations, the final test set is kept separate from the development process.

## Scope and Limitations

This notebook is primarily an educational experiment focused on understanding model fit and the bias-variance trade-off.

It is **not intended to represent a production-ready medical prediction system**.

The held-out test set is also intentionally left unevaluated because the objective of this experiment is model diagnosis and hyperparameter selection rather than final model approval.

## Technologies

* Python
* Jupyter Notebook
* NumPy
* pandas
* Matplotlib
* scikit-learn

## Author

Created as part of a machine learning learning workflow focused on understanding model behavior, validation, and generalization.

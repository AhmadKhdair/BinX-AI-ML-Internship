# Week 3 — Day 1: Linear Regression Fundamentals

> A complete, reproducible introduction to supervised regression using NumPy, Pandas, Matplotlib, and Scikit-learn.

[Open the notebook](./01_linear_regression.ipynb)

---

## Overview

This notebook explains how a Linear Regression model represents a numeric relationship, learns its parameters from training data, makes predictions on unseen observations, and is evaluated using both numerical metrics and residual diagnostics.

A controlled synthetic dataset is used because its true mathematical relationship is known in advance. This allows the learned parameters to be compared directly with the values that generated the data.

## At a Glance

| Item | Details |
|---|---|
| Learning type | Supervised learning |
| Task | Regression |
| Model | Ordinary Least Squares Linear Regression |
| Dataset | Synthetic, 500 observations, 5 features |
| Data split | 80% training, 20% testing |
| Main library | Scikit-learn |
| Evaluation | MSE, RMSE, R², baseline comparison, residual diagnostics |
| Reproducibility | Fixed random seed: `42` |

---

## What the Notebook Contains

The notebook is organized as a complete learning path rather than a single model-training cell.

| Section | What is covered | Why it matters |
|---:|---|---|
| 1 | Supervised learning | Introduces labeled data, features, and targets. |
| 2 | Regression vs. classification | Explains why predicting a number differs from predicting a class. |
| 3 | Linear relationships | Introduces weights, the intercept, and the prediction equation. |
| 4 | Manual prediction | Shows how each feature contributes to one prediction. |
| 5 | Dot products and matrix multiplication | Connects the mathematical equation to vectorized NumPy operations. |
| 6 | Multiple observations | Applies one weight vector to an entire feature matrix. |
| 7 | Synthetic dataset design | Creates data with known informative and irrelevant features. |
| 8 | Train/test split | Separates model learning from fair evaluation on unseen data. |
| 9 | Model training | Uses `LinearRegression().fit()` to learn weights and an intercept. |
| 10 | Parameter comparison | Compares learned parameters with the known true parameters. |
| 11 | Prediction | Uses `model.predict()` on the test set. |
| 12 | Model evaluation | Calculates MSE, RMSE, and R². |
| 13 | Baseline comparison | Verifies that the trained model adds value over predicting the mean. |
| 14 | Overfitting check | Compares training and test performance. |
| 15 | Residual diagnostics | Inspects error patterns using three diagnostic plots. |

After completing the notebook, the reader should be able to explain the full workflow:

```text
features and target
        ↓
train/test split
        ↓
model.fit(...)
        ↓
model.predict(...)
        ↓
evaluation and diagnostics
```

---

## Experiment Design

The target was generated from the following known relationship:

```math
y = 4 + 3x_1 - 2x_2 + 1.5x_3 + 0x_4 + 0x_5 + \varepsilon
```

Where:

- `x1`, `x2`, and `x3` are informative features.
- `x4` and `x5` are irrelevant features with true weights of zero.
- `4` is the true intercept.
- `ε` is random noise added to make the observations realistic.

The experiment intentionally includes irrelevant features to test whether the model keeps their learned weights close to zero.

---

## Core Machine Learning Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

The notebook explains each stage before applying it:

- `train_test_split()` creates separate learning and evaluation data.
- `fit()` learns the model parameters from the training set.
- `predict()` applies the learned relationship to unseen observations.

---

## Results

### Learned Parameters

| Parameter | True Value | Learned Value |
|---|---:|---:|
| Intercept | 4.0000 | 4.0313 |
| `x1` | 3.0000 | 3.0933 |
| `x2` | -2.0000 | -1.9932 |
| `x3` | 1.5000 | 1.4705 |
| `x4` | 0.0000 | -0.0329 |
| `x5` | 0.0000 | 0.0398 |

The learned values are close to the true values. The informative features retained meaningful weights, while the two irrelevant features remained close to zero.

### Performance

| Dataset / Model | RMSE | R² |
|---|---:|---:|
| Training set | 1.0250 | 0.9316 |
| Test set | 0.9378 | 0.9423 |
| Mean baseline | 3.9041 | -0.0001 |

Additional test metric:

- **Test MSE:** `0.8794`

### Interpretation

- The model substantially outperformed the mean-prediction baseline.
- Training and test scores were similar, so there was no clear sign of overfitting.
- The test RMSE was close to the scale of the intentionally added random noise.
- The high test R² indicates that the model recovered most of the underlying signal.

---

## Residual Diagnostics

The notebook contains three diagnostic visualizations:

1. **Actual vs. Predicted Values** — checks whether predictions follow the ideal diagonal line.
2. **Residuals vs. Predicted Values** — checks for curves, trends, or changing error spread.
3. **Residual Distribution** — checks whether the errors are centered around zero.

The resulting residuals were distributed around zero without a strong systematic pattern, which is consistent with the linear relationship used to generate the data.

---

## Learning Outcomes

By the end of this notebook, the reader should be able to:

- Identify `X` as the feature matrix and `y` as the target vector.
- Explain the role of feature weights and the intercept.
- Connect a linear equation with dot products and matrix multiplication.
- Distinguish between `fit()` and `predict()`.
- Explain why training and test data must be separated.
- Interpret MSE, RMSE, and R² at a practical level.
- Compare a trained model with a simple baseline.
- Recognize the basic signs of overfitting.
- Use residual plots to detect problems hidden by summary metrics.

---

## Project Structure

```text
week-3/
└── d1/
    ├── README.md
    └── 01_linear_regression.ipynb
```

## Requirements

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook or JupyterLab

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Run

From the repository root:

```bash
jupyter notebook
```

Then:

1. Open `week-3/d1/01_linear_regression.ipynb`.
2. Restart the kernel.
3. Run all cells from top to bottom.
4. Confirm that the metric values and plots are generated without errors.

No external dataset is required for this notebook.

---

## Key Takeaways

- Linear Regression learns weights and an intercept from labeled numeric data.
- `fit()` performs learning; `predict()` uses what was learned.
- Test data provides a more honest estimate of performance on unseen observations.
- Metrics should be interpreted alongside a baseline and diagnostic plots.
- Synthetic data is useful here because the hidden relationship is known and can be verified.

## Next Step

Continue with **Logistic Regression and binary classification** using the Titanic dataset, where the target is `Survived`.

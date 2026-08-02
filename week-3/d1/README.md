# Week 3 — Day 1: Linear Regression Fundamentals

This notebook introduces supervised machine learning through a complete Linear Regression workflow using Scikit-learn.

A controlled synthetic dataset is used so that the true mathematical relationship is known in advance. This makes it possible to verify whether the trained model successfully recovers the underlying signal, ignores irrelevant features, and generalizes to unseen data.

**Notebook:** [Open `01_linear_regression.ipynb`](./01_linear_regression.ipynb)

## Learning Objectives

By completing this notebook, I was able to:

- Distinguish supervised learning, regression, and classification.
- Explain features, targets, weights, and the intercept.
- Calculate linear predictions manually and with vectorized NumPy operations.
- Understand how Ordinary Least Squares estimates model parameters.
- Split data into training and test sets without data leakage.
- Train a `LinearRegression` model using Scikit-learn.
- Compare learned parameters with known true parameters.
- Evaluate regression performance using MSE, RMSE, and $R^2$.
- Compare the trained model with a mean-prediction baseline.
- Inspect residual plots for systematic error patterns.

## Experiment Design

The target was generated using the following known relationship:

$$
y
=
4
+
3x_1
-
2x_2
+
1.5x_3
+
0x_4
+
0x_5
+
\varepsilon
$$

Where:

- $x_1$, $x_2$, and $x_3$ are informative features.
- $x_4$ and $x_5$ are irrelevant noise features with true weights of zero.
- The true intercept is $4$.
- $\varepsilon$ represents random noise.

The dataset contains 500 observations and five input features. A fixed random seed of `42` is used to make the experiment reproducible.

## Workflow

1. Review the structure of supervised learning problems.
2. Calculate a linear prediction manually.
3. Express predictions using dot products and matrix multiplication.
4. Generate a controlled synthetic regression dataset.
5. Split the data into 80% training and 20% testing sets.
6. Train an Ordinary Least Squares Linear Regression model.
7. Compare learned and true model parameters.
8. Generate predictions on unseen test data.
9. Evaluate the model and compare it with a baseline.
10. Inspect residual diagnostic plots.

## Results

### Learned Parameters

| Parameter | True Value | Learned Value |
|---|---:|---:|
| Intercept | 4.0000 | 4.0313 |
| $x_1$ | 3.0000 | 3.0933 |
| $x_2$ | -2.0000 | -1.9932 |
| $x_3$ | 1.5000 | 1.4705 |
| $x_4$ | 0.0000 | -0.0329 |
| $x_5$ | 0.0000 | 0.0398 |

The model recovered the informative feature weights accurately, while the irrelevant features remained close to zero.

### Model Performance

| Dataset / Model | RMSE | $R^2$ |
|---|---:|---:|
| Training set | 1.0250 | 0.9316 |
| Test set | 0.9378 | 0.9423 |
| Mean baseline | 3.9041 | -0.0001 |

Additional test metric:

- **Test MSE:** `0.8794`

The trained model substantially outperformed the mean baseline. Training and test performance were also similar, indicating no clear overfitting in this experiment.

## Residual Diagnostics

The notebook includes:

- Actual versus predicted values.
- Residuals versus predicted values.
- A histogram of residuals.

The predictions remain close to the ideal reference line, and the residuals are distributed around zero without a strong systematic pattern. This supports the suitability of a linear model for the generated data.

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

Install the required packages with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Run

1. Activate the project's virtual environment.
2. Launch Jupyter from the repository directory:

```bash
jupyter notebook
```

3. Open `week-3/d1/01_linear_regression.ipynb`.
4. Restart the kernel and run all cells from top to bottom.

The notebook should execute without requiring an external dataset.

## Key Takeaways

- `fit()` learns the model parameters from training data.
- `predict()` applies the learned relationship to new observations.
- A train/test split provides a fairer estimate of generalization performance.
- A baseline is necessary to determine whether a trained model adds real predictive value.
- Residual analysis reveals model behavior that a single metric may not show.
- Synthetic data is useful here because the true parameters are known and can be compared directly with the learned parameters.

## Next Step

Continue with **Logistic Regression and binary classification** using the Titanic dataset, where the target variable is `Survived`.

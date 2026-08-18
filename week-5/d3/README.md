# PCA and Dimensionality Reduction

Day 3 of Week 5 focuses on reducing high-dimensional data with Principal Component Analysis (PCA), measuring how much variance the components preserve, and choosing a reasonable number of components instead of reducing dimensions arbitrarily.

## Notebook

`03_pca_dimensionality_reduction.ipynb`

## What I Covered

The notebook follows this workflow:

1. inspect a dataset with 30 numeric features;
2. standardize the features with `StandardScaler`;
3. fit PCA across all available components;
4. inspect explained variance and cumulative explained variance;
5. find the smallest number of components that retains about 95% of the variance;
6. reduce the dataset using that component count;
7. inspect component loadings to see how original features contribute;
8. reduce the same data to two components for visualization;
9. summarize the main benefits and trade-offs of PCA.

## Dataset

The experiment uses Scikit-learn's Breast Cancer Wisconsin dataset.

It contains 569 samples and 30 numeric input features. PCA is fitted only on the feature matrix. The provided diagnosis target is kept out of the PCA fitting process and is used only to color the final 2D plot for visual interpretation.

## Why Scaling Comes First

PCA is variance-based, so features with larger numeric scales can dominate the components if the raw columns are used directly.

The workflow therefore starts with:

```text
Original features
      ↓
StandardScaler
      ↓
PCA
```

After scaling, every feature has mean 0 and standard deviation 1, so the PCA directions are not driven by the original measurement units.

## Choosing the Number of Components

PCA was first fitted without reducing the number of dimensions. I then calculated cumulative explained variance and selected the smallest number of components that reached the 95% threshold.

For this dataset:

- Original dimensions: `30`
- Components selected: `10`
- Variance retained: approximately `95.16%`

The 95% threshold is a practical choice rather than a fixed rule. It gives a clear balance between compression and information retention for this experiment.

## 2D Visualization

A separate PCA transformation with two components was used for visualization.

- PC1 explains approximately `44.27%` of the variance.
- PC2 explains approximately `18.97%`.
- Together they show approximately `63.24%` of the total variance.

This makes the 2D plot useful for inspecting structure, but it should not be treated as a complete representation of the original 30-dimensional data.

## Main Takeaway

PCA does not select a few original features. It creates new principal components from weighted combinations of the original features.

For this experiment, the main result is:

```text
30 standardized features
          ↓
PCA variance analysis
          ↓
10 components
          ↓
~95% of total variance retained
```

The main trade-off is that the representation becomes smaller and more compact, but the new components are less directly interpretable than the original feature names.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook or JupyterLab

Install the required packages with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Running the Notebook

From the repository environment:

```bash
jupyter notebook
```

Then open:

```text
week-5/d3/03_pca_dimensionality_reduction.ipynb
```

Run the notebook from top to bottom so the scaling, variance analysis, component selection, reduction, and visualization are produced in order.

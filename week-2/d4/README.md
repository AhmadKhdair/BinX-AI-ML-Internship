# Week 2 — Day 4: Univariate EDA, Distributions, and Outliers

This folder contains the Day 4 work for Week 2 of the BinX Tech AI and Machine Learning Internship.

The notebook performs the first part of Exploratory Data Analysis on the Palmer Penguins dataset. It focuses on univariate analysis, where each variable is examined independently before moving to relationships between variables in Day 5.

## Objectives

The main objectives of this notebook are to:

- inspect the dataset structure and variable types;
- identify missing values and exact duplicate rows;
- examine numeric distributions using histograms and KDE curves;
- inspect potential outliers using box plots;
- verify potential outliers numerically using the IQR method;
- examine categorical frequencies using count plots;
- identify uneven category representation;
- document findings before making cleaning or modelling decisions.

## Dataset

The analysis uses the Palmer Penguins dataset, which contains physical measurements and categorical information for penguins observed in the Palmer Archipelago, Antarctica.

The dataset contains:

- 344 observations;
- three penguin species: Adelie, Gentoo, and Chinstrap;
- three islands: Biscoe, Dream, and Torgersen;
- four physical measurement variables;
- sex and observation-year information.

The dataset is stored at:

```text
../data/penguins.csv
```

Dataset reference: [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)

## Notebook

The completed analysis is available in:

[04_univariate_eda_distributions_and_outliers.ipynb](./04_univariate_eda_distributions_and_outliers.ipynb)

The notebook is organised as a short analytical report, with Markdown explanations before the relevant code and written interpretations after important outputs.

## Analysis Workflow

The notebook follows this sequence:

```text
Load → Inspect → Understand → Analyse → Decide
```

It does not modify or remove observations immediately after loading the data. Missing values, duplicates, unusual distributions, and potential outliers are investigated before any decision is made.

## Topics Covered

### Data Structure

- Dataset shape and column names
- Data types
- Numeric and categorical variables
- Analytical treatment of the `year` column

### Data Quality

- Missing-value counts and percentages
- Inspection of incomplete rows
- Exact duplicate detection
- Documented decision to preserve the original DataFrame

### Numeric Distributions

The following measurements are analysed:

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

Histograms and KDE curves are used to examine:

- centre and spread;
- skewness;
- multiple peaks;
- gaps;
- potentially unusual observations.

### Outlier Detection

Potential outliers are investigated using:

- box plots;
- the 1.5 × IQR rule;
- calculated lower and upper bounds for every physical measurement.

### Categorical Distributions

Count plots and frequency tables are created for:

- `species`;
- `island`;
- `sex`;
- `year`.

Missing values in `sex` are displayed explicitly during visualisation without changing the original dataset.

## Key Findings

### Data Quality

- The dataset contains 11 rows with at least one missing value.
- The `sex` column contains 11 missing values, representing 3.20% of the dataset.
- Two observations are missing all four physical measurements.
- No exact duplicate rows were found.

### Numeric Distributions

- The physical measurements do not follow simple single-peaked distributions.
- Bill length and bill depth show more than one area of concentration.
- Flipper length shows the clearest separation between two concentration areas.
- Body mass contains noticeable lighter and heavier groups.
- These patterns may reflect natural subgroups, but univariate analysis alone cannot confirm their cause.

### Outliers

- The box plots show no observations beyond their whiskers.
- The numerical IQR method flags zero potential outliers in all four physical measurement columns.
- No observations are removed, capped, or modified for outlier treatment.

### Categorical Frequencies

- Adelie is the most represented species with 152 observations (44.19%).
- Gentoo has 124 observations (36.05%).
- Chinstrap has the fewest observations with 68 (19.77%).
- Biscoe contains the largest number of observations, while Torgersen contains the fewest.
- Recorded male and female counts are nearly balanced.
- Observations are distributed fairly evenly across 2007, 2008, and 2009.

## Analytical Decisions

The original DataFrame remains unchanged during this notebook.

Specifically:

- missing values are not filled or removed;
- no rows are removed as duplicates;
- no values are removed or capped as outliers;
- categories are not removed or resampled;
- relationships between variables are left for the Day 5 bivariate analysis.

Future preprocessing decisions will depend on the specific machine-learning objective and the variables used by the model.

## Tools and Libraries

- Python
- Pandas
- Matplotlib
- Seaborn
- Pathlib
- Jupyter Notebook

## How to Run

1. Keep the notebook inside `week-2/d4/`.
2. Keep the dataset at `week-2/data/penguins.csv`.
3. Activate the project’s Python virtual environment.
4. Launch Jupyter Notebook from the repository.
5. Open `04_univariate_eda_distributions_and_outliers.ipynb`.
6. Run all cells from top to bottom.

The notebook uses the following relative path:

```python
data_path = Path("../data/penguins.csv")
```

This allows it to run correctly as long as the repository structure remains unchanged.

## Scope

This notebook covers Day 4 univariate EDA only.

The following topics are intentionally reserved for Day 5:

- bivariate analysis;
- scatter plots;
- grouped box plots;
- correlation matrices;
- heatmaps;
- pairplots;
- data storytelling across variable relationships.

## Status

Day 4 completed.
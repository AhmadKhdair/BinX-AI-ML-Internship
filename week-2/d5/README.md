# Week 2 — Day 5: Complete Exploratory Data Analysis

This directory contains the final deliverable for **Week 2, Day 5** of the BinX Tech AI & Machine Learning Internship.

The notebook presents a complete exploratory data analysis of the Palmer Penguins dataset. It combines the main results from descriptive statistics and univariate analysis with the Day 5 requirements: bivariate analysis, grouped comparisons, correlation analysis, pairwise visualization, and data storytelling.

## Project Objective

The goal of this analysis is to understand the dataset before any machine-learning model is trained.

The notebook focuses on the following questions:

- How are the penguins' physical measurements distributed?
- Which measurements are strongly related?
- How do physical measurements differ across species and sex groups?
- Which features may be useful for predicting penguin species in a future classification task?
- What data-quality issues should be considered before modeling?

## Dataset

The dataset contains **344 penguin observations** and the following variables:

| Column | Description |
|---|---|
| `species` | Penguin species: Adelie, Chinstrap, or Gentoo |
| `island` | Island where the observation was recorded |
| `bill_length_mm` | Bill length in millimetres |
| `bill_depth_mm` | Bill depth in millimetres |
| `flipper_length_mm` | Flipper length in millimetres |
| `body_mass_g` | Body mass in grams |
| `sex` | Recorded sex |
| `year` | Collection year |

The analysis keeps the original dataset unchanged and uses temporary subsets only when a specific calculation or plot requires complete values.

## Analysis Workflow

The notebook follows this sequence:

1. Load and inspect the dataset.
2. Check data types, missing values, and duplicate rows.
3. Review descriptive statistics and univariate distributions.
4. Investigate potential outliers using the IQR method.
5. Examine numeric relationships with scatter plots.
6. Compare numeric measurements across categories using grouped box plots.
7. Calculate and visualize the correlation matrix.
8. Review all physical measurements together using a pairplot.
9. Summarize the main findings and their implications for future modeling.

## Key Findings

- The strongest numeric relationship is between `flipper_length_mm` and `body_mass_g`, with a correlation of **0.871**.
- Gentoo penguins generally have longer flippers, greater body mass, and shallower bills than the other species.
- Adelie penguins generally have shorter bills.
- Chinstrap penguins tend to have longer and deeper bills.
- Within each species, male penguins generally have a higher median body mass than female penguins.
- Bill length and bill depth have a weak overall correlation, but their combination reveals useful species-related clusters.
- No potential outliers were flagged in the four physical measurement columns when the IQR rule was applied to the complete numeric distributions.
- No single measurement separates all three species perfectly; combining several physical features is more informative.

## Scope

This notebook performs **exploratory data analysis only**. It does not train or evaluate a machine-learning model.

A natural next step would be to build a classification model that predicts `species` using the four physical measurements after applying appropriate preprocessing.

## Files

```text
week-2/
├── data/
│   └── penguins.csv
└── d5/
    ├── 05_complete_eda.ipynb
    └── README.md
```

- [Open the completed notebook](./05_complete_eda.ipynb)
- [View the dataset](../data/penguins.csv)

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AhmadKhdair/BinX-AI-ML-Internship.git
cd BinX-AI-ML-Internship
```

### 2. Create and activate a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required packages

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
week-2/d5/05_complete_eda.ipynb
```

Then run the notebook from top to bottom.

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Git and GitHub

## Reproducibility Notes

- The dataset is loaded using the relative path `../data/penguins.csv`.
- The original DataFrame is preserved, and a separate working copy is used for the analysis.
- Missing values are excluded only from the calculations or plots that require the affected columns.
- Outliers are investigated rather than removed automatically.
- All plots include titles and labelled axes, and the notebook includes written interpretations of the results.

## Dataset Attribution

The Palmer Penguins data were made available by Dr. Kristen Gorman and the Palmer Station Long Term Ecological Research Program.

Recommended citation:

> Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. DOI: 10.5281/zenodo.3960218.

The dataset is available under the **CC0** public-domain dedication.

## Internship Context

This work was completed as part of the **BinX Tech Artificial Intelligence & Machine Learning Internship — Week 2: Math Foundations and Exploratory Data Analysis**.

## Maintainer

**Ahmad Khdair**

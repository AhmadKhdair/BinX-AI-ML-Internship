# Week 1 - Python and Data Science Foundations

## Overview

Week 1 of the BinX Tech AI and Machine Learning Internship focused on building the Python and data science foundations needed for the rest of the training program.

The week covered environment setup, Python fundamentals, NumPy, Pandas, Matplotlib, Jupyter Notebook, Git, and GitHub.

Each day contains a separate notebook and README file. The week also includes an integrated mini-notebook that combines NumPy, Pandas, and Matplotlib in one complete data analysis workflow.

## Week 1 Objectives

The main objectives of this week were to:

- Set up a reproducible Python environment
- Work with Jupyter Notebook code and Markdown cells
- Review the main Python concepts used in data science
- Perform numerical operations using NumPy
- Load, inspect, clean, filter, and group data using Pandas
- Create clear visualizations using Matplotlib
- Document the analysis using Markdown
- Track and submit the work using Git and GitHub

## Week Structure

| Day | Topic | Main Notebook |
|---|---|---|
| Day 1 | Environment Setup and Jupyter Workflow | `d1/01_environment_setup.ipynb` |
| Day 2 | Python Fundamentals | `d2/02_python_fundamentals.ipynb` |
| Day 3 | NumPy Fundamentals | `d3/03_numpy_fundamentals.ipynb` |
| Day 4 | Pandas Fundamentals | `d4/04_pandas_fundamentals.ipynb` |
| Day 5 | Matplotlib Fundamentals | `d5/05_matplotlib_fundamentals.ipynb` |
| Final Task | Week 1 Integrated Mini-Notebook | `week_1_integrated_mini_notebook.ipynb` |

---

## Day 1 - Environment Setup and Jupyter Workflow

### Work Completed

- Installed and verified Python
- Created a project virtual environment
- Activated the virtual environment using PowerShell
- Installed the required Python libraries
- Started Jupyter Notebook
- Practiced using Code and Markdown cells
- Created `requirements.txt`
- Initialized Git and connected the project to GitHub
- Practiced the basic Git workflow

### Main Tools

- Python
- Virtual environments
- pip
- Jupyter Notebook
- PowerShell
- Git
- GitHub

---

## Day 2 - Python Fundamentals

### Topics Covered

- Variables and basic data types
- Strings, numbers, and booleans
- Lists, tuples, dictionaries, and sets
- Comparison and logical operators
- Conditions using `if`, `elif`, and `else`
- Loops using `for` and `while`
- Functions and return values
- List comprehensions
- Basic object-oriented programming
- Classes, objects, attributes, and methods

### Practice Completed

- Filtering values using loops
- Rewriting loops as list comprehensions
- Creating a function that returns the mean, minimum, and maximum
- Creating a small `StudentRecord` class
- Adding attributes and methods to a class

---

## Day 3 - NumPy Fundamentals

### Topics Covered

- Creating NumPy arrays
- Array shape, dimensions, size, and data type
- One-dimensional and two-dimensional arrays
- Indexing and slicing
- Selecting rows and columns
- Boolean masking
- Vectorized operations
- NumPy mathematical and statistical functions
- Broadcasting
- Random number generation
- Reshaping and flattening arrays

### Practice Completed

- Created arrays in different ways
- Selected values using indexes and slices
- Filtered arrays using conditions
- Applied operations to complete arrays without loops
- Used broadcasting with compatible array shapes
- Compared NumPy arrays with regular Python lists

---

## Day 4 - Pandas Fundamentals

### Dataset

The Titanic dataset was used to practice working with real tabular data.

### Topics Covered

- Pandas Series and DataFrames
- Loading CSV files
- Inspecting a dataset using:
  - `head()`
  - `shape`
  - `columns`
  - `info()`
  - `describe()`
- Selecting single and multiple columns
- Filtering rows using conditions
- Using `loc` and `iloc`
- Checking missing values
- Filling missing values
- Removing columns
- Checking and removing duplicate rows
- Inspecting data types
- Grouping and aggregation

### Cleaning Decisions

- Missing `Age` values were filled using the median
- Missing `Embarked` values were filled using the mode
- The `Cabin` column was removed because most of its values were missing
- Duplicate rows were checked and removed when necessary

---

## Day 5 - Matplotlib Fundamentals

### Topics Covered

- Figure and Axes objects
- Line plots
- Scatter plots
- Bar plots
- Histograms
- Subplots
- Titles and axis labels
- Legends and grids
- Figure size and layout
- Plot annotations
- The object-oriented Matplotlib approach
- Saving figures
- Common visualization mistakes

### Practice Data

A small store dataset was created to practice visualizing:

- Monthly sales
- Monthly expenses
- Customer counts
- Monthly profit
- Transaction value distributions

---

# Week 1 Integrated Mini-Notebook

## Project

The final task for Week 1 is:

`week_1_integrated_mini_notebook.ipynb`

The notebook combines the main tools covered during the week:

- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- Markdown documentation

## Dataset

The project uses the Titanic passenger dataset stored at:

`data/titanic.csv`

## Analysis Workflow

The project follows a complete basic data analysis workflow:

1. Importing the required libraries
2. Loading the Titanic dataset
3. Inspecting the dataset structure
4. Understanding the main columns
5. Checking missing values and duplicate rows
6. Inspecting data types and categorical values
7. Cleaning the dataset
8. Verifying the cleaning results
9. Creating new features with NumPy
10. Filtering the data
11. Grouping and aggregating the results
12. Creating visualizations
13. Writing observations and final findings

## Data Cleaning

The project includes the following cleaning steps:

- Filled missing `Age` values using the median age
- Filled missing `Embarked` values using the most common value
- Removed the `Cabin` column because approximately 77% of its values were missing
- Checked and removed duplicate rows
- Verified the dataset after cleaning

## NumPy Usage

NumPy was used to:

- Convert Pandas columns to NumPy arrays
- Create the `FamilySize` feature
- Create the `IsAlone` feature
- Calculate the mean ticket fare
- Calculate the median ticket fare
- Apply vectorized numerical operations

## Pandas Analysis

Pandas was used to:

- Load the CSV dataset
- Inspect rows, columns, and data types
- Check missing values
- Clean the dataset
- Filter passengers who paid more than the average fare
- Calculate survival rates using `groupby()`
- Compare ticket fares between passenger classes

## Visualizations

The integrated project contains the following labeled visualizations:

### Age Distribution Histogram

Shows how passenger ages are distributed across the dataset.

### Age and Ticket Fare Scatter Plot

Explores whether a visible relationship exists between passenger age and ticket fare.

### Survival Rate by Passenger Class

Compares the survival percentage of first, second, and third-class passengers.

### Survival Comparison Subplots

Compares:

- Survival rate by sex
- Survival rate for passengers travelling alone or with family

## Main Findings

The analysis found that:

- The dataset originally contained missing values in `Age`, `Cabin`, and `Embarked`
- No duplicate rows were found
- The median passenger age was 28
- The mean fare was higher than the median because of a small number of expensive tickets
- First-class passengers had the highest survival rate
- Third-class passengers had the lowest survival rate
- Female passengers had a much higher survival rate than male passengers
- Passengers travelling with family had a higher survival rate than passengers travelling alone
- Passenger age did not show a clear visual relationship with ticket fare

---

## Project Structure

```text
week-1/
│
├── README.md
├── requirements.txt
├── week_1_integrated_mini_notebook.ipynb
│
├── data/
│   └── titanic.csv
│
├── d1/
│   ├── README.md
│   └── 01_environment_setup.ipynb
│
├── d2/
│   ├── README.md
│   └── 02_python_fundamentals.ipynb
│
├── d3/
│   ├── README.md
│   └── 03_numpy_fundamentals.ipynb
│
├── d4/
│   ├── README.md
│   └── 04_pandas_fundamentals.ipynb
│
└── d5/
    ├── README.md
    └── 05_matplotlib_fundamentals.ipynb
```

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

The exact package versions used in the environment are stored in:

`requirements.txt`

## Running the Project

Open PowerShell and move to the Week 1 directory:

```powershell
cd C:\Users\HP\Desktop\BinX-AI-ML-Internship\week-1
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages when setting up the project on a new environment:

```powershell
pip install -r requirements.txt
```

Start Jupyter Notebook:

```powershell
jupyter notebook
```

Open any notebook and run its cells from top to bottom.

## Week 1 Completion

By the end of Week 1, I completed:

- A reproducible Python environment
- A Python fundamentals notebook
- A NumPy fundamentals notebook
- A Pandas fundamentals notebook
- A Matplotlib fundamentals notebook
- A complete integrated data analysis notebook
- Markdown documentation for each day
- A documented GitHub project structure

Week 1 is complete and ready for review.
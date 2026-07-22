# Day 4 — Pandas Fundamentals

This folder contains my Day 4 work for the BinX Tech AI and Machine Learning Internship.

The notebook introduces the main Pandas concepts and applies them to the Titanic passenger dataset. The work includes creating Series and DataFrames, loading and inspecting a CSV file, selecting and filtering data, calculating summary statistics, grouping records, and cleaning missing values.

## Files

- `04_pandas_fundamentals.ipynb` — the Day 4 Jupyter Notebook
- `train.csv` — the Titanic dataset used in the notebook
- `README.md` — a summary of the completed work

## Topics Covered

### Pandas Series

- Creating a Series from lists and dictionaries
- Checking Series data types
- Using custom index labels
- Selecting values with `.loc` and `.iloc`
- Updating Series values
- Filtering values using conditions

### Pandas DataFrames

- Creating a DataFrame from a dictionary
- Understanding rows, columns, and indexes
- Adding custom row labels
- Selecting rows with `.loc` and `.iloc`
- Adding new columns
- Adding new rows with `pd.concat()`

### Loading and Inspecting Data

The Titanic dataset was loaded using:

```python
titanic_df = pd.read_csv("train.csv")
```

The dataset contains 891 rows and 12 columns.

I inspected the dataset using:

- `head()`
- `tail()`
- `sample()`
- `shape`
- `columns`
- `dtypes`
- `info()`
- `describe()`

### Selecting Data

I selected individual and multiple columns from the dataset.

I also used:

- `.loc` to select rows using index labels
- `.iloc` to select rows and columns using integer positions
- `PassengerId` as a custom index

### Filtering Data

I filtered the dataset using single and multiple conditions.

Examples included:

- Passengers aged 18 or older
- Female passengers who survived

The filtering result showed that 233 female passengers survived.

### Aggregation

I used aggregation functions to calculate:

- Average passenger age
- Average ticket fare
- Minimum ticket fare
- Maximum ticket fare
- Number of known age values

### Grouping

I grouped passengers by ticket class using `groupby()`.

For every passenger class, I calculated:

- Average survival rate
- Average ticket fare
- Number of passengers

The results showed that first-class passengers had the highest survival rate and average fare. Third-class passengers had the lowest survival rate and formed the largest passenger group.

### Data Cleaning

I checked the number of missing values in every column.

The original dataset contained:

- 177 missing values in `Age`
- 687 missing values in `Cabin`
- 2 missing values in `Embarked`

The missing values were handled by:

- Filling missing `Age` values with the median age
- Filling missing `Embarked` values with the most common port
- Removing the `Cabin` column because most of its values were missing

I also checked for fully duplicated rows. No duplicated rows were found.

Finally, I checked the column data types after cleaning. The existing data types were suitable, so no additional type conversion was required.

## Main Tools Used

- Python
- Pandas
- Jupyter Notebook
- CSV data
- Git and GitHub

## How to Run the Notebook

From the `week-1` directory, activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start Jupyter Notebook:

```powershell
jupyter notebook
```

Open:

```text
d4/04_pandas_fundamentals.ipynb
```

Make sure `train.csv` is stored in the same folder as the notebook.

## Day 4 Outcome

By completing this notebook, I practiced the main Pandas workflow:

```text
Create → Load → Inspect → Select → Filter → Aggregate → Clean
```
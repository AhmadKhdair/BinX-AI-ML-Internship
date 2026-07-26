# Week 2 — Day 1: Descriptive Statistics

## 📌 Overview

This notebook introduces the foundations of **descriptive statistics** and applies them to a real-world dataset.

The notebook begins with small numerical examples to explain each statistical concept clearly. It then applies the same concepts to the **Palmer Penguins dataset**, focusing mainly on the `body_mass_g` column.

The goal is not only to calculate statistical values, but also to:

- Understand what each measure represents.
- Compare different statistical measures.
- Study how extreme values affect the results.
- Make justified analytical decisions.
- Interpret results using clear, plain language.
- Connect descriptive statistics to future EDA and machine-learning tasks.

---

## 🎯 Learning Objectives

By completing this notebook, I was able to:

- Calculate and interpret the **mean, median, and mode**.
- Explain when each measure of central tendency is useful.
- Understand how extreme values affect the mean and median differently.
- Calculate and interpret the **range, variance, and standard deviation**.
- Explain why measures of centre and spread should be considered together.
- Understand **percentiles, quartiles, and the interquartile range (IQR)**.
- Compare the range and IQR.
- Apply descriptive statistics to a real numeric variable.
- Inspect the structure and quality of a real dataset.
- Handle missing values using a documented analytical decision.
- Choose between the mean and median as a typical value.
- Summarise statistical findings in clear, plain language.
- Connect descriptive statistics to machine-learning preparation.

---

## 📂 Dataset

This notebook uses the **Palmer Penguins dataset**.

The dataset contains observations for **344 penguins** from three different species:

- Adelie
- Chinstrap
- Gentoo

It contains both categorical and numeric variables.

### Dataset Variables

| Column | Description |
|---|---|
| `species` | Penguin species |
| `island` | Island where the penguin was observed |
| `bill_length_mm` | Bill length in millimetres |
| `bill_depth_mm` | Bill depth in millimetres |
| `flipper_length_mm` | Flipper length in millimetres |
| `body_mass_g` | Body mass in grams |
| `sex` | Recorded sex |
| `year` | Year of observation |

The main variable analysed in this notebook is:

```text
body_mass_g
```

Body mass was selected because it is a clear numeric measurement that supports all the descriptive-statistics concepts covered during Day 1.

---

## 🧭 Notebook Workflow

The notebook follows a gradual learning and analysis process:

1. Import the required Python libraries.
2. Explain why descriptive statistics are important before modelling.
3. Introduce measures of central tendency.
4. Calculate the mean manually and using NumPy.
5. Calculate the median manually and using NumPy.
6. Understand the role of the mode.
7. Compare the mean, median, and mode.
8. Examine how an extreme value affects the mean and median.
9. Explain why the centre alone is not enough to describe data.
10. Introduce measures of spread.
11. Calculate the range.
12. Calculate variance step by step.
13. Calculate and interpret standard deviation.
14. Compare datasets with the same centre but different spread.
15. Introduce percentiles and quartiles.
16. Calculate Q1, Q2, Q3, and the IQR.
17. Compare the range and IQR.
18. Load the Palmer Penguins dataset.
19. Inspect the dataset structure and variables.
20. Check missing values.
21. Prepare the `body_mass_g` variable.
22. Calculate measures of central tendency.
23. Calculate measures of spread.
24. Build a complete statistical summary.
25. Visualise the distribution using a histogram.
26. Choose the most appropriate typical value.
27. Write a plain-language summary.
28. Connect the findings to future machine-learning work.

---

## 📊 Statistical Concepts Covered

### 1. Measures of Central Tendency

Measures of central tendency describe the centre or typical value of a dataset.

#### Mean

The mean is the arithmetic average of all observations.

```text
Mean = Sum of all values / Number of values
```

The mean uses every observation, making it useful for reasonably balanced data. However, it is sensitive to unusually high or low values.

#### Median

The median is the middle value after sorting the observations.

For an odd number of observations, it is the value in the middle.

For an even number of observations, it is the average of the two middle values.

The median is more resistant to extreme values because it depends mainly on the positions of the observations rather than their exact sizes.

#### Mode

The mode is the most frequently occurring value.

A dataset may have:

- One mode.
- More than one mode.
- No clear mode.

The mode can be used with both numeric and categorical variables.

---

### 2. Measures of Spread

Measures of central tendency are not enough to describe a dataset. Two datasets may have the same mean and median but very different levels of variation.

Measures of spread describe how far apart the observations are.

#### Range

The range is the difference between the maximum and minimum values.

```text
Range = Maximum - Minimum
```

It is easy to calculate and understand, but it is highly sensitive to extreme values because it depends only on the two endpoints.

#### Variance

Variance measures the average squared distance of the observations from the mean.

The general process is:

1. Calculate the mean.
2. Find the distance between every value and the mean.
3. Square each distance.
4. Calculate the average of the squared distances.

Squaring prevents negative and positive distances from cancelling each other out.

#### Standard Deviation

Standard deviation is the square root of variance.

It is easier to interpret than variance because it is expressed in the same unit as the original variable.

For example, body mass is measured in grams, so its standard deviation is also measured in grams.

---

### 3. Percentiles and Quartiles

Percentiles describe positions within ordered data.

A percentile tells us approximately what percentage of the observations lies at or below a particular point.

Quartiles are special percentiles that divide ordered observations into four sections:

- **Q1:** the 25th percentile.
- **Q2:** the 50th percentile and the median.
- **Q3:** the 75th percentile.

Quartiles may fall between two recorded values because statistical libraries may use interpolation.

---

### 4. Interquartile Range

The interquartile range measures the spread of the middle 50% of the observations.

```text
IQR = Q3 - Q1
```

The IQR is less sensitive to unusually high or low observations than the full range.

The range describes the complete distance from the minimum to the maximum, while the IQR describes the central spread of the data.

---

## ⚠️ Effect of Extreme Values

The notebook demonstrates that extreme values affect statistical measures differently.

| Measure | Effect of an Extreme Value |
|---|---|
| Mean | Can change substantially because every value is included |
| Median | Usually changes less because it depends mainly on position |
| Mode | Usually remains unchanged unless the extreme value is repeated |
| Range | Can change strongly because it uses the minimum and maximum |
| Variance | Usually increases because large distances are squared |
| Standard deviation | Usually increases as the overall spread increases |
| IQR | Is relatively resistant because it focuses on the middle 50% |

No statistical measure is automatically the best one.

The appropriate choice depends on:

- The shape of the distribution.
- The presence of extreme values.
- The type of variable.
- The purpose of the analysis.
- The real-world context.

---

## 🔍 Dataset Inspection

Before analysing body mass, the notebook checks:

- The number of rows and columns.
- Column names.
- Data types.
- Non-null value counts.
- Missing values.

The dataset contains:

```text
344 rows
8 columns
```

The dataset includes categorical variables such as:

```text
species
island
sex
```

It also includes numeric variables such as:

```text
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g
year
```

Although `year` is numeric, it represents the observation year rather than a physical measurement.

---

## 🧹 Missing-Value Decision

The `body_mass_g` column contains **two missing values**.

For this analysis, the missing values were excluded using:

```python
body_mass = df["body_mass_g"].dropna().copy()
```

The missing values were not filled using the mean or median because inserting estimated values could change:

- The mean.
- The variance.
- The standard deviation.
- The mode.
- The distribution of the variable.

The original DataFrame remains unchanged.

After removing the missing values from the analysis Series, the final analysis uses:

```text
342 recorded body masses
```

---

## 📈 Main Statistical Results

| Measure | Result |
|---|---:|
| Count | 342 observations |
| Mean | 4,201.75 g |
| Median | 4,050.00 g |
| Mode | 3,800.00 g |
| Minimum | 2,700.00 g |
| Maximum | 6,300.00 g |
| Range | 3,600.00 g |
| Sample variance | 643,131.08 g² |
| Sample standard deviation | 801.95 g |
| Q1 | 3,550.00 g |
| Q2 | 4,050.00 g |
| Q3 | 4,750.00 g |
| IQR | 1,200.00 g |

---

## 🧠 Interpretation of the Results

### Centre of the Distribution

The mean body mass is approximately:

```text
4,201.75 g
```

The median body mass is:

```text
4,050 g
```

The most frequently recorded body mass is:

```text
3,800 g
```

The mean is approximately **151.75 g higher than the median**.

This suggests that the upper end of the body-mass distribution has more influence on the arithmetic average.

However, the difference between the mean and median alone is not enough to classify any observation as an outlier.

---

### Full Spread

The lightest recorded penguin has a body mass of:

```text
2,700 g
```

The heaviest recorded penguin has a body mass of:

```text
6,300 g
```

The full range is therefore:

```text
3,600 g
```

This shows that the dataset contains penguins with substantially different body sizes.

Because the range depends only on the minimum and maximum, it does not show how most observations are distributed between those values.

---

### Standard Deviation

The sample standard deviation is approximately:

```text
801.95 g
```

This indicates considerable variation in the recorded body masses.

Standard deviation is easier to communicate than variance because it is measured in grams, the same unit as body mass.

It does not mean that every observation is exactly `801.95 g` away from the mean. It provides a general measure of how spread out the observations are around the mean.

---

### Quartiles and IQR

The quartiles are:

```text
Q1 = 3,550 g
Q2 = 4,050 g
Q3 = 4,750 g
```

This means that the middle 50% of the recorded penguins have body masses between:

```text
3,550 g and 4,750 g
```

The IQR is:

```text
1,200 g
```

The IQR represents the width of the central 50% of the distribution and is less influenced by the lightest and heaviest observations than the full range.

---

## ✅ Choosing the Better Typical Value

For one simple description of a typical penguin across the complete dataset, the **median is slightly more appropriate**.

The mean remains useful because it represents the arithmetic average of all recorded body masses.

However, the mean is influenced more by heavier observations, while the median is more resistant to the upper end of the distribution.

Therefore:

```text
Typical recorded body mass ≈ 4,050 g
```

This does not mean that the mean is incorrect. It means that the median gives a slightly more robust description of a typical body mass for this combined dataset.

---

## 📉 Visualisation

The notebook includes a histogram of recorded penguin body masses.

The visualisation contains:

- A descriptive title.
- Body mass on the x-axis.
- Number of penguins on the y-axis.
- A vertical reference line for the mean.
- A vertical reference line for the median.
- A legend describing both lines.

The mean appears slightly to the right of the median, which supports the numerical comparison.

The histogram is used as a simple visual check rather than a complete exploratory-data-analysis process.

More detailed distribution, grouped, and outlier analyses will be covered later during Week 2.

---

## 📝 Plain-Language Summary

The analysis used **342 recorded body masses**. Two missing observations were excluded without modifying the original dataset.

A typical penguin in the combined dataset weighs approximately **4,050 g** when the median is used.

The recorded body masses range from **2,700 g** to **6,300 g**.

The middle half of the recorded penguins weighs between **3,550 g** and **4,750 g**, giving an IQR of **1,200 g**.

The standard deviation is approximately **802 g**, which shows that body mass varies considerably across the recorded penguins.

Because the dataset contains several penguin species, the overall statistics may hide important differences between groups.

---

## 🤖 Connection to Machine Learning

Descriptive statistics help us understand a feature before using it in a machine-learning model.

This analysis highlights several considerations that may become important later:

- Missing values require a documented handling strategy.
- Body mass uses a different scale from measurements recorded in millimetres.
- Some machine-learning algorithms may require feature scaling.
- The difference between the mean and median suggests that the distribution should be inspected rather than assumed to be perfectly symmetric.
- Penguin species may explain part of the variation in body mass.
- Grouped analysis may reveal patterns hidden by overall statistics.
- Statistical summaries can help identify data-quality issues before modelling.

Descriptive statistics summarise the data, but they do not:

- Explain relationships between different variables.
- Prove why patterns exist.
- Establish causation.
- Replace a complete EDA process.

---

## 🛠️ Tools and Libraries

The following tools and libraries were used:

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- `pathlib`
- Git
- GitHub

---

## 📁 Project Structure

```text
week-2/
│
├── data/
│   └── penguins.csv
│
└── d1/
    ├── README.md
    └── 01_descriptive_statistics.ipynb
```

The notebook loads the dataset using the relative path:

```python
Path("../data/penguins.csv")
```

The folder structure should remain unchanged so that the notebook can locate the dataset correctly.

---

## ▶️ How to Run the Notebook

Open Windows PowerShell and navigate to the repository:

```powershell
cd C:\Users\HP\Desktop\BinX-AI-ML-Internship
```

Activate the existing virtual environment:

```powershell
.\week-1\.venv\Scripts\Activate.ps1
```

Move to the Week 2 directory:

```powershell
cd week-2
```

Start Jupyter Notebook:

```powershell
jupyter notebook
```

Then open:

```text
d1/01_descriptive_statistics.ipynb
```

Before reviewing or submitting the notebook, run all cells from top to bottom:

```text
Kernel → Restart Kernel and Run All Cells
```

Confirm that:

- The dataset loads successfully.
- No errors appear.
- All outputs are visible.
- The statistical-summary table is displayed.
- The histogram appears correctly.

---

## 📄 Files

| File | Description |
|---|---|
| `01_descriptive_statistics.ipynb` | Complete Day 1 learning and analysis notebook |
| `README.md` | Documentation for the Day 1 work |
| `../data/penguins.csv` | Dataset used in the analysis |

---

## ⚠️ Limitations

- The final real-data analysis focuses mainly on one numeric variable: `body_mass_g`.
- Two missing body-mass values are excluded rather than estimated.
- The dataset contains several penguin species, but the species are not analysed separately in this notebook.
- Formal outlier detection and treatment are not performed during Day 1.
- Relationships between multiple variables are not examined yet.
- The overall statistics may hide important differences between species.

These topics will be explored later during the EDA sections of Week 2.

---

## 🏁 Conclusion

This notebook covers the main foundations of descriptive statistics and applies them to a real numeric variable.

It demonstrates how to calculate and interpret:

- Mean, median, and mode.
- Minimum, maximum, and range.
- Variance and standard deviation.
- Percentiles and quartiles.
- Interquartile range.
- The effect of extreme values.

The notebook also:

- Inspects the structure of a real dataset.
- Documents the missing-value decision.
- Chooses an appropriate measure of a typical value.
- Presents the results using a statistical table and histogram.
- Explains the findings using plain language.
- Connects descriptive statistics to future machine-learning work.

---

## 🚀 Next Step

The next notebook will cover:

```text
Probability fundamentals and common probability distributions
```
# Day 5 - Matplotlib Fundamentals

## Objective

The objective of Day 5 was to learn how to create clear and useful data visualizations using Matplotlib.

The notebook focuses on the main plot types used in basic data analysis and explains how to choose the correct plot based on the type of information being explored.

## Work Completed

### Matplotlib Setup

- Imported NumPy and Matplotlib
- Used a fixed random seed for reproducible results
- Created a small store dataset for visualization practice
- Created variables for months, sales, expenses, customers, and profit

### Figure and Axes

- Learned the difference between a Figure and an Axes object
- Used `figsize` to control the size of a figure
- Added plot titles and axis labels
- Used legends and grid lines
- Used `tight_layout()` to improve spacing
- Used `show()` to display figures

### Line Plots

- Created a basic line plot
- Added markers and different line styles
- Changed line width
- Compared sales and expenses in the same plot
- Added a legend to identify multiple lines

### Scatter Plots

- Used scatter plots to study relationships between numeric variables
- Explored the relationship between customer count and store sales
- Changed point size using `s`
- Added transparency using `alpha`
- Added month labels using `annotate()`

### Bar Plots

- Used a bar plot to compare monthly profit
- Learned when to use vertical and horizontal bar plots
- Practiced comparing values across categories

### Histograms

- Created simulated transaction values using NumPy
- Used a histogram to study the distribution of numeric data
- Practiced changing the number of bins
- Added borders around histogram bars
- Learned the difference between a bar plot and a histogram

### Subplots

- Created a figure containing multiple plots
- Used a `2 × 2` subplot layout
- Added a line plot, scatter plot, bar plot, and histogram to one figure
- Accessed each plot using the `axes` array
- Used `axes.flat` to apply settings to all plots
- Added a main figure title using `fig.suptitle()`

### Matplotlib Approaches

- Practiced the Pyplot approach using functions such as `plt.plot()`
- Practiced the object-oriented approach using `fig` and `ax`
- Used the object-oriented approach for better control over figures

### Saving Figures

- Used `savefig()` to save a figure as an image
- Used `dpi` to control image quality
- Used `bbox_inches="tight"` to remove unnecessary empty space

### Visualization Practice

- Added a guide explaining when to use each plot type
- Documented common visualization mistakes
- Wrote observations based on the practice plots
- Added a conclusion summarizing the main Matplotlib concepts

## Plot Types Covered

| Plot Type | Main Use |
|---|---|
| Line Plot | Showing change or trends across an ordered variable |
| Scatter Plot | Exploring the relationship between two numeric variables |
| Bar Plot | Comparing values across categories |
| Histogram | Studying the distribution of one numeric variable |
| Subplots | Displaying multiple related plots in one figure |

## File

- `05_matplotlib_fundamentals.ipynb`

## What I Learned

By the end of Day 5, I was able to:

- Create and customize the main Matplotlib plot types
- Select an appropriate plot based on the analysis question
- Add titles, labels, legends, and grids
- Compare multiple variables in one figure
- Organize several plots using subplots
- Use the object-oriented Matplotlib approach
- Save figures as image files
- Write short observations that explain what a visualization shows

The final Week 1 integrated project is stored separately in:

`week-1/week_1_integrated_mini_notebook.ipynb`
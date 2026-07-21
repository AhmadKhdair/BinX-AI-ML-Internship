# Day 3 - NumPy Fundamentals

This folder contains my work for Day 3 of Week 1. The main goal was to understand how NumPy arrays work and how they can be used for numerical operations without writing a loop for every value.

## Files

- `03_numpy_fundamentals.ipynb` - the main notebook containing the explanations, examples, practice tasks, and the final Day 3 exercise.

## Topics I Practiced

### NumPy Arrays

I started by comparing normal Python lists with NumPy arrays. Although they may look similar, NumPy arrays support mathematical operations directly on all their values.

I also created arrays using:

- `np.array()`
- `np.zeros()`
- `np.ones()`
- `np.arange()`
- `np.linspace()`
- `np.random.rand()`

I used a fixed random seed so the random results can be reproduced when the notebook is run again.

### Array Dimensions and Attributes

I worked with 1D, 2D, and 3D arrays and used the following attributes:

- `ndim`
- `shape`
- `size`
- `dtype`

I also used `reshape()` to arrange the same values into different dimensions.

### Indexing and Slicing

I practiced accessing individual values using indices and extracting sections from arrays using slicing.

This included:

- Accessing the first and last values
- Selecting rows and columns
- Using negative indices
- Selecting ranges of rows and columns
- Reversing rows or columns
- Extracting smaller sections from a matrix

### Arithmetic and Vectorized Operations

I applied scalar arithmetic directly to complete arrays and performed elementwise operations between arrays.

I also used NumPy functions such as:

- `np.sqrt()`
- `np.round()`
- `np.floor()`
- `np.ceil()`

### Broadcasting

Broadcasting was one of the parts that needed more attention at first.

I learned that NumPy can perform operations between arrays with different but compatible shapes. A dimension with size `1` can be virtually expanded during the operation.

I practiced adding a one-dimensional row to every row in a two-dimensional array.

### Aggregate Functions

I used aggregate functions to summarize array values:

- `sum()`
- `mean()`
- `min()`
- `max()`
- `std()`
- `var()`
- `argmin()`
- `argmax()`

I also practiced using `axis=0` for column results and `axis=1` for row results.

### Boolean Filtering

I created Boolean conditions and used them to select matching values.

I practiced:

- Filtering using comparison operators
- Combining conditions with `&` and `|`
- Finding even and odd values
- Selecting values above the mean
- Updating values using a condition
- Using `np.where()` while keeping the original array shape

## Final Exercise

At the end of the notebook, I completed the official Day 3 exercise by:

1. Creating a 4 × 4 array containing the values from 1 to 16.
2. Printing its shape and data type.
3. Extracting the second column and the last row.
4. Selecting values greater than the array mean.
5. Adding a one-dimensional row to every row using broadcasting.
6. Checking the broadcasting result manually.

## How to Run the Notebook

From the `week-1` folder, activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start Jupyter Notebook:

```powershell
jupyter notebook
```

Then open:

```text
d3/03_numpy_fundamentals.ipynb
```

## Notes

The sections that required the most practice were broadcasting, the `axis` parameter, and combining multiple Boolean conditions.

Writing the shapes and Boolean masks helped me understand what NumPy was doing instead of only looking at the final output.
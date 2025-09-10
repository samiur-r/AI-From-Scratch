# NumPy Quick Reference

NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides a powerful N-dimensional array object, broadcasting functions, linear algebra operations, and much more. NumPy arrays are faster and more memory-efficient than Python lists, making them essential for data science, machine learning, and scientific computing.

### Installation
```bash
pip install numpy
```

### Importing NumPy

```
import numpy as np
```

* * * * *

2\. Creating Arrays
-------------------

```
import numpy as np

# From a Python list
arr1 = np.array([1, 2, 3, 4])
print(arr1)  # [1, 2, 3, 4]

# Zeros and ones
arr2 = np.zeros((2, 3))  # [[0. 0. 0.] [0. 0. 0.]]
arr3 = np.ones((3, 2))   # [[1. 1.] [1. 1.] [1. 1.]]

# Range of numbers
arr4 = np.arange(0, 10, 2)  # [0 2 4 6 8] (start, stop, step)

# Evenly spaced numbers
arr5 = np.linspace(0, 1, 5)  # [0. , 0.25, 0.5 , 0.75, 1.]

```

* * * * *

3\. Array Attributes
--------------------

```
a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape)     # (2, 3) - dimensions
print(a.ndim)      # 2 - number of dimensions
print(a.size)      # 6 - total number of elements
print(a.dtype)     # int64 - data type
print(a.itemsize)  # 8 - bytes per element
print(a.nbytes)    # 48 - total bytes used

# Memory layout
print(a.flags.c_contiguous)  # True - C-style contiguous
print(a.strides)   # (24, 8) - bytes to next element in each dimension

```

* * * * *

4\. Indexing and Slicing
------------------------

```
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])      # First element
print(arr[-1])     # Last element
print(arr[1:4])    # Slice [20, 30, 40]

# 2D array
mat = np.array([[1, 2, 3], [4, 5, 6]])

print(mat[0, 1])   # 2 (row 0, col 1)
print(mat[:, 0])   # First column [1, 4]

```

* * * * *

5\. Array Operations
--------------------

```
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])

print(x + y)     # [11 22 33]
print(x * y)     # [10 40 90]
print(x ** 2)    # [1 4 9]

# Aggregations
print(y.sum())    # 60
print(y.mean())   # 20.0

```

* * * * *

6\. Broadcasting
---------------

```
# Broadcasting allows operations between different shaped arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 array
scalar = 10

print(arr + scalar)  # [[11 12 13] [14 15 16]] (adds 10 to all elements)

# 1D array with 2D array
col_vec = np.array([[1], [2]])  # 2x1 array
print(arr + col_vec)  # [[2 3 4] [6 7 8]] (adds column-wise)

# Different sized arrays
x = np.array([1, 2, 3])  # 1x3
y = np.array([[10], [20]])  # 2x1
print(x + y)  # [[11 12 13] [21 22 23]]

```

* * * * *

7\. Linear Algebra
------------------

```
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)  # [[19 22] [43 50]] (Matrix multiplication)

# Transpose
print(A.T)  # [[1 3] [2 4]]

# Determinant
print(np.linalg.det(A))  # -2.0

# Inverse
print(np.linalg.inv(A))  # [[-2.   1. ] [ 1.5 -0.5]]

```

* * * * *

8\. Random Numbers
------------------

```
np.random.seed(42)  # Reproducibility

print(np.random.rand(3))      # 3 random numbers (0-1)
print(np.random.randint(1, 10, 5))  # 5 random ints between 1 and 9
print(np.random.randn(2, 2))  # Normal distribution

```

* * * * *

9\. Advanced Operations
-----------------------

```
arr = np.array([1, 2, 3, 2, 1, 4, 5])

# Unique values
print(np.unique(arr))  # [1 2 3 4 5]

# Boolean indexing/masking
print(arr > 2)          # [False False True False False True True]
print(arr[arr > 2])     # [3 4 5]
print(arr[(arr > 1) & (arr < 4)])  # [2 3 2] (multiple conditions)

# Where function
print(np.where(arr > 2, arr, 0))  # [0 0 3 0 0 4 5] (replace values)

# Sorting
print(np.sort(arr))     # [1 1 2 2 3 4 5]
print(np.argsort(arr))  # [0 4 1 3 2 5 6] (indices that would sort)

```

* * * * *

10\. Array Reshaping & Manipulation
----------------------------------

```
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshaping
print(arr.reshape(2, 3))    # [[1 2 3] [4 5 6]]
print(arr.reshape(3, -1))   # [[1 2] [3 4] [5 6]] (-1 infers dimension)

# Flattening
arr2d = np.array([[1, 2], [3, 4]])
print(arr2d.flatten())      # [1 2 3 4]
print(arr2d.ravel())        # [1 2 3 4] (returns view when possible)

# Transpose
print(arr2d.T)              # [[1 3] [2 4]]

# Adding/removing dimensions
print(arr[:, np.newaxis])   # Convert to column vector
print(np.squeeze(arr[:, np.newaxis]))  # Remove single dimensions

```

* * * * *

11\. Concatenation & Stacking
-----------------------------

```
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenation
print(np.concatenate([a, b]))  # [1 2 3 4 5 6]

# Stacking
print(np.vstack([a, b]))    # [[1 2 3] [4 5 6]] (vertical)
print(np.hstack([a, b]))    # [1 2 3 4 5 6] (horizontal)

# 2D arrays
c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])

print(np.vstack([c, d]))    # [[1 2] [3 4] [5 6] [7 8]]
print(np.hstack([c, d]))    # [[1 2 5 6] [3 4 7 8]]

```

* * * * *

12\. Handling Missing Data
-------------------------

```
data = np.array([1, 2, np.nan, 4])

print(np.isnan(data))          # [False False True False]
print(np.nan_to_num(data))     # Replace nan with 0

```

* * * * *

13\. Performance Tip
--------------------

Vectorization is much faster than loops!

```
arr = np.arange(1_000_000)

# Slow (Python loop)
result_loop = [x * 2 for x in arr]

# Fast (NumPy vectorized)
result_np = arr * 2

```

* * * * *

Summary
=========

-   NumPy arrays are **faster** and more **memory-efficient** than Python lists.

-   Learn to use **slicing, broadcasting, and vectorization** for best performance.

-   NumPy integrates with **Pandas, Matplotlib, and SciPy**.
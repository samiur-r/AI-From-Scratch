1) What is NumPy and why use it?
================================

**NumPy** = "Numerical Python." It provides:

-   **ndarray**: fast, contiguous, typed arrays

-   **Vectorized math**: write arithmetic on whole arrays (no Python loops)

-   **Broadcasting**: mix shapes without manual tiling

-   **Linear algebra / random** utilities

### Key ideas vs plain Python lists

-   Arrays are **homogeneous** (one dtype); lists can mix types.

-   Arrays store data in **contiguous memory** → faster operations.

-   Vectorized code is **shorter** and **much faster** than loops.

* * * * *

Install & import
----------------

```
pip install numpy

```

```
import numpy as np

```

* * * * *

Quick start: arrays vs lists
----------------------------

### Elementwise arithmetic

```
# Python lists
L = [1, 2, 3, 4]
# L * 2  -> [1, 2, 3, 4, 1, 2, 3, 4]  (list repetition)

# NumPy array
a = np.array([1, 2, 3, 4])
a * 2       # array([2, 4, 6, 8])
a + 10      # array([11, 12, 13, 14])
a ** 2      # array([ 1,  4,  9, 16])

```

### Types & attributes (meet the `ndarray`)

```
a = np.array([1, 2, 3], dtype=np.int32)
a.ndim      # 1 (dimensions)
a.shape     # (3,)
a.dtype     # dtype('int32')
a.size      # 3 (number of elements)
a.itemsize  # bytes per element, e.g. 4

```

### 2D arrays (matrices)

```
A = np.array([[1, 2, 3],
              [4, 5, 6]])
A.ndim      # 2
A.shape     # (2, 3)
A.T         # transpose -> array([[1, 4],[2, 5],[3, 6]])

```

* * * * *

Creating arrays (the "toolbox")
-------------------------------

### From Python sequences

```
np.array([1, 2, 3])                 # from list
np.array((1.0, 2.0, 3.0))           # from tuple
np.array([[1, 2], [3, 4]], dtype=float)

```

### From constructors (preferred in practice)

```
np.zeros((2, 3))                    # 2x3 of 0s
np.ones((3,))                       # length-3 of 1s
np.full((2, 2), fill_value=7)       # 2x2 filled with 7
np.arange(0, 10, 2)                 # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)                # [0., 0.25, 0.5, 0.75, 1.]

```

### Random arrays (intro)

```
rng = np.random.default_rng(seed=42) # modern Generator API
rng.random((2, 3))                   # uniform [0,1), shape (2,3)
rng.normal(loc=0, scale=1, size=5)   # 1D standard normal
rng.integers(0, 10, size=(2,2))      # ints in [0,10)

```

* * * * *

Indexing & slicing (just a taste)
---------------------------------

```
a = np.arange(10)        # [0 1 2 3 4 5 6 7 8 9]
a[0], a[-1]              # (0, 9)
a[2:7:2]                 # [2 4 6]

A = np.arange(12).reshape(3, 4)
# A =
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]
A[0, 2]                  # 2
A[1:, :2]                # rows 1..end, first two cols -> [[4,5],[8,9]]

```

* * * * *

Vectorized math (why NumPy is fast)
-----------------------------------

```
x = np.linspace(0, 2*np.pi, 5)
np.sin(x)                # elementwise sine
np.exp(x) + 3*np.cos(x)  # combine ufuncs without loops

# reductions (aggregations)
a = np.array([1, 2, 3, 4], dtype=float)
a.sum()                  # 10.0
a.mean()                 # 2.5
a.min(), a.max()         # (1.0, 4.0)

```

* * * * *

Broadcasting (intro intuition)
------------------------------

```
A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2,3)
b = np.array([10, 20, 30])     # shape (3,)
A + b                          # adds b to each row
# -> [[11, 22, 33],
#     [14, 25, 36]]

```

* * * * *

Comparing arrays & boolean masks
--------------------------------

```
a = np.array([3, 7, 2, 9, 5])
mask = a > 5           # array([False, True, False, True, False])
a[mask]                # array([7, 9])
np.where(a % 2 == 0, a, -1)  # keep evens, else -1 -> [ -1, -1,  2, -1, -1]

```

* * * * *

Common pitfalls (so you don't trip)
-----------------------------------

```
# 1) dtype inference: mixing ints & floats promotes to float
np.array([1, 2.0]).dtype       # float64

# 2) List-style repetition vs numeric multiply
[1,2,3]*2                      # [1,2,3,1,2,3]
(np.array([1,2,3])*2).tolist() # [2,4,6]

# 3) Shape matters: (3,1) vs (3,)
x = np.array([1,2,3])      # shape (3,)
x_col = x[:, None]         # shape (3,1) via new axis

```

* * * * *

Mini-exercises (5--10 mins)
--------------------------

1.  Create a 3×3 array with values 1..9 and compute:

    -   the row means, the column sums, and the overall standard deviation.

2.  Make an array `t` of 100 points from 0 to 2π2\pi. Compute\
    y=sin⁡(t)+0.3cos⁡(3t)y = \sin(t) + 0.3\cos(3t). Return `y.min()`, `y.max()`, and the indices where `y > 1`.

3.  Using broadcasting, standardize each column of

```
X = np.array([[ 1.,  2.,  3.],
              [ 2.,  4.,  6.],
              [ 3.,  6.,  9.]])

```

to zero mean and unit variance.

* * * * *
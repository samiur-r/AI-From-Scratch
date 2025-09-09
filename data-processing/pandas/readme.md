# Pandas Quick Reference

Pandas is a powerful data analysis and manipulation library for Python. It provides data structures like DataFrame and Series, along with tools for reading/writing data, cleaning, transforming, and analyzing datasets. Pandas is built on top of NumPy and is essential for data science workflows.

### Installation
```bash
pip install pandas
```

### Importing Pandas

```
import pandas as pd
import numpy as np
```

* * * * *

2\. Data Structures
-------------------

```
# Series - 1D labeled array
s = pd.Series([1, 3, 5, 7, 9])
print(s)  # 0    1, 1    3, 2    5, 3    7, 4    9

# Series with custom index
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s2['a'])  # 10

# DataFrame - 2D labeled data structure
data = {'Name': ['Alice', 'Bob', 'Charlie'], 
        'Age': [25, 30, 35], 
        'City': ['NY', 'LA', 'Chicago']}
df = pd.DataFrame(data)
print(df.shape)  # (3, 3)

```

* * * * *

3\. Creating DataFrames
-----------------------

```
# From dictionary
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# From lists
df = pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=['A', 'B'])

# From NumPy array
arr = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(arr, columns=['X', 'Y'])

# Empty DataFrame
empty_df = pd.DataFrame(columns=['Name', 'Age'])

```

* * * * *

4\. Reading & Writing Data
--------------------------

```
# Read from CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', index_col=0)  # Use first column as index

# Read from Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read from JSON
df = pd.read_json('data.json')

# Write to CSV
df.to_csv('output.csv', index=False)  # Don't include index

# Write to Excel
df.to_excel('output.xlsx', index=False)

```

* * * * *

5\. DataFrame Attributes
------------------------

```
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

print(df.shape)      # (3, 3) - rows, columns
print(df.size)       # 9 - total elements
print(df.ndim)       # 2 - dimensions
print(df.columns)    # Index(['A', 'B', 'C'])
print(df.index)      # RangeIndex(start=0, stop=3, step=1)
print(df.dtypes)     # Data types of each column
print(df.info())     # Comprehensive info about DataFrame

```

* * * * *

6\. Indexing and Selection
--------------------------

```
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 
                   'Age': [25, 30, 35], 
                   'Salary': [50000, 60000, 70000]})

# Select column
print(df['Name'])        # Series
print(df[['Name', 'Age']])  # DataFrame with multiple columns

# Select rows by index
print(df.iloc[0])        # First row by position
print(df.loc[0])         # First row by label
print(df.iloc[0:2])      # First two rows

# Boolean indexing
print(df[df['Age'] > 30])    # Rows where Age > 30
print(df[df['Name'].str.contains('A')])  # Names containing 'A'

# Select specific cell
print(df.at[0, 'Name'])     # 'Alice'
print(df.iat[0, 0])         # 'Alice' (by position)

```

* * * * *

7\. Data Cleaning
-----------------

```
df = pd.DataFrame({'A': [1, 2, None, 4], 
                   'B': [5, None, 7, 8], 
                   'C': ['x', 'y', 'z', 'z']})

# Handle missing values
print(df.isnull())           # Boolean mask for null values
print(df.isnull().sum())     # Count nulls per column
df_clean = df.dropna()       # Remove rows with any null
df_filled = df.fillna(0)     # Fill nulls with 0
df_filled = df.fillna(method='ffill')  # Forward fill

# Remove duplicates
df_unique = df.drop_duplicates()
df_unique = df.drop_duplicates(subset=['C'])  # Based on column C

# Data type conversion
df['A'] = df['A'].astype('float')
df['C'] = df['C'].astype('category')

```

* * * * *

8\. Data Manipulation
---------------------

```
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Add new column
df['C'] = df['A'] + df['B']  # [5, 7, 9]
df['D'] = [10, 20, 30]       # From list

# Apply functions
df['A_squared'] = df['A'].apply(lambda x: x**2)  # [1, 4, 9]
df['sum'] = df.apply(lambda row: row['A'] + row['B'], axis=1)

# Rename columns
df = df.rename(columns={'A': 'col_A', 'B': 'col_B'})

# Drop columns/rows
df = df.drop(['D'], axis=1)     # Drop column
df = df.drop([0], axis=0)       # Drop row

# Sort
df_sorted = df.sort_values('A')              # Sort by column A
df_sorted = df.sort_values(['A', 'B'])       # Sort by multiple columns

```

* * * * *

9\. Groupby Operations
----------------------

```
df = pd.DataFrame({'Category': ['A', 'B', 'A', 'B', 'A'], 
                   'Value': [10, 20, 30, 40, 50], 
                   'Count': [1, 2, 3, 4, 5]})

# Basic groupby
grouped = df.groupby('Category')
print(grouped.sum())      # Sum by category: A=90, B=60 for Value

# Multiple aggregations
result = df.groupby('Category').agg({
    'Value': ['sum', 'mean'],
    'Count': 'max'
})

# Transform (keeps original shape)
df['Value_mean'] = df.groupby('Category')['Value'].transform('mean')

# Filter groups
large_groups = df.groupby('Category').filter(lambda x: len(x) > 2)

```

* * * * *

10\. Merging and Joining
------------------------

```
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

# Inner join (default)
merged = pd.merge(df1, df2, on='key')  # Only A, B

# Left join
merged = pd.merge(df1, df2, on='key', how='left')  # A, B, C

# Outer join
merged = pd.merge(df1, df2, on='key', how='outer')  # A, B, C, D

# Concatenate
df_concat = pd.concat([df1, df2])           # Vertically
df_concat = pd.concat([df1, df2], axis=1)   # Horizontally

```

* * * * *

11\. String Operations
----------------------

```
df = pd.DataFrame({'Name': ['Alice Smith', 'bob jones', 'CHARLIE BROWN']})

# String methods
df['Name_upper'] = df['Name'].str.upper()          # Uppercase
df['Name_lower'] = df['Name'].str.lower()          # Lowercase
df['First_name'] = df['Name'].str.split().str[0]   # First word
df['Length'] = df['Name'].str.len()                # String length

# Pattern matching
df['Has_Alice'] = df['Name'].str.contains('Alice')  # Boolean
df['Starts_A'] = df['Name'].str.startswith('A')     # Boolean

# Replace
df['Clean_name'] = df['Name'].str.replace('CHARLIE', 'Charlie')

```

* * * * *

12\. Date and Time
------------------

```
# Create datetime
dates = pd.date_range('2023-01-01', periods=5, freq='D')
df = pd.DataFrame({'Date': dates, 'Value': [1, 2, 3, 4, 5]})

# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract components
df['Year'] = df['Date'].dt.year      # 2023
df['Month'] = df['Date'].dt.month    # 1
df['Weekday'] = df['Date'].dt.dayname()  # Monday, Tuesday...

# Set as index
df.set_index('Date', inplace=True)

# Resample (time series)
monthly = df.resample('M').sum()     # Group by month

```

* * * * *

13\. Statistical Operations
---------------------------

```
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# Basic statistics
print(df.describe())     # Summary statistics
print(df.mean())         # A=3.0, B=30.0
print(df.std())          # Standard deviation
print(df.corr())         # Correlation matrix

# More statistics
print(df.quantile(0.5))  # Median (50th percentile)
print(df.skew())         # Skewness
print(df.kurt())         # Kurtosis

# Value counts
df = pd.DataFrame({'Grade': ['A', 'B', 'A', 'C', 'B']})
print(df['Grade'].value_counts())  # A=2, B=2, C=1

```

* * * * *

14\. Performance Tips
---------------------

Use vectorized operations instead of loops!

```
df = pd.DataFrame({'A': range(100000), 'B': range(100000)})

# Slow (iterating)
result_slow = []
for i in df.index:
    result_slow.append(df.loc[i, 'A'] * 2)

# Fast (vectorized)
result_fast = df['A'] * 2

# Use categorical for repeated strings
df['Category'] = df['Category'].astype('category')

# Use appropriate dtypes
df['Int_col'] = df['Int_col'].astype('int32')  # Instead of int64

```

* * * * *

Summary
=======

-   **DataFrames** and **Series** are the core data structures for tabular data.

-   Use **vectorized operations** and **built-in methods** for best performance.

-   Pandas integrates seamlessly with **NumPy, Matplotlib, and scikit-learn**.

-   Master **groupby, merge, and indexing** for advanced data manipulation.
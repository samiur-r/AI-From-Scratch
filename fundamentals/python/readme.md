# Python Advanced Concepts Quick Reference

Python offers powerful features that make code more concise, readable, and efficient. This reference covers advanced concepts that are commonly used in modern Python development, from list comprehensions to decorators and beyond.

* * * * *

1\. List Comprehensions
-----------------------

```
# Basic list comprehension
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]

# With condition
evens = [x for x in numbers if x % 2 == 0]  # [2, 4]

# Nested loops
matrix = [[1, 2], [3, 4], [5, 6]]
flattened = [item for row in matrix for item in row]  # [1, 2, 3, 4, 5, 6]

# Dictionary comprehension
word_lengths = {word: len(word) for word in ['hello', 'world']}  # {'hello': 5, 'world': 5}

# Set comprehension
unique_squares = {x**2 for x in [1, 2, 2, 3]}  # {1, 4, 9}

# Generator expression
gen = (x**2 for x in range(5))  # Generator object, memory efficient

```

* * * * *

2\. Lambda Functions
--------------------

```
# Basic lambda
square = lambda x: x**2
print(square(5))  # 25

# Multiple arguments
add = lambda x, y: x + y
print(add(3, 4))  # 7

# With higher-order functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]

# Sorting with lambda
students = [('Alice', 85), ('Bob', 90), ('Charlie', 78)]
sorted_by_grade = sorted(students, key=lambda x: x[1])  # Sort by grade

# Conditional lambda
max_func = lambda a, b: a if a > b else b
print(max_func(10, 20))  # 20

```

* * * * *

3\. Map, Filter, Reduce
-----------------------

```
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Map - apply function to all elements
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]
strings = list(map(str, numbers))  # ['1', '2', '3', '4', '5']

# Filter - keep elements that match condition  
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]
positive = list(filter(lambda x: x > 0, [-1, 0, 1, 2]))  # [1, 2]

# Reduce - reduce to single value
sum_all = reduce(lambda x, y: x + y, numbers)  # 15
product = reduce(lambda x, y: x * y, numbers)  # 120

# Multiple iterables with map
list1 = [1, 2, 3]
list2 = [4, 5, 6] 
sums = list(map(lambda x, y: x + y, list1, list2))  # [5, 7, 9]

```

* * * * *

4\. F-Strings and String Formatting
-----------------------------------

```
name = "Alice"
age = 30
pi = 3.14159

# F-strings (Python 3.6+)
greeting = f"Hello, {name}!"  # "Hello, Alice!"
info = f"{name} is {age} years old"  # "Alice is 30 years old"

# Expressions in f-strings
result = f"Next year {name} will be {age + 1}"  # "Next year Alice will be 31"

# Formatting numbers
formatted = f"Pi is approximately {pi:.2f}"  # "Pi is approximately 3.14"
percentage = f"Success rate: {0.85:.1%}"  # "Success rate: 85.0%"

# Alignment and padding
aligned = f"{name:>10}"    # "     Alice" (right aligned)
padded = f"{age:04d}"      # "0030" (zero padded)

# .format() method
old_style = "Hello, {}! You are {} years old.".format(name, age)
named = "Hello, {name}! You are {age} years old.".format(name=name, age=age)

```

* * * * *

5\. Decorators
--------------

```
# Simple decorator
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# say_hello() outputs:
# Before function call
# Hello!
# After function call

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

# Built-in decorators
class MyClass:
    @property
    def value(self):
        return self._value
    
    @staticmethod
    def utility_function():
        return "This is a static method"

```

* * * * *

6\. Generators and Iterators
----------------------------

```
# Generator function
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

fib = fibonacci(5)
print(list(fib))  # [0, 1, 1, 2, 3]

# Generator expression
squares_gen = (x**2 for x in range(5))
print(next(squares_gen))  # 0
print(next(squares_gen))  # 1

# Custom iterator
class CountDown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Usage: for i in CountDown(3): print(i)  # 3, 2, 1

# yield from
def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

```

* * * * *

7\. Context Managers
--------------------

```
# Built-in context manager
with open('file.txt', 'w') as f:
    f.write('Hello, World!')  # File automatically closed

# Custom context manager (class-based)
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end = time.time()
        print(f"Execution time: {self.end - self.start:.2f} seconds")

# Usage: with Timer(): time.sleep(1)

# Context manager with contextlib
from contextlib import contextmanager

@contextmanager
def database_connection():
    print("Opening database connection")
    conn = "fake_connection"
    try:
        yield conn
    finally:
        print("Closing database connection")

# Usage: with database_connection() as conn: pass

```

* * * * *

8\. Exception Handling
----------------------

```
# Basic exception handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("No exception occurred")
finally:
    print("This always executes")

# Multiple exceptions
try:
    value = int(input())
    result = 10 / value
except (ValueError, ZeroDivisionError) as e:
    print(f"Error: {e}")

# Custom exceptions
class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def validate_age(age):
    if age < 0:
        raise CustomError("Age cannot be negative")
    return age

# Exception chaining
try:
    validate_age(-5)
except CustomError as e:
    raise ValueError("Invalid input") from e

```

* * * * *

9\. Advanced Data Structures
----------------------------

```
from collections import defaultdict, Counter, deque, namedtuple
from dataclasses import dataclass

# defaultdict
dd = defaultdict(list)
dd['key'].append('value')  # No KeyError, creates empty list

# Counter
text = "hello world"
char_count = Counter(text)  # Counter({'l': 3, 'o': 2, 'h': 1, ...})
most_common = char_count.most_common(2)  # [('l', 3), ('o', 2)]

# deque (double-ended queue)
dq = deque([1, 2, 3])
dq.appendleft(0)  # deque([0, 1, 2, 3])
dq.append(4)      # deque([0, 1, 2, 3, 4])
dq.popleft()      # 0, deque([1, 2, 3, 4])

# namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20

# dataclasses (Python 3.7+)
@dataclass
class Person:
    name: str
    age: int
    
    def greet(self):
        return f"Hi, I'm {self.name}"

```

* * * * *

10\. Advanced Function Features
-------------------------------

```
# *args and **kwargs
def flexible_function(*args, **kwargs):
    print(f"Args: {args}")      # Tuple of positional arguments
    print(f"Kwargs: {kwargs}")  # Dict of keyword arguments

flexible_function(1, 2, 3, name="Alice", age=30)

# Unpacking
numbers = [1, 2, 3]
print(*numbers)  # 1 2 3 (unpacks list)

person = {'name': 'Bob', 'age': 25}
flexible_function(**person)  # Unpacks dict as keyword args

# Partial functions
from functools import partial

def multiply(x, y):
    return x * y

double = partial(multiply, 2)  # Fix first argument
print(double(5))  # 10

# Function annotations
def add_numbers(x: int, y: int) -> int:
    """Add two integers and return the result."""
    return x + y

```

* * * * *

11\. Comprehensions and Itertools
---------------------------------

```
import itertools

# Advanced comprehensions
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
diagonal = [matrix[i][i] for i in range(len(matrix))]  # [1, 5, 9]

# Conditional expressions in comprehensions
numbers = range(10)
result = [x if x % 2 == 0 else -x for x in numbers]  # [0, -1, 2, -3, ...]

# itertools examples
# Chain multiple iterables
chained = list(itertools.chain([1, 2], [3, 4], [5, 6]))  # [1, 2, 3, 4, 5, 6]

# Combinations and permutations
combos = list(itertools.combinations([1, 2, 3], 2))  # [(1, 2), (1, 3), (2, 3)]
perms = list(itertools.permutations([1, 2, 3], 2))   # [(1, 2), (1, 3), (2, 1), ...]

# Groupby
data = [('A', 1), ('A', 2), ('B', 3), ('B', 4)]
grouped = {k: list(v) for k, v in itertools.groupby(data, key=lambda x: x[0])}

```

* * * * *

12\. Performance Tips
---------------------

```
import timeit

# Use built-ins when possible
# Slow
squares_slow = []
for i in range(1000):
    squares_slow.append(i**2)

# Fast
squares_fast = [i**2 for i in range(1000)]

# Even faster for simple operations
squares_fastest = list(map(lambda x: x**2, range(1000)))

# String concatenation
# Slow
result = ""
for i in range(1000):
    result += str(i)

# Fast
result = "".join(str(i) for i in range(1000))

# Use enumerate instead of range(len())
items = ['a', 'b', 'c']
# Better
for i, item in enumerate(items):
    print(f"{i}: {item}")

# Use zip for parallel iteration
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

```

* * * * *

Summary
=======

-   **List comprehensions** and **generator expressions** make code concise and readable.

-   **Lambda functions** are perfect for short, one-line functions used with map/filter.

-   **Decorators** add functionality to functions without modifying their code.

-   **Context managers** ensure proper resource management and cleanup.

-   **F-strings** provide the most readable and efficient string formatting.

-   Master these concepts to write more **Pythonic** and **efficient** code.
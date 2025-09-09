# Matplotlib Quick Reference

Matplotlib is the most popular plotting library for Python, providing a comprehensive set of tools for creating static, animated, and interactive visualizations. It offers both a MATLAB-like interface and object-oriented API, making it suitable for simple plots and complex data visualizations.

### Installation
```bash
pip install matplotlib
```

### Importing Matplotlib

```
import matplotlib.pyplot as plt
import numpy as np
```

* * * * *

2\. Basic Plotting
------------------

```
# Simple line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)  # Creates straight line from (1,2) to (5,10)
plt.show()      # Displays plot window

# Multiple lines
x = np.linspace(0, 10, 100)  # 100 points from 0 to 10
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label='sin(x)')  # Blue sine wave
plt.plot(x, y2, label='cos(x)')  # Orange cosine wave
plt.legend()  # Shows legend box with labels
plt.show()    # Displays both curves

# Quick plotting with NumPy
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))  # Smooth sine wave from 0 to 2π

```

* * * * *

3\. Plot Customization
----------------------

```
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color='red', linestyle='--', linewidth=2, marker='o', markersize=4)
# Creates red dashed line with small circle markers

# Labels and title
plt.xlabel('X-axis')    # Adds label to x-axis
plt.ylabel('Y-axis')    # Adds label to y-axis
plt.title('Sine Wave')  # Adds title at top of plot

# Grid and limits
plt.grid(True, alpha=0.3)  # Adds light gray grid lines
plt.xlim(0, 10)           # Sets x-axis range from 0 to 10
plt.ylim(-1.5, 1.5)       # Sets y-axis range from -1.5 to 1.5

# Color options: 'red', 'blue', 'green', '#FF5733', (0.1, 0.2, 0.5)
# Line styles: '-', '--', '-.', ':', 'solid', 'dashed'
# Markers: 'o', 's', '^', 'v', '<', '>', 'D', '*', '+'

```

* * * * *

4\. Different Plot Types
------------------------

```
# Scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.randn(100)
plt.scatter(x, y, c=colors, alpha=0.6, cmap='viridis')
# Creates 100 colored dots, colors vary from purple to yellow
plt.colorbar()  # Adds color scale bar on the side

# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values, color=['red', 'blue', 'green', 'orange'])
# Creates 4 vertical bars with different colors and heights

# Horizontal bar plot
plt.barh(categories, values)  # Same as above but horizontal bars

# Histogram
data = np.random.normal(100, 15, 1000)  # Normal distribution, mean=100, std=15
plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
# Shows bell curve distribution with 30 bins, black edges

# Pie chart
sizes = [30, 25, 20, 25]
labels = ['A', 'B', 'C', 'D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
# Creates circle divided into 4 sections with percentages shown

```

* * * * *

5\. Subplots and Figure Management
----------------------------------

```
# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# Creates 2x2 grid of subplots in 10x8 inch figure

# Plot on different subplots
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))         # Top-left: sine wave
axes[0, 0].set_title('Sine')

axes[0, 1].plot(x, np.cos(x))         # Top-right: cosine wave
axes[0, 1].set_title('Cosine')

axes[1, 0].plot(x, np.tan(x))         # Bottom-left: tangent (with vertical lines)
axes[1, 0].set_title('Tangent')

axes[1, 1].plot(x, np.exp(-x))        # Bottom-right: decay curve
axes[1, 1].set_title('Exponential Decay')

plt.tight_layout()  # Adjusts spacing to prevent overlap
plt.show()          # Displays all 4 subplots together

# Single subplot
plt.subplot(2, 1, 1)  # Top half of figure
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)  # Bottom half of figure  
plt.plot(x, np.cos(x))

```

* * * * *

6\. Styling and Themes
----------------------

```
# Available styles
print(plt.style.available)  # Lists: ['default', 'classic', 'seaborn', 'ggplot'...]

# Use a style
plt.style.use('seaborn-v0_8')  # Changes plot appearance to seaborn style
plt.plot([1, 2, 3], [1, 4, 9])  # Plot now has seaborn styling

# Custom styling with rcParams
plt.rcParams['font.size'] = 14        # All text becomes size 14
plt.rcParams['lines.linewidth'] = 2   # All lines become thicker
plt.rcParams['figure.figsize'] = (10, 6)  # Default figure size changes

# Color cycles and palettes
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['red', 'blue', 'green'])
# Next plots will cycle through red, blue, green colors

# Reset to default
plt.rcdefaults()  # Returns all settings to original defaults

# Context managers for temporary styling
with plt.style.context('dark_background'):
    plt.plot([1, 2, 3], [1, 4, 9])  # Plot with dark background
    plt.show()  # Only this plot uses dark style

```

* * * * *

7\. Annotations and Text
------------------------

```
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)

# Add text
plt.text(5, 0.5, 'Maximum point', fontsize=12, ha='center')

# Annotations with arrows
max_idx = np.argmax(y)
plt.annotate('Peak', xy=(x[max_idx], y[max_idx]), xytext=(8, 0.8),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12)

# Mathematical expressions (LaTeX)
plt.xlabel(r'$\theta$')  # Greek letters
plt.ylabel(r'$\sin(\theta)$')
plt.title(r'$y = \sin(x)$ function')

# Text positioning: ha='left'/'center'/'right', va='bottom'/'center'/'top'

```

* * * * *

8\. Statistical Plots
---------------------

```
# Box plot
data = [np.random.normal(100, 10, 100) for _ in range(4)]
plt.boxplot(data, labels=['A', 'B', 'C', 'D'])
# Shows 4 box plots with median, quartiles, and outliers

# Error bars
x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 17, 20]
errors = [1, 2, 1.5, 2.5, 1.8]
plt.errorbar(x, y, yerr=errors, fmt='o-', capsize=5)
# Line plot with vertical error bars and caps

# Fill between (confidence intervals)
x = np.linspace(0, 10, 100)
y = np.sin(x)
y_upper = y + 0.2
y_lower = y - 0.2
plt.plot(x, y, 'b-', label='sin(x)')        # Blue sine wave
plt.fill_between(x, y_lower, y_upper, alpha=0.3, label='±0.2')
# Light blue shaded area around the sine wave

# Heatmap
data = np.random.randn(10, 10)
plt.imshow(data, cmap='coolwarm', aspect='auto')
# 10x10 grid with red/blue colors representing values
plt.colorbar()  # Color scale showing value-to-color mapping

```

* * * * *

9\. 3D Plotting
---------------

```
from mpl_toolkits.mplot3d import Axes3D

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.randn(100)
y = np.random.randn(100) 
z = np.random.randn(100)
ax.scatter(x, y, z)

# 3D surface plot
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# 3D line plot
t = np.linspace(0, 20, 100)
x = np.sin(t)
y = np.cos(t)
z = t
ax.plot(x, y, z)

```

* * * * *

10\. Saving Figures
-------------------

```
# Save in different formats
plt.plot([1, 2, 3], [1, 4, 9])

plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # Saves high-res PNG file
plt.savefig('plot.pdf', bbox_inches='tight')           # Saves vector PDF file
plt.savefig('plot.svg')                                # Saves vector SVG file
plt.savefig('plot.jpg', quality=95)                    # Saves JPEG with 95% quality

# Save with transparent background
plt.savefig('plot_transparent.png', transparent=True)  # PNG with no background

# Control figure size when saving
fig = plt.figure(figsize=(12, 8))  # Creates 12x8 inch figure
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig('large_plot.png')      # Saves large resolution image
plt.close(fig)  # Closes figure to free memory

```

* * * * *

11\. Interactive Features
-------------------------

```
# Interactive mode
plt.ion()  # Turn on interactive mode
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
# Window stays open, can add more plots

plt.ioff()  # Turn off interactive mode

# Event handling (basic example)
def onclick(event):
    print(f'Clicked at: {event.xdata}, {event.ydata}')

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
fig.canvas.mpl_connect('button_press_event', onclick)

# Widgets (requires additional setup)
from matplotlib.widgets import Slider, Button

# Animation (basic example)
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def animate(frame):
    line.set_ydata(np.sin(x + frame/10))
    return line,

ani = FuncAnimation(fig, animate, frames=100, blit=True)

```

* * * * *

12\. Common Patterns and Tips
-----------------------------

```
# Figure and axes object-oriented approach
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_title('Title')

# Multiple y-axes
fig, ax1 = plt.subplots()
ax1.plot([1, 2, 3], [1, 4, 9], 'b-')
ax1.set_ylabel('Left Y', color='b')

ax2 = ax1.twinx()  # Share x-axis
ax2.plot([1, 2, 3], [10, 20, 30], 'r-')
ax2.set_ylabel('Right Y', color='r')

# Logarithmic scales
plt.semilogy([1, 2, 3], [1, 100, 10000])  # Log y-axis
plt.semilogx([1, 10, 100], [1, 2, 3])     # Log x-axis
plt.loglog([1, 10, 100], [1, 100, 10000]) # Both log

# Date plotting
import datetime as dt
dates = [dt.date(2023, 1, i) for i in range(1, 11)]
values = np.random.randn(10).cumsum()
plt.plot(dates, values)
plt.xticks(rotation=45)

```

* * * * *

13\. Performance and Memory
---------------------------

```
# Clear figures to save memory
plt.clf()    # Clear current figure
plt.cla()    # Clear current axes
plt.close()  # Close current figure
plt.close('all')  # Close all figures

# Batch processing
for i in range(10):
    plt.figure()
    plt.plot(np.random.randn(100))
    plt.savefig(f'plot_{i}.png')
    plt.close()  # Important: close each figure

# Use generators for large datasets
def data_generator():
    for i in range(1000000):
        yield i, np.sin(i/1000)

# Plot in chunks
chunk_size = 1000
x_chunk, y_chunk = [], []
for x, y in data_generator():
    x_chunk.append(x)
    y_chunk.append(y)
    if len(x_chunk) >= chunk_size:
        plt.plot(x_chunk, y_chunk)
        x_chunk, y_chunk = [], []

```

* * * * *

Summary
=======

-   **pyplot** provides MATLAB-like interface, **object-oriented API** gives more control.

-   Use **subplots** and **figure management** for complex multi-panel visualizations.

-   **Customization** includes colors, markers, styles, annotations, and themes.

-   **Save figures** in multiple formats (PNG, PDF, SVG) with proper DPI settings.

-   Master **statistical plots** and **3D plotting** for advanced data visualization.

-   Always **close figures** in loops to prevent memory issues.
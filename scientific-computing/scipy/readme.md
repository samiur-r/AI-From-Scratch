# SciPy Quick Reference

SciPy is a fundamental library for scientific and technical computing in Python. Built on NumPy, it provides algorithms for optimization, linear algebra, integration, interpolation, special functions, FFT, signal processing, image processing, ODE solving, and statistical functions. Essential for scientific computing workflows.

### Installation
```bash
pip install scipy
```

### Importing SciPy

```
import scipy
import numpy as np
from scipy import linalg, optimize, integrate, interpolate, stats
```

* * * * *

2\. Linear Algebra (scipy.linalg)
---------------------------------

```
from scipy import linalg
import numpy as np

A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

# Solve linear system Ax = b
x = linalg.solve(A, b)  # [−4, 4.5]

# Matrix decompositions
# LU decomposition
P, L, U = linalg.lu(A)  # PA = LU

# QR decomposition  
Q, R = linalg.qr(A)     # A = QR

# SVD (Singular Value Decomposition)
U, s, Vt = linalg.svd(A)  # A = U @ diag(s) @ Vt

# Eigenvalues and eigenvectors
eigenvals, eigenvecs = linalg.eig(A)
print(eigenvals)  # [-0.37, 5.37]

# Matrix functions
matrix_exp = linalg.expm(A)      # Matrix exponential
matrix_sqrt = linalg.sqrtm(A)    # Matrix square root
matrix_log = linalg.logm(A)      # Matrix logarithm

# Norms and condition numbers
frobenius_norm = linalg.norm(A, 'fro')  # Frobenius norm
condition_num = linalg.cond(A)          # Condition number

# Pseudoinverse
A_pinv = linalg.pinv(A)  # Moore-Penrose pseudoinverse

```

* * * * *

3\. Optimization (scipy.optimize)
---------------------------------

```
from scipy import optimize
import numpy as np

# Minimize scalar functions
def f(x):
    return x**2 + 10*np.sin(x)  # Function with local minima

# Find minimum
result = optimize.minimize_scalar(f, bounds=(-10, 10), method='bounded')
print(f"Minimum at x = {result.x:.3f}, f(x) = {result.fun:.3f}")

# Multivariable optimization
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Starting point
x0 = [0, 0]
result = optimize.minimize(rosenbrock, x0, method='BFGS')
print(f"Minimum at: {result.x}")  # Should be close to [1, 1]

# With constraints
def constraint(x):
    return x[0] + x[1] - 1  # x[0] + x[1] = 1

cons = {'type': 'eq', 'fun': constraint}
result = optimize.minimize(rosenbrock, x0, method='SLSQP', constraints=cons)

# Root finding
def equation(x):
    return x**3 - 2*x - 5

root = optimize.root_scalar(equation, bracket=[2, 3], method='brentq')
print(f"Root: {root.root}")  # x ≈ 2.094

# Curve fitting
def model(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate noisy data
x_data = np.linspace(0, 4, 50)
y_data = model(x_data, 2.5, 1.3, 0.5) + 0.2*np.random.normal(size=50)

# Fit parameters
popt, pcov = optimize.curve_fit(model, x_data, y_data)
print(f"Fitted parameters: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}")

```

* * * * *

4\. Integration (scipy.integrate)
---------------------------------

```
from scipy import integrate
import numpy as np

# Numerical integration (quadrature)
def integrand(x):
    return np.exp(-x**2)  # Gaussian function

# Integrate from 0 to infinity
result, error = integrate.quad(integrand, 0, np.inf)
print(f"∫₀^∞ e^(-x²) dx = {result:.6f} ± {error:.2e}")  # Should be √π/2

# Double integration
def double_integrand(y, x):
    return x * y**2

# ∫₀¹ ∫₀¹ xy² dy dx
result = integrate.dblquad(double_integrand, 0, 1, lambda x: 0, lambda x: 1)
print(f"Double integral: {result[0]:.6f}")  # Should be 1/6

# Triple integration
def triple_integrand(z, y, x):
    return x + y + z

result = integrate.tplquad(triple_integrand, 0, 1, lambda x: 0, lambda x: 1, 
                          lambda x, y: 0, lambda x, y: 1)

# Solving ODEs (Ordinary Differential Equations)
def dydt(t, y):
    return -2 * y  # dy/dt = -2y, solution: y = y₀e^(-2t)

# Initial conditions
t_span = (0, 5)  # Time range
y0 = [1]         # y(0) = 1

# Solve ODE
sol = integrate.solve_ivp(dydt, t_span, y0, dense_output=True)
t_eval = np.linspace(0, 5, 100)
y_eval = sol.sol(t_eval)

# System of ODEs (e.g., predator-prey)
def lotka_volterra(t, z):
    x, y = z
    dxdt = x - x*y      # Prey equation
    dydt = -y + x*y     # Predator equation
    return [dxdt, dydt]

sol = integrate.solve_ivp(lotka_volterra, (0, 15), [1, 1], dense_output=True)

```

* * * * *

5\. Interpolation (scipy.interpolate)
-------------------------------------

```
from scipy import interpolate
import numpy as np

# 1D interpolation
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])  # y = x²

# Linear interpolation
f_linear = interpolate.interp1d(x, y, kind='linear')
x_new = np.linspace(0, 5, 50)
y_linear = f_linear(x_new)

# Cubic spline interpolation
f_cubic = interpolate.interp1d(x, y, kind='cubic')
y_cubic = f_cubic(x_new)

# B-spline interpolation
tck = interpolate.splrep(x, y, s=0)  # s=0 for exact interpolation
y_spline = interpolate.splev(x_new, tck)

# 2D interpolation
x = np.arange(0, 5, 1)
y = np.arange(0, 5, 1)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # Function values

# Regular grid interpolation
f_2d = interpolate.interp2d(x, y, Z, kind='cubic')
x_new_2d = np.linspace(0, 4, 20)
y_new_2d = np.linspace(0, 4, 20)
Z_new = f_2d(x_new_2d, y_new_2d)

# Scattered data interpolation
np.random.seed(42)
x_scatter = np.random.random(100) * 4
y_scatter = np.random.random(100) * 4
z_scatter = x_scatter**2 + y_scatter**2 + np.random.normal(0, 0.1, 100)

# Radial basis function interpolation
rbf = interpolate.Rbf(x_scatter, y_scatter, z_scatter, function='multiquadric')
X_grid, Y_grid = np.meshgrid(np.linspace(0, 4, 50), np.linspace(0, 4, 50))
Z_rbf = rbf(X_grid, Y_grid)

```

* * * * *

6\. Statistics (scipy.stats)
----------------------------

```
from scipy import stats
import numpy as np

# Probability distributions
# Normal distribution
mu, sigma = 100, 15
normal_dist = stats.norm(loc=mu, scale=sigma)

# Generate random samples
samples = normal_dist.rvs(size=1000)

# PDF (Probability Density Function)
x = np.linspace(50, 150, 100)
pdf_vals = normal_dist.pdf(x)

# CDF (Cumulative Distribution Function)
cdf_vals = normal_dist.cdf(x)

# Probability calculations
prob_below_85 = normal_dist.cdf(85)      # P(X < 85)
prob_above_115 = 1 - normal_dist.cdf(115)  # P(X > 115)
prob_between = normal_dist.cdf(115) - normal_dist.cdf(85)  # P(85 < X < 115)

# Other distributions
# Binomial
binomial = stats.binom(n=20, p=0.3)
binom_samples = binomial.rvs(size=1000)

# Poisson
poisson = stats.poisson(mu=3)
poisson_samples = poisson.rvs(size=1000)

# Exponential
exponential = stats.expon(scale=2)  # scale = 1/λ
exp_samples = exponential.rvs(size=1000)

# Statistical tests
# t-test (one sample)
data = np.random.normal(100, 10, 50)
t_stat, p_value = stats.ttest_1samp(data, 100)  # Test if mean = 100
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# t-test (two independent samples)
group1 = np.random.normal(100, 10, 50)
group2 = np.random.normal(105, 10, 50)
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"Two-sample t-test: t={t_stat:.3f}, p={p_value:.3f}")

# Chi-square test
observed = np.array([16, 18, 16, 14, 12, 12])
expected = np.array([16, 16, 16, 16, 16, 8])
chi2_stat, p_value = stats.chisquare(observed, expected)

# Correlation tests
x = np.random.randn(100)
y = 2*x + np.random.randn(100)
pearson_corr, p_val_pearson = stats.pearsonr(x, y)
spearman_corr, p_val_spearman = stats.spearmanr(x, y)

```

* * * * *

7\. Signal Processing (scipy.signal)
------------------------------------

```
from scipy import signal
import numpy as np

# Generate signals
t = np.linspace(0, 1, 1000)
freq1, freq2 = 5, 20
sig = np.sin(2*np.pi*freq1*t) + 0.5*np.sin(2*np.pi*freq2*t)
noise = 0.2*np.random.randn(len(t))
noisy_sig = sig + noise

# Filtering
# Low-pass Butterworth filter
nyquist = 0.5 * 1000  # Nyquist frequency (sampling_rate/2)
low_cutoff = 10 / nyquist
b, a = signal.butter(4, low_cutoff, btype='low')
filtered_sig = signal.filtfilt(b, a, noisy_sig)

# Band-pass filter
low_cutoff = 3 / nyquist
high_cutoff = 8 / nyquist
b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
bandpass_sig = signal.filtfilt(b, a, noisy_sig)

# Moving average filter
window_size = 50
moving_avg = signal.savgol_filter(noisy_sig, window_size, 3)  # Savitzky-Golay

# Spectral analysis
# FFT
frequencies, fft_vals = signal.periodogram(noisy_sig, fs=1000)

# Spectrogram
f_spec, t_spec, Sxx = signal.spectrogram(noisy_sig, fs=1000)

# Cross-correlation
sig1 = np.random.randn(100)
sig2 = np.roll(sig1, 10) + 0.1*np.random.randn(100)  # Shifted version
correlation = signal.correlate(sig1, sig2, mode='full')
lags = signal.correlation_lags(len(sig1), len(sig2), mode='full')

# Peak finding
peaks, properties = signal.find_peaks(sig, height=0.5, distance=50)

# Window functions
hamming_window = signal.hamming(100)
hanning_window = signal.hann(100)
blackman_window = signal.blackman(100)

```

* * * * *

8\. Fourier Transforms (scipy.fft)
----------------------------------

```
from scipy import fft
import numpy as np

# Generate signal
t = np.linspace(0, 1, 1000, endpoint=False)
sig = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*10*t) + np.random.normal(0, 0.1, 1000)

# 1D FFT
fft_vals = fft.fft(sig)
freqs = fft.fftfreq(len(sig), t[1] - t[0])

# Real FFT (for real-valued signals)
rfft_vals = fft.rfft(sig)
rfreqs = fft.rfftfreq(len(sig), t[1] - t[0])

# Inverse FFT
reconstructed = fft.ifft(fft_vals)

# 2D FFT (for images)
image = np.random.randn(256, 256)
fft_2d = fft.fft2(image)
fft_2d_shifted = fft.fftshift(fft_2d)  # Center zero frequency

# Power spectral density
psd = np.abs(fft_vals)**2

# Filtering in frequency domain
# Create a low-pass filter mask
mask = np.abs(freqs) < 8  # Keep frequencies below 8 Hz
fft_filtered = fft_vals * mask
filtered_signal = fft.ifft(fft_filtered).real

# DCT (Discrete Cosine Transform)
from scipy.fft import dct, idct
dct_vals = dct(sig, type=2, norm='ortho')
reconstructed_dct = idct(dct_vals, type=2, norm='ortho')

```

* * * * *

9\. Sparse Matrices (scipy.sparse)
----------------------------------

```
from scipy import sparse
import numpy as np

# Create sparse matrices
# From dense matrix
dense = np.array([[1, 0, 0, 2], [0, 0, 3, 0], [4, 0, 0, 5]])
sparse_csr = sparse.csr_matrix(dense)  # Compressed Sparse Row
sparse_csc = sparse.csc_matrix(dense)  # Compressed Sparse Column

# Create directly
row = np.array([0, 0, 1, 2, 2])
col = np.array([0, 3, 2, 0, 3])
data = np.array([1, 2, 3, 4, 5])
sparse_coo = sparse.coo_matrix((data, (row, col)), shape=(3, 4))

# Convert between formats
sparse_csr = sparse_coo.tocsr()
sparse_csc = sparse_coo.tocsc()

# Sparse matrix operations
A = sparse.random(1000, 1000, density=0.01)  # Random sparse matrix
B = sparse.random(1000, 1000, density=0.01)

# Matrix multiplication
C = A @ B  # or A.dot(B)

# Element access
print(A[0, 0])  # Access single element
print(A[0:5, 0:5].toarray())  # Convert submatrix to dense

# Sparse linear algebra
from scipy.sparse.linalg import spsolve, norm
b = np.random.randn(1000)
x = spsolve(A, b)  # Solve Ax = b

# Eigenvalue problems
from scipy.sparse.linalg import eigs
eigenvals, eigenvecs = eigs(A, k=6, which='LM')  # 6 largest magnitude eigenvalues

```

* * * * *

10\. Spatial Data Structures (scipy.spatial)
---------------------------------------------

```
from scipy import spatial
import numpy as np

# Generate random points
np.random.seed(42)
points = np.random.random((100, 2)) * 10

# Nearest neighbors
tree = spatial.KDTree(points)

# Find k nearest neighbors
query_point = [5, 5]
distances, indices = tree.query(query_point, k=5)
nearest_points = points[indices]

# Range queries
indices_in_range = tree.query_ball_point(query_point, r=2.0)
points_in_range = points[indices_in_range]

# Distance calculations
# Pairwise distances
dist_matrix = spatial.distance_matrix(points[:10], points[:10])

# Specific distance metrics
euclidean_dist = spatial.distance.euclidean([1, 2], [4, 6])
manhattan_dist = spatial.distance.cityblock([1, 2], [4, 6])
cosine_dist = spatial.distance.cosine([1, 2, 3], [4, 5, 6])

# Convex hull
hull = spatial.ConvexHull(points)
hull_points = points[hull.vertices]

# Delaunay triangulation
tri = spatial.Delaunay(points)
# Check if point is inside convex hull
test_point = [5, 5]
is_inside = tri.find_simplex(test_point) >= 0

# Voronoi diagram
vor = spatial.Voronoi(points)
voronoi_vertices = vor.vertices
voronoi_regions = vor.regions

# Procrustes analysis (shape comparison)
# Compare two sets of points
set1 = np.random.random((10, 2))
set2 = set1 + 0.1*np.random.random((10, 2))  # Slightly different
mtx1, mtx2, disparity = spatial.procrustes(set1, set2)

```

* * * * *

11\. Image Processing (scipy.ndimage)
-------------------------------------

```
from scipy import ndimage
import numpy as np

# Create sample image
image = np.random.random((100, 100))
# Add some structure
image[20:80, 20:80] = 1.0

# Filtering
# Gaussian filter (blur)
blurred = ndimage.gaussian_filter(image, sigma=2)

# Median filter (noise reduction)
noisy_image = image + 0.1*np.random.random(image.shape)
denoised = ndimage.median_filter(noisy_image, size=3)

# Edge detection
edges = ndimage.sobel(image)  # Sobel filter
laplacian = ndimage.laplace(image)  # Laplacian

# Morphological operations
# Binary image
binary_image = image > 0.5

# Erosion and dilation
eroded = ndimage.binary_erosion(binary_image)
dilated = ndimage.binary_dilation(binary_image)

# Opening and closing
opened = ndimage.binary_opening(binary_image)
closed = ndimage.binary_closing(binary_image)

# Geometric transformations
# Rotation
rotated = ndimage.rotate(image, angle=45, reshape=False)

# Shifting
shifted = ndimage.shift(image, shift=[10, 5])

# Zooming
zoomed = ndimage.zoom(image, zoom=1.5)

# Affine transformation
transformation_matrix = np.array([[1.2, 0.1], [0.1, 1.2]])
transformed = ndimage.affine_transform(image, transformation_matrix)

# Measurements
# Label connected components
labeled, num_labels = ndimage.label(binary_image)

# Measure properties
sizes = ndimage.sum(binary_image, labeled, range(1, num_labels+1))
centers = ndimage.center_of_mass(image, labeled, range(1, num_labels+1))

```

* * * * *

12\. Special Functions (scipy.special)
--------------------------------------

```
from scipy import special
import numpy as np

# Gamma function and related
x = np.linspace(0.1, 5, 100)
gamma_vals = special.gamma(x)        # Γ(x)
loggamma_vals = special.loggamma(x)  # ln(Γ(x))
digamma_vals = special.digamma(x)    # ψ(x) = Γ'(x)/Γ(x)

# Beta function
beta_vals = special.beta(2, 3)  # B(2,3) = Γ(2)Γ(3)/Γ(5)

# Error functions
erf_vals = special.erf(x)      # Error function
erfc_vals = special.erfc(x)    # Complementary error function
erfcx_vals = special.erfcx(x)  # Scaled complementary error function

# Bessel functions
j0_vals = special.j0(x)  # Bessel function of first kind, order 0
j1_vals = special.j1(x)  # Bessel function of first kind, order 1
y0_vals = special.y0(x)  # Bessel function of second kind, order 0

# Modified Bessel functions
i0_vals = special.i0(x)  # Modified Bessel function of first kind, order 0
k0_vals = special.k0(x)  # Modified Bessel function of second kind, order 0

# Combinatorial functions
factorial_vals = special.factorial(5)  # 5! = 120
comb_vals = special.comb(10, 3, exact=True)  # C(10,3) = 120
perm_vals = special.perm(10, 3, exact=True)  # P(10,3) = 720

# Orthogonal polynomials
# Legendre polynomials
legendre_vals = special.eval_legendre(3, x)  # P₃(x)

# Chebyshev polynomials
chebyshev_vals = special.eval_chebyt(3, x)   # T₃(x)

# Hermite polynomials
hermite_vals = special.eval_hermite(3, x)    # H₃(x)

# Elliptic integrals
ellipk_vals = special.ellipk(0.5)  # Complete elliptic integral of first kind
ellipe_vals = special.ellipe(0.5)  # Complete elliptic integral of second kind

```

* * * * *

13\. Clustering (scipy.cluster)
-------------------------------

```
from scipy import cluster
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Generate sample data
np.random.seed(42)
data = np.random.randn(50, 2)
data[:25] += [2, 2]  # Create two clusters

# Hierarchical clustering
# Compute pairwise distances
distances = pdist(data, metric='euclidean')

# Linkage methods
linkage_single = cluster.hierarchy.linkage(distances, method='single')
linkage_complete = cluster.hierarchy.linkage(distances, method='complete')
linkage_average = cluster.hierarchy.linkage(distances, method='average')
linkage_ward = cluster.hierarchy.linkage(distances, method='ward')

# Form clusters
clusters = cluster.hierarchy.fcluster(linkage_ward, t=2, criterion='maxclust')
print(f"Cluster assignments: {clusters}")

# Dendrogram data (for plotting)
dendrogram_data = cluster.hierarchy.dendrogram(linkage_ward, no_plot=True)

# Distance threshold clustering
clusters_dist = cluster.hierarchy.fcluster(linkage_ward, t=1.5, criterion='distance')

# K-means clustering
from scipy.cluster.vq import kmeans2, whiten

# Normalize data (whitening)
whitened_data = whiten(data)

# Perform k-means
centroids, labels = kmeans2(whitened_data, 2)
print(f"K-means labels: {labels}")

# Vector quantization
from scipy.cluster.vq import vq
# Assign points to nearest centroids
codes, distances = vq(whitened_data, centroids)

```

* * * * *

14\. Input/Output (scipy.io)
----------------------------

```
from scipy import io
import numpy as np

# MATLAB files
data = {'array': np.arange(10), 'label': 'test'}
io.savemat('data.mat', data)  # Save to MATLAB format

# Load MATLAB file
loaded_data = io.loadmat('data.mat')
print(loaded_data['array'])

# WAV files (audio)
# Create a simple sine wave
sample_rate = 44100
duration = 2  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 440  # A4 note
audio_data = np.sin(2 * np.pi * frequency * t)

# Save as WAV file
io.wavfile.write('sine_wave.wav', sample_rate, audio_data.astype(np.float32))

# Read WAV file
sample_rate, audio_data = io.wavfile.read('sine_wave.wav')
print(f"Sample rate: {sample_rate}, Duration: {len(audio_data)/sample_rate:.2f}s")

# NetCDF files (scientific data format)
# Requires netCDF4 package
# from scipy.io import netcdf
# f = netcdf.netcdf_file('data.nc', 'w')
# f.createDimension('x', 10)
# v = f.createVariable('data', 'f', ('x',))
# v[:] = np.arange(10)
# f.close()

# Fortran binary files
fortran_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
with open('fortran_data.bin', 'wb') as f:
    io.FortranFile(f).write_record(fortran_data)

# Read Fortran binary
with open('fortran_data.bin', 'rb') as f:
    loaded_fortran = io.FortranFile(f).read_record(dtype=np.float32)

```

* * * * *

15\. Performance Tips and Advanced Usage
----------------------------------------

```
import numpy as np
from scipy import linalg, sparse
import time

# Use appropriate data types
# Float32 vs Float64
data_f64 = np.random.randn(1000, 1000)  # Default: float64
data_f32 = data_f64.astype(np.float32)  # Half the memory

# Sparse matrices for large, mostly zero matrices
dense_matrix = np.zeros((10000, 10000))
dense_matrix[np.random.randint(0, 10000, 1000), 
             np.random.randint(0, 10000, 1000)] = 1
sparse_matrix = sparse.csr_matrix(dense_matrix)
print(f"Dense size: {dense_matrix.nbytes} bytes")
print(f"Sparse size: {sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes} bytes")

# Vectorization vs loops
def slow_function(arr):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = np.sin(arr[i]) + np.cos(arr[i])
    return result

def fast_function(arr):
    return np.sin(arr) + np.cos(arr)

# In-place operations
large_array = np.random.randn(1000000)
# Memory efficient
large_array += 1  # In-place addition
np.sin(large_array, out=large_array)  # In-place sine

# Use appropriate solvers
# For positive definite matrices, use Cholesky
A = np.random.randn(100, 100)
A = A.T @ A  # Make positive definite
b = np.random.randn(100)

# Cholesky is faster for positive definite matrices
x_chol = linalg.solve(A, b, assume_a='pos')  # Assumes positive definite

# For symmetric matrices
x_sym = linalg.solve(A, b, assume_a='sym')  # Assumes symmetric

# Parallel processing with multiple threads
# SciPy automatically uses optimized BLAS/LAPACK libraries
# Control number of threads with environment variables:
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4

```

* * * * *

Summary
=======

-   **Linear algebra** with `scipy.linalg` for matrix operations, decompositions, and solving systems.

-   **Optimization** tools in `scipy.optimize` for finding minima, roots, and fitting curves.

-   **Integration** methods in `scipy.integrate` for numerical integration and ODE solving.

-   **Statistical functions** in `scipy.stats` for probability distributions and hypothesis testing.

-   **Signal processing** with `scipy.signal` for filtering, spectral analysis, and correlation.

-   **Sparse matrices** in `scipy.sparse` for memory-efficient large matrix operations.

-   Always consider **data types**, **sparse representations**, and **appropriate algorithms** for performance.
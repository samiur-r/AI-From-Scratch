# JAX Quick Reference

JAX is a high-performance numerical computing library that combines NumPy-style array operations with automatic differentiation, just-in-time compilation, and vectorization for accelerated computing on CPUs, GPUs, and TPUs.

## What JAX Does

JAX provides a unified interface for numerical computing that brings together:
- **Automatic Differentiation**: Compute gradients of complex functions automatically
- **JIT Compilation**: Compile Python functions to optimized machine code using XLA
- **Vectorization**: Apply functions over batches efficiently with `vmap`
- **Parallelization**: Distribute computations across multiple devices with `pmap`

The core concept is functional programming with pure functions that can be transformed by JAX's function transformations (`grad`, `jit`, `vmap`, `pmap`). JAX enforces functional purity, meaning no side effects or in-place mutations.

## When to Use JAX

### Problem Types
- **Deep learning research**: When you need flexible, composable neural network building blocks
- **Scientific computing**: Complex mathematical simulations requiring automatic differentiation
- **High-performance computing**: Computations that benefit from JIT compilation and parallelization
- **Gradient-based optimization**: Any problem requiring efficient gradient computation

### Data Characteristics
- **Large-scale computations**: JAX excels with large arrays and complex mathematical operations
- **Batch processing**: Natural support for vectorized operations over batches
- **Multi-device workloads**: Seamless scaling across GPUs and TPUs

### Business Contexts
- Research environments requiring rapid prototyping
- Production systems needing high-performance numerical computing
- Scientific applications with complex mathematical models
- ML/AI applications requiring custom gradient computations

### Comparison with Alternatives
- **vs PyTorch**: More functional, better JIT compilation, but steeper learning curve
- **vs TensorFlow**: More flexible for research, but less ecosystem support
- **vs NumPy**: Much faster with JIT, automatic differentiation, but functional programming required

## Strengths & Weaknesses

### Strengths
- **Performance**: JIT compilation provides near-C speed for numerical computations
- **Flexibility**: Functional transformations allow composable, reusable code
- **Hardware agnostic**: Same code runs efficiently on CPU, GPU, and TPU
- **Mathematical elegance**: Clean, mathematical approach to automatic differentiation
- **Debugging**: Excellent error messages and debugging tools
- **Ecosystem**: Growing ecosystem with Flax, Optax, and other high-quality libraries

### Weaknesses
- **Learning curve**: Functional programming paradigm can be challenging for beginners
- **Ecosystem maturity**: Smaller ecosystem compared to PyTorch/TensorFlow
- **Memory management**: Requires understanding of functional programming for efficient memory usage
- **Debugging complexity**: Harder to debug JIT-compiled code
- **Documentation**: Some advanced features lack comprehensive documentation
- **State management**: Handling stateful computations requires careful design patterns

## Important Hyperparameters

### JIT Compilation
- **Static arguments**: Use `static_argnums` for shape-determining arguments
- **Donation**: Use `donate_argnums` for memory-efficient in-place-style operations

### Automatic Differentiation
- **`argnums`**: Specify which arguments to differentiate with respect to
- **`has_aux`**: Handle auxiliary data alongside gradients

### Vectorization (vmap)
- **`in_axes`**: Specify which axes to vectorize over
- **`out_axes`**: Control output axis arrangement
- **`axis_size`**: Explicitly specify vectorization dimension

### Parallelization (pmap)
- **`axis_name`**: Name for collective operations
- **`devices`**: Specify target devices for parallel execution

## Key Assumptions

### Data Assumptions
- **Immutability**: Arrays are immutable; operations return new arrays
- **Pure functions**: Functions must be side-effect free for transformations
- **Static shapes**: JIT compilation works best with static array shapes
- **Numerical stability**: Operations assume standard floating-point arithmetic

### Programming Assumptions
- **Functional style**: Code should avoid side effects and mutations
- **JAX arrays**: Use `jax.numpy` arrays instead of regular NumPy arrays
- **Device placement**: Understanding of device memory and computation placement

### Violations
- **Side effects in JIT**: Will cause compilation errors or unexpected behavior
- **Dynamic shapes**: Can prevent JIT optimization or cause recompilation
- **Global state**: Can lead to incorrect gradient computations

## Performance Characteristics

### Time Complexity
- **First call**: JIT compilation adds overhead on first execution
- **Subsequent calls**: Near-optimal performance after compilation
- **Gradient computation**: Typically 2-4x slower than forward pass

### Space Complexity
- **Memory efficient**: Functional style can reduce memory usage with proper design
- **Gradient memory**: Automatic differentiation requires storing intermediate values
- **Device memory**: Explicit control over CPU/GPU memory placement

### Scalability
- **Multi-device**: Excellent scaling across multiple GPUs/TPUs
- **Large arrays**: Handles large arrays efficiently with proper chunking
- **Batch processing**: Linear scaling with batch size for vectorized operations

## Evaluation & Comparison

### Performance Metrics
- **Compilation time**: Measure JIT overhead for your use case
- **Execution time**: Compare with NumPy/PyTorch for similar operations
- **Memory usage**: Monitor device memory consumption
- **Throughput**: Measure operations per second for your workload

### Benchmarking Strategies
- **Warm-up runs**: Always exclude first JIT compilation from timing
- **Device synchronization**: Use `jax.block_until_ready()` for accurate timing
- **Memory profiling**: Use JAX's built-in memory profiling tools

### Comparison Baselines
- Pure NumPy implementations
- PyTorch equivalent operations
- TensorFlow/Keras implementations
- Specialized libraries (scipy, scikit-learn)

## Practical Usage Guidelines

### Implementation Tips
- Start simple without JIT, then add transformations incrementally
- Use `jax.debug.print()` for debugging JIT-compiled functions
- Prefer `jax.numpy` over `numpy` for all array operations
- Structure code as pure functions from the beginning

### Common Mistakes
- Mixing JAX and NumPy arrays inconsistently
- Using mutable state inside JIT-compiled functions
- Not understanding when recompilation occurs
- Ignoring device placement and memory management

### Debugging Strategies
- Use `jax.disable_jit()` to debug JIT compilation issues
- Check array shapes and dtypes carefully
- Use `jax.make_jaxpr()` to inspect compiled representations
- Enable double precision with `jax.config.update("jax_enable_x64", True)` for numerical debugging

### Production Considerations
- Implement proper error handling for device availability
- Monitor compilation times and cache hit rates
- Use checkpointing for long-running computations
- Consider memory usage patterns for large-scale deployments

## Complete Example

### Step 1: Data Preparation
```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

# What's happening: Creating sample data for a simple regression problem
# Why this step: JAX works with its own array type, so we convert NumPy arrays
key = jax.random.PRNGKey(42)
X = jax.random.normal(key, (1000, 5))  # 1000 samples, 5 features
true_weights = jnp.array([1.5, -2.0, 0.5, 3.0, -1.0])
y = X @ true_weights + 0.1 * jax.random.normal(key, (1000,))

print(f"Data shapes: X={X.shape}, y={y.shape}")
print(f"Data types: X={X.dtype}, y={y.dtype}")
```

### Step 2: Model Definition
```python
# What's happening: Defining a linear regression model as a pure function
# Why this approach: Pure functions are required for JAX transformations
def linear_model(params, X):
    """Linear regression: y = X @ weights + bias"""
    weights, bias = params
    return X @ weights + bias

def loss_fn(params, X, y):
    """Mean squared error loss function"""
    predictions = linear_model(params, X)
    return jnp.mean((predictions - y) ** 2)

# Initialize parameters
key, subkey = jax.random.split(key)
weights = jax.random.normal(subkey, (5,)) * 0.1
bias = 0.0
params = (weights, bias)

print(f"Initial loss: {loss_fn(params, X, y):.4f}")
```

### Step 3: Gradient Computation
```python
# What's happening: Creating gradient function using automatic differentiation
# Why this step: JAX automatically computes exact gradients efficiently
grad_fn = grad(loss_fn)

# Compute gradients
gradients = grad_fn(params, X, y)
print(f"Gradient shapes: weights={gradients[0].shape}, bias={gradients[1].shape}")
print(f"Weight gradients: {gradients[0]}")
```

### Step 4: JIT Compilation
```python
# What's happening: JIT-compiling functions for performance
# Why this step: JIT compilation provides significant speedup for repeated calls
@jit
def update_params(params, X, y, learning_rate):
    """Single gradient descent step"""
    grads = grad_fn(params, X, y)
    weights, bias = params
    grad_weights, grad_bias = grads

    new_weights = weights - learning_rate * grad_weights
    new_bias = bias - learning_rate * grad_bias

    return (new_weights, new_bias)

# Test compilation (first call will be slower)
learning_rate = 0.01
params = update_params(params, X, y, learning_rate)
print(f"Loss after one step: {loss_fn(params, X, y):.4f}")
```

### Step 5: Training Loop
```python
# What's happening: Training the model with JIT-compiled update function
# What the algorithm is learning: Finding weights that minimize prediction error
import time

start_time = time.time()
losses = []

for epoch in range(100):
    params = update_params(params, X, y, learning_rate)

    if epoch % 20 == 0:
        current_loss = loss_fn(params, X, y)
        losses.append(current_loss)
        print(f"Epoch {epoch}, Loss: {current_loss:.6f}")

training_time = time.time() - start_time
print(f"Training completed in {training_time:.3f} seconds")

# Final results
final_weights, final_bias = params
print(f"\nTrue weights: {true_weights}")
print(f"Learned weights: {final_weights}")
print(f"Learned bias: {final_bias:.4f}")
```

### Step 6: Vectorized Prediction
```python
# What's happening: Using vmap for efficient batch prediction
# How to use in practice: vmap automatically vectorizes over batch dimension

# Create vectorized prediction function
@jit
def predict_batch(params, X_batch):
    """Predict for multiple inputs simultaneously"""
    return vmap(lambda x: linear_model(params, x.reshape(1, -1)).squeeze())(X_batch)

# Generate test data
key, subkey = jax.random.split(key)
X_test = jax.random.normal(subkey, (100, 5))
y_test = X_test @ true_weights

# Make predictions
predictions = predict_batch(params, X_test)
test_loss = jnp.mean((predictions - y_test) ** 2)

print(f"Test MSE: {test_loss:.6f}")
print(f"Test R²: {1 - test_loss / jnp.var(y_test):.4f}")

# How to interpret results:
# - Low MSE indicates good fit
# - R² close to 1.0 shows the model explains most variance
# - Compare predicted vs true weights to assess parameter recovery
```

### Step 7: Advanced Features Demo
```python
# What's happening: Demonstrating advanced JAX features
# Why these features: Show JAX's unique capabilities for research and production

# 1. Hessian computation (second derivatives)
from jax import hessian

hessian_fn = hessian(loss_fn)
H = hessian_fn(params, X[:100], y[:100])  # Use subset for efficiency
print(f"Hessian shape: {H[0][0].shape}")  # Hessian w.r.t. weights

# 2. Multiple device parallelization (if available)
devices = jax.devices()
print(f"Available devices: {len(devices)} - {devices}")

if len(devices) > 1:
    # Parallel map across devices
    @jit
    def parallel_loss(params, X_shard, y_shard):
        return loss_fn(params, X_shard, y_shard)

    # This would split computation across available devices
    print("Multi-device computation is available")

# 3. Custom gradient transformation
def clip_gradients(grad_fn, clip_norm=1.0):
    """Clip gradients by norm"""
    def clipped_grad_fn(*args, **kwargs):
        grads = grad_fn(*args, **kwargs)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_leaves(grads)))
        clip_factor = jnp.minimum(1.0, clip_norm / grad_norm)
        return jax.tree_map(lambda g: g * clip_factor, grads)
    return clipped_grad_fn

clipped_grad_fn = clip_gradients(grad_fn)
clipped_grads = clipped_grad_fn(params, X, y)
print("Gradient clipping applied successfully")
```

## Summary

JAX is a powerful framework for high-performance numerical computing that excels in research environments and production systems requiring:

**Key Takeaways:**
- **Functional programming**: Embrace pure functions for maximum benefit from JAX transformations
- **JIT compilation**: Significant performance gains after initial compilation overhead
- **Automatic differentiation**: Exact gradients with minimal code changes
- **Hardware acceleration**: Seamless scaling from CPU to GPU to TPU
- **Composable transformations**: Mix and match `grad`, `jit`, `vmap`, and `pmap` as needed

**Quick Start Checklist:**
1. Install JAX with appropriate hardware support (`pip install jax[cpu/cuda/tpu]`)
2. Replace `numpy` imports with `jax.numpy`
3. Structure code as pure functions
4. Add `@jit` decorator for performance-critical functions
5. Use `grad()` for automatic differentiation
6. Apply `vmap()` for vectorization over batches

**When to Choose JAX:**
-  Research requiring flexible gradient computations
-  High-performance numerical computing
-  Custom neural network architectures
-  Scientific computing with complex mathematical models
- L Simple scripting tasks (use NumPy)
- L Beginners to functional programming (start with PyTorch)
- L Production systems requiring extensive ecosystem support
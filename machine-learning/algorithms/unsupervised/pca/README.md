# Principal Component Analysis (PCA) Quick Reference

A dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance in the data.

## What the Algorithm Does

Principal Component Analysis (PCA) is a statistical technique that reduces the dimensionality of datasets by finding new coordinate axes (principal components) that capture the maximum variance in the data. It transforms correlated variables into a set of linearly uncorrelated variables called principal components, ordered by the amount of variance they explain.

**Core Concept**: PCA finds the directions (eigenvectors) in which the data varies the most and projects the data onto these directions. The first principal component explains the most variance, the second explains the second most variance (orthogonal to the first), and so on.

**Algorithm Type**: Unsupervised dimensionality reduction and feature extraction

## When to Use It

### Problem Types
- **Dimensionality reduction**: When you have too many features and need to reduce computational complexity
- **Data visualization**: Reducing high-dimensional data to 2D or 3D for plotting
- **Noise reduction**: Filtering out less important variations in data
- **Feature extraction**: Creating new features that capture the essence of the original data
- **Data compression**: Reducing storage requirements while preserving information

### Data Characteristics
- **Continuous numerical data**: Works best with numerical features
- **Correlated features**: Most effective when original features are correlated
- **Standardized data**: Features should be on similar scales
- **Sufficient sample size**: Generally need more samples than features
- **Linear relationships**: Assumes linear relationships between variables

### Business Contexts
- **Image processing**: Face recognition, image compression
- **Finance**: Risk assessment, portfolio optimization
- **Genomics**: Gene expression analysis
- **Customer analytics**: Customer segmentation, recommendation systems
- **Quality control**: Manufacturing process monitoring

### Comparison with Alternatives
- **Choose PCA over t-SNE**: When you need interpretable components and linear transformation
- **Choose PCA over autoencoders**: When you want a simple, interpretable solution
- **Choose PCA over feature selection**: When you want to combine features rather than just select them
- **Choose LDA over PCA**: When you have labeled data and want supervised dimensionality reduction

## Strengths & Weaknesses

### Strengths
- **Variance preservation**: Maximally preserves variance in reduced dimensions
- **Interpretability**: Principal components have clear mathematical meaning
- **Computational efficiency**: Fast computation using eigenvalue decomposition
- **No hyperparameter tuning**: Deterministic algorithm with clear results
- **Orthogonal components**: Creates uncorrelated features
- **Reversible**: Can approximately reconstruct original data

### Weaknesses
- **Linear assumptions**: Cannot capture non-linear relationships
- **Scale sensitivity**: Sensitive to feature scaling
- **Interpretability loss**: Original feature meaning is lost in components
- **Outlier sensitivity**: Heavily influenced by outliers
- **All-or-nothing**: Must specify number of components beforehand
- **Information loss**: Lower dimensions mean some information is discarded

## Important Hyperparameters

### Critical Parameters

1. **n_components** (int, float, or 'mle')
   - **Range**: 1 to min(n_features, n_samples)
   - **Purpose**: Number of principal components to keep
   - **Tuning strategy**: Use explained variance ratio or cross-validation
   - **Default recommendation**: Start with components explaining 95% variance

2. **svd_solver** (string)
   - **Options**: 'auto', 'full', 'arpack', 'randomized'
   - **Purpose**: Algorithm for singular value decomposition
   - **Tuning strategy**: 'auto' for most cases, 'randomized' for large datasets
   - **Default recommendation**: 'auto'

3. **whiten** (boolean)
   - **Options**: True, False
   - **Purpose**: Whether to whiten the components (unit variance)
   - **Tuning strategy**: True for downstream algorithms sensitive to scale
   - **Default recommendation**: False for most cases

### Parameter Selection Strategies

```python
# Method 1: Explained variance threshold
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1

# Method 2: Elbow method
plt.plot(range(1, len(explained_var) + 1), explained_var)
# Look for "elbow" in the plot

# Method 3: Kaiser criterion (eigenvalues > 1)
n_components = np.sum(pca.explained_variance_ > 1)
```

## Key Assumptions

### Data Assumptions
- **Linearity**: Relationships between variables are linear
- **Large variances are important**: High variance directions contain important information
- **Orthogonality**: Principal components should be orthogonal (uncorrelated)
- **Continuous data**: Works best with continuous numerical variables
- **No missing values**: Complete data matrix required

### Statistical Assumptions
- **Standardization**: Variables should be standardized if on different scales
- **Sufficient variance**: Variables should have meaningful variance to capture
- **No perfect multicollinearity**: Avoid perfectly correlated features

### Violations and Consequences
- **Non-linear relationships**: Consider kernel PCA or autoencoders
- **Different scales**: Always standardize features first
- **Categorical data**: Use correspondence analysis or other techniques
- **Missing data**: Impute values or use specialized PCA variants

### Preprocessing Requirements
```python
# Essential preprocessing
from sklearn.preprocessing import StandardScaler

# Standardize features (crucial for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle missing values before PCA
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

## Performance Characteristics

### Time Complexity
- **Training**: O(min(n³, p³)) where n=samples, p=features
- **Transformation**: O(n × k) where k=n_components
- **Full SVD**: O(min(np², n²p))
- **Randomized SVD**: O(npk) for k components

### Space Complexity
- **Storage**: O(p × k) for transformation matrix
- **Memory during training**: O(np) for data matrix
- **Efficient variants**: Incremental PCA for large datasets

### Scalability
- **Feature scaling**: Becomes slow with very high dimensions (p > 10,000)
- **Sample scaling**: Generally scales well with number of samples
- **Incremental PCA**: Use for datasets that don't fit in memory
- **Sparse PCA**: Use for sparse, high-dimensional data

### Convergence Properties
- **Deterministic**: Always converges to the same solution
- **Global optimum**: Finds globally optimal solution for variance maximization
- **Numerical stability**: Generally stable but sensitive to numerical precision

## How to Evaluate & Compare Models

### Appropriate Metrics

1. **Explained Variance Ratio**
   ```python
   # Proportion of variance explained by each component
   explained_var_ratio = pca.explained_variance_ratio_
   cumulative_var = np.cumsum(explained_var_ratio)
   ```

2. **Reconstruction Error**
   ```python
   # How well can we reconstruct original data
   X_reconstructed = pca.inverse_transform(pca.transform(X))
   reconstruction_error = np.mean((X - X_reconstructed) ** 2)
   ```

3. **Silhouette Score** (if using for clustering)
   ```python
   from sklearn.metrics import silhouette_score
   silhouette_avg = silhouette_score(X_pca, cluster_labels)
   ```

### Cross-Validation Strategies

```python
# Method 1: Cross-validation on downstream task
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Test different numbers of components
scores = []
for n_comp in range(1, min(20, X.shape[1])):
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    clf = LogisticRegression()
    score = cross_val_score(clf, X_pca, y, cv=5).mean()
    scores.append(score)

# Method 2: Reconstruction error validation
def reconstruction_cv_score(X, n_components, cv=5):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    errors = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]

        pca = PCA(n_components=n_components)
        pca.fit(X_train)

        X_val_transformed = pca.transform(X_val)
        X_val_reconstructed = pca.inverse_transform(X_val_transformed)

        error = np.mean((X_val - X_val_reconstructed) ** 2)
        errors.append(error)

    return np.mean(errors)
```

### Baseline Comparisons
- **Random projection**: Simple baseline for dimensionality reduction
- **Feature selection**: Compare against selecting top k features
- **Original features**: Performance without dimensionality reduction
- **Other methods**: Compare with t-SNE, UMAP, or LDA when appropriate

### Statistical Significance
```python
# Bootstrap confidence intervals for explained variance
from scipy import stats

def bootstrap_explained_variance(X, n_components, n_bootstrap=1000):
    explained_vars = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(X_boot)

        explained_vars.append(pca.explained_variance_ratio_.sum())

    # Calculate confidence interval
    lower = np.percentile(explained_vars, 2.5)
    upper = np.percentile(explained_vars, 97.5)

    return np.mean(explained_vars), lower, upper
```

## Practical Usage Guidelines

### Implementation Tips

1. **Always standardize features**
   ```python
   # Critical: standardize before PCA
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   pca = PCA()
   X_pca = pca.fit_transform(X_scaled)
   ```

2. **Handle the full pipeline**
   ```python
   from sklearn.pipeline import Pipeline

   # Create pipeline for consistent preprocessing
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('pca', PCA(n_components=0.95)),
       ('classifier', LogisticRegression())
   ])
   ```

3. **Save components for interpretation**
   ```python
   # Access principal components for interpretation
   components_df = pd.DataFrame(
       pca.components_.T,
       columns=[f'PC{i+1}' for i in range(pca.n_components_)],
       index=feature_names
   )
   ```

### Common Mistakes

1. **Not standardizing features**: Leads to bias toward high-variance features
2. **Using too many components**: Defeats the purpose of dimensionality reduction
3. **Applying to categorical data**: PCA assumes continuous, linear relationships
4. **Ignoring explained variance**: Not checking how much information is retained
5. **Overfitting on component selection**: Choosing components based on test performance

### Debugging Strategies

1. **Check explained variance**
   ```python
   print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
   print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
   ```

2. **Visualize components**
   ```python
   # Plot first two components
   plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
   plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
   plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
   ```

3. **Check for numerical issues**
   ```python
   # Check for NaN or infinite values
   print(f"NaN in data: {np.isnan(X).any()}")
   print(f"Infinite values: {np.isinf(X).any()}")
   ```

### Production Considerations

1. **Model persistence**
   ```python
   import joblib

   # Save both scaler and PCA
   joblib.dump(scaler, 'scaler.pkl')
   joblib.dump(pca, 'pca_model.pkl')
   ```

2. **Memory optimization**
   ```python
   # Use incremental PCA for large datasets
   from sklearn.decomposition import IncrementalPCA

   ipca = IncrementalPCA(n_components=50, batch_size=1000)
   for batch in data_batches:
       ipca.partial_fit(batch)
   ```

3. **Monitoring drift**
   ```python
   # Monitor explained variance over time
   def check_pca_drift(new_data, original_pca, threshold=0.1):
       new_pca = PCA(n_components=original_pca.n_components_)
       new_pca.fit(new_data)

       variance_diff = abs(new_pca.explained_variance_ratio_.sum() -
                          original_pca.explained_variance_ratio_.sum())

       return variance_diff > threshold
   ```

## Complete Example

### Step 1: Data Preparation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# What's happening: Loading the breast cancer dataset for dimensionality reduction
# Why this step: This dataset has 30 features, making it perfect for demonstrating
# how PCA can reduce dimensionality while preserving predictive power
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"Original dataset shape: {X.shape}")
print(f"Features: {len(feature_names)} numerical features")
print(f"Target classes: {len(np.unique(y))} (malignant/benign)")

# Basic data exploration
df = pd.DataFrame(X, columns=feature_names)
print("\nFeature statistics:")
print(df.describe())
```

### Step 2: Preprocessing

```python
# What's happening: Standardizing features to have mean=0 and std=1
# Why this step: PCA is sensitive to feature scales. Without standardization,
# features with larger values would dominate the principal components
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Before scaling - Mean: {X.mean(axis=0)[:3]}")
print(f"Before scaling - Std: {X.std(axis=0)[:3]}")
print(f"After scaling - Mean: {X_scaled.mean(axis=0)[:3]}")
print(f"After scaling - Std: {X_scaled.std(axis=0)[:3]}")

# Split the data for evaluation
# What's happening: Creating train/test split to evaluate PCA effectiveness
# Why this step: We need to test how well dimensionality reduction preserves
# predictive performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
```

### Step 3: Model Configuration

```python
# What's happening: First, we'll analyze how many components we need
# Why these parameters: We start with all components to see the explained variance
# distribution, then choose an optimal number based on the results

# Fit PCA with all components to analyze variance
pca_full = PCA()
pca_full.fit(X_train)

# Calculate cumulative explained variance
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Find number of components for 99% variance
n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
print(f"Components needed for 99% variance: {n_components_99}")

# Visualize explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Explained Variance')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
plt.axhline(y=0.95, color='k', linestyle='--', label='95% variance')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# What's happening: Configuring PCA with optimal number of components
# Why these parameters: Using 95% variance threshold balances dimensionality
# reduction with information preservation
pca = PCA(n_components=n_components_95, random_state=42)
print(f"\nConfigured PCA with {n_components_95} components")
print(f"Dimensionality reduction: {X_train.shape[1]} → {n_components_95}")
```

### Step 4: Training

```python
# What's happening: Fitting PCA to find the principal components
# What the algorithm is learning: PCA is finding the directions (eigenvectors)
# in the feature space that capture maximum variance. It's learning a linear
# transformation that projects 30D data into a lower-dimensional space
pca.fit(X_train)

print("PCA Training Complete!")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Transform the training data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"\nTransformed training data shape: {X_train_pca.shape}")
print(f"Transformed test data shape: {X_test_pca.shape}")

# Analyze the principal components
print("\nTop contributing features for first 3 components:")
components_df = pd.DataFrame(
    pca.components_[:3].T,
    columns=['PC1', 'PC2', 'PC3'],
    index=feature_names
)

for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
    print(f"\n{pc} (explains {pca.explained_variance_ratio_[i]:.1%} variance):")
    top_features = components_df[pc].abs().sort_values(ascending=False)[:5]
    for feature, loading in top_features.items():
        print(f"  {feature}: {loading:.3f}")
```

### Step 5: Evaluation

```python
# What's happening: Comparing model performance before and after PCA
# How to interpret results: We're measuring if dimensionality reduction
# significantly hurts predictive performance. Small performance drops indicate
# that PCA successfully preserved the important information

# Train classifier on original data
clf_original = LogisticRegression(random_state=42, max_iter=1000)
clf_original.fit(X_train, y_train)
y_pred_original = clf_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

# Train classifier on PCA-transformed data
clf_pca = LogisticRegression(random_state=42, max_iter=1000)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

print("Performance Comparison:")
print(f"Original features ({X_train.shape[1]}D): {accuracy_original:.3f} accuracy")
print(f"PCA features ({X_train_pca.shape[1]}D): {accuracy_pca:.3f} accuracy")
print(f"Performance difference: {accuracy_original - accuracy_pca:.3f}")
print(f"Dimensionality reduction: {(1 - X_train_pca.shape[1]/X_train.shape[1]):.1%}")

# Detailed classification report for PCA model
print(f"\nClassification Report (PCA Model):")
print(classification_report(y_test, y_pred_pca, target_names=data.target_names))

# Calculate reconstruction error
X_test_reconstructed = pca.inverse_transform(X_test_pca)
reconstruction_error = np.mean((X_test - X_test_reconstructed) ** 2)
print(f"\nReconstruction error: {reconstruction_error:.6f}")

# Visualize the transformed data (first 2 components)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,
                     cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Data Projected onto First Two Principal Components')
plt.grid(True, alpha=0.3)
plt.show()
```

### Step 6: Prediction

```python
# What's happening: Demonstrating how to use PCA for new predictions
# How to use in practice: Always apply the same preprocessing pipeline
# (scaling + PCA transformation) to new data before making predictions

# Simulate new data (first 5 test samples)
new_data = X_test[:5]  # This would be your new, unseen data
print("Predicting on new data...")
print(f"New data shape: {new_data.shape}")

# Step 1: Apply the same scaling
new_data_scaled = scaler.transform(new_data)
print("✓ Applied standardization")

# Step 2: Apply PCA transformation
new_data_pca = pca.transform(new_data_scaled)
print(f"✓ Applied PCA transformation: {new_data.shape} → {new_data_pca.shape}")

# Step 3: Make predictions
predictions = clf_pca.predict(new_data_pca)
prediction_probs = clf_pca.predict_proba(new_data_pca)

# Display results
print("\nPredictions:")
for i, (pred, prob) in enumerate(zip(predictions, prediction_probs)):
    class_name = data.target_names[pred]
    confidence = prob.max()
    print(f"Sample {i+1}: {class_name} (confidence: {confidence:.3f})")

# Demonstrate the complete pipeline for production use
print("\nProduction Pipeline Example:")
```

```python
# Production-ready pipeline
from sklearn.pipeline import Pipeline

# What's happening: Creating a complete pipeline that handles all preprocessing
# How to use in practice: This pipeline can be saved and deployed for consistent
# preprocessing and prediction on new data

# Create the complete pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components_95, random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Fit the entire pipeline
pipeline.fit(X_train, y_train)

# Test the pipeline
pipeline_accuracy = pipeline.score(X_test, y_test)
print(f"Pipeline accuracy: {pipeline_accuracy:.3f}")

# Save the pipeline for production use
import joblib
joblib.dump(pipeline, 'breast_cancer_pca_pipeline.pkl')
print("✓ Pipeline saved for production use")

# Load and use the pipeline (production simulation)
loaded_pipeline = joblib.load('breast_cancer_pca_pipeline.pkl')
new_predictions = loaded_pipeline.predict(X_test[:3])
print(f"✓ Loaded pipeline predictions: {new_predictions}")

# Feature importance analysis
print(f"\nFeature Importance Analysis:")
print(f"Original features: {len(feature_names)}")
print(f"Principal components: {pca.n_components_}")
print(f"Variance preserved: {pca.explained_variance_ratio_.sum():.1%}")

# Show which original features contribute most to top PCs
for i in range(min(3, pca.n_components_)):
    print(f"\nPC{i+1} ({pca.explained_variance_ratio_[i]:.1%} variance):")
    component_contributions = pd.Series(
        np.abs(pca.components_[i]),
        index=feature_names
    ).sort_values(ascending=False)[:5]

    for feature, contribution in component_contributions.items():
        print(f"  {feature}: {contribution:.3f}")
```

## Summary

### Key Takeaways

1. **Purpose**: PCA reduces dimensionality while preserving maximum variance in the data
2. **When to use**: High-dimensional data with correlated features, need for visualization or noise reduction
3. **Critical step**: Always standardize features before applying PCA
4. **Component selection**: Use explained variance ratio to choose optimal number of components
5. **Evaluation**: Compare downstream task performance and reconstruction error
6. **Production**: Use pipelines for consistent preprocessing and deployment

### Quick Reference Points

- **Optimal components**: Usually 95-99% explained variance
- **Preprocessing**: StandardScaler is essential
- **Interpretation**: Principal components are linear combinations of original features
- **Limitations**: Only captures linear relationships, sensitive to outliers
- **Alternatives**: Consider t-SNE for visualization, autoencoders for non-linear reduction
- **Memory**: Use IncrementalPCA for large datasets that don't fit in memory

### Decision Checklist

- ✅ Use PCA when: High dimensions, correlated features, need interpretable reduction
- ✅ Always: Standardize features, check explained variance, validate on downstream tasks
- ⚠️ Be careful: With categorical data, small sample sizes, non-linear relationships
- ❌ Don't use: When all features are important, with purely categorical data, for clustering discovery
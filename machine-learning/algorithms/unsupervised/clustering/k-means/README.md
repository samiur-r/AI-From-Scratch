# K-Means Clustering Quick Reference

A centroid-based clustering algorithm that partitions data into k clusters by iteratively assigning points to the nearest cluster center and updating centroids until convergence.

## What the Algorithm Does

K-Means clustering divides a dataset into k distinct, non-overlapping clusters by:

1. **Initialization**: Randomly placing k cluster centroids in the feature space
2. **Assignment**: Assigning each data point to the nearest centroid (using Euclidean distance)
3. **Update**: Recalculating centroids as the mean of all assigned points
4. **Iteration**: Repeating assignment and update steps until centroids stabilize

The algorithm minimizes the Within-Cluster Sum of Squares (WCSS): $\sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$

Where $\mu_i$ is the centroid of cluster $C_i$ and $||x - \mu_i||^2$ represents the squared Euclidean distance.

## When to Use It

### Problem Types
- **Customer segmentation**: Grouping customers by purchasing behavior
- **Market research**: Identifying distinct consumer segments
- **Image compression**: Reducing color palettes by clustering similar colors
- **Data preprocessing**: Creating features or reducing dimensionality before supervised learning
- **Anomaly detection**: Identifying outliers as points far from cluster centers

### Data Characteristics
- **Continuous/numerical features**: Works best with quantitative variables
- **Moderate dimensionality**: Effective with 2-20 features (curse of dimensionality affects higher dimensions)
- **Spherical clusters**: Assumes clusters are roughly circular/spherical in shape
- **Similar cluster sizes**: Performs optimally when clusters have comparable sizes
- **Well-separated clusters**: Clear boundaries between groups improve results

### Comparison with Alternatives
- **Choose K-Means over Hierarchical Clustering**: When you know approximate number of clusters and need faster computation
- **Choose DBSCAN over K-Means**: When clusters have irregular shapes or varying densities
- **Choose Gaussian Mixture Models over K-Means**: When you need probabilistic cluster assignments or overlapping clusters

## Strengths & Weaknesses

### Strengths
- **Computational efficiency**: O(n*k*i*d) complexity, scales well with large datasets
- **Simplicity**: Easy to understand, implement, and interpret
- **Guaranteed convergence**: Algorithm always converges to a local minimum
- **Memory efficient**: Requires minimal storage during execution
- **Versatile**: Applicable across many domains and data types

### Weaknesses
- **Requires pre-specified k**: Must know or estimate number of clusters beforehand
- **Sensitive to initialization**: Different starting centroids can yield different results
- **Assumes spherical clusters**: Struggles with elongated, irregular, or nested cluster shapes
- **Sensitive to outliers**: Extreme values can significantly skew centroid positions
- **Equal cluster size bias**: Tends to create clusters of similar sizes regardless of natural groupings
- **Scale sensitivity**: Features with larger ranges dominate distance calculations

## Important Hyperparameters

### Critical Parameters

**n_clusters (k)**
- **Purpose**: Number of clusters to form
- **Range**: 2 to √(n/2) as a rough upper bound
- **Tuning strategy**: Use elbow method, silhouette analysis, or domain knowledge
- **Default recommendation**: Start with 3-5 for initial exploration

**init**
- **Purpose**: Centroid initialization method
- **Options**: 'k-means++' (smart initialization), 'random'
- **Recommendation**: Use 'k-means++' for better convergence
- **Impact**: Significantly affects final clustering quality

**max_iter**
- **Purpose**: Maximum number of iterations
- **Range**: 100-1000 typically sufficient
- **Default recommendation**: 300 (scikit-learn default)
- **Tuning**: Increase if algorithm doesn't converge

**tol**
- **Purpose**: Tolerance for convergence (change in centroids)
- **Range**: 1e-6 to 1e-3
- **Default recommendation**: 1e-4
- **Impact**: Lower values = more precise but slower convergence

**n_init**
- **Purpose**: Number of random initializations
- **Range**: 1-20
- **Default recommendation**: 10
- **Trade-off**: Higher values = better results but longer computation

## Key Assumptions

### Data Assumptions
- **Euclidean distance meaningfulness**: Features should be on similar scales
- **Cluster convexity**: Assumes clusters are convex (roughly circular/spherical)
- **Isotropic clusters**: Clusters have similar variance in all directions
- **Linear separability**: Clusters can be separated by linear boundaries from centroids

### Statistical Assumptions
- **Gaussian-like distributions**: Works best when clusters follow roughly normal distributions
- **Independence**: Data points are independent observations
- **Homoscedasticity**: Clusters have similar within-cluster variance

### Violations and Consequences
- **Non-spherical clusters**: Algorithm may split natural clusters or merge separate ones
- **Different cluster densities**: Dense clusters may be oversplit, sparse ones undersplit
- **Unscaled features**: Dimensions with larger ranges dominate clustering decisions
- **Categorical data**: Algorithm cannot handle non-numerical features directly

### Preprocessing Requirements
- **Feature scaling**: Standardize or normalize features to similar ranges
- **Outlier handling**: Remove or transform extreme values
- **Dimensionality reduction**: Consider PCA for high-dimensional data
- **Encoding**: Convert categorical variables to numerical (if applicable)

## Performance Characteristics

### Time Complexity
- **Training**: O(n × k × i × d) where n=samples, k=clusters, i=iterations, d=dimensions
- **Prediction**: O(k × d) for assigning new points
- **Typical iterations**: 10-50 for convergence

### Space Complexity
- **Memory**: O(n × d + k × d) for storing data and centroids
- **Storage**: Minimal - only centroids need to be saved

### Scalability
- **Sample size**: Scales linearly with number of data points
- **Dimensionality**: Performance degrades with high dimensions (>20-50)
- **Number of clusters**: Linear scaling with k, but quality may decrease

### Convergence Properties
- **Guaranteed convergence**: Always reaches a local minimum
- **Convergence speed**: Usually fast (10-50 iterations)
- **Global optimum**: No guarantee of finding global optimum
- **Stability**: Results can vary between runs due to random initialization

## How to Evaluate & Compare Models

### Appropriate Metrics

**Inertia/WCSS (Within-Cluster Sum of Squares)**
- **Formula**: $\sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$
- **Interpretation**: Lower values indicate tighter clusters
- **Limitation**: Always decreases as k increases

**Silhouette Score**
- **Range**: -1 to 1
- **Interpretation**: Higher values indicate better-defined clusters
- **Optimal**: Values > 0.5 considered good clustering

**Calinski-Harabasz Index**
- **Interpretation**: Higher values indicate better clustering
- **Advantage**: Considers both within-cluster and between-cluster distances

**Davies-Bouldin Index**
- **Range**: 0 to ∞
- **Interpretation**: Lower values indicate better clustering
- **Consideration**: Less influenced by number of clusters

### Cross-Validation Strategies

**Stability Analysis**
- Run K-Means multiple times with different random seeds
- Measure consistency of cluster assignments
- Use Adjusted Rand Index (ARI) to compare clustering solutions

**Subsampling Validation**
- Train on random subsets of data
- Evaluate consistency across different samples
- Useful for large datasets

### Baseline Comparisons
- **Random clustering**: Assign points randomly to k clusters
- **Single cluster**: All points in one cluster
- **Hierarchical clustering**: Compare with agglomerative clustering
- **Different k values**: Compare across multiple cluster numbers

### Optimal k Selection

**Elbow Method**
```python
# Plot WCSS vs number of clusters
# Look for "elbow" where improvement rate decreases
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

**Silhouette Analysis**
```python
# Find k that maximizes average silhouette score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
```

## Practical Usage Guidelines

### Implementation Tips
- **Always scale features** before clustering to ensure equal weight
- **Use multiple random initializations** (n_init=10+) for stable results
- **Validate cluster number** using multiple methods (elbow + silhouette)
- **Visualize results** in 2D/3D when possible to verify cluster quality
- **Check convergence** by monitoring inertia changes across iterations

### Common Mistakes
- **Forgetting to scale features**: Leads to dominance by high-variance features
- **Not validating k**: Choosing arbitrary number of clusters
- **Ignoring cluster interpretation**: Failing to validate clusters make business sense
- **Using with categorical data**: K-Means requires numerical features
- **Assuming global optimum**: Not accounting for local minima from different initializations

### Debugging Strategies
- **Empty clusters**: Increase n_init or try different initialization
- **Poor convergence**: Increase max_iter or adjust tolerance
- **Unstable results**: Use more random initializations and feature scaling
- **Unexpected clusters**: Visualize data and check for outliers
- **Low silhouette scores**: Try different k values or consider alternative algorithms

### Production Considerations
- **Model persistence**: Save centroids for consistent predictions
- **Drift monitoring**: Track silhouette scores on new data
- **Retraining triggers**: Retrain when cluster quality degrades
- **Scalability**: Consider Mini-Batch K-Means for large datasets
- **Real-time prediction**: Pre-compute centroids for fast assignment

## Complete Example

### Step 1: Data Preparation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# What's happening: Creating synthetic data with 4 natural clusters
# Why this step: Demonstrates K-Means on data with known ground truth
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60,
                       random_state=42, n_features=2)

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
print(f"Dataset shape: {df.shape}")
print(f"Feature ranges:\n{df.describe()}")
```

### Step 2: Preprocessing
```python
# What's happening: Standardizing features to have mean=0 and std=1
# Why this step: Ensures both features contribute equally to distance calculations
# Without scaling, features with larger ranges would dominate clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verify scaling worked
print(f"Original feature means: {X.mean(axis=0)}")
print(f"Scaled feature means: {X_scaled.mean(axis=0)}")
print(f"Original feature stds: {X.std(axis=0)}")
print(f"Scaled feature stds: {X_scaled.std(axis=0)}")
```

### Step 3: Model Configuration
```python
# What's happening: Finding optimal number of clusters using elbow method
# Why these parameters: k-means++ initialization for better convergence,
# multiple random starts for stability, sufficient iterations for convergence

# Elbow method to find optimal k
wcss = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    # Why k-means++: Smart initialization reduces chances of poor local minima
    # Why n_init=10: Multiple random starts ensure stable, reproducible results
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,
                    max_iter=300, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

    # Calculate silhouette score for this k
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

    print(f"k={k}: WCSS={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.3f}")

# Find optimal k (highest silhouette score)
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k based on silhouette score: {optimal_k}")
```

### Step 4: Training
```python
# What's happening: Training K-Means with optimal number of clusters
# What the algorithm is learning: Finding 4 cluster centroids that minimize
# the sum of squared distances from each point to its nearest centroid

final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++',
                     n_init=10, max_iter=300, random_state=42)
cluster_labels = final_kmeans.fit_predict(X_scaled)

# Get final centroids (in original scale for interpretation)
centroids_scaled = final_kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

print(f"Algorithm converged in {final_kmeans.n_iter_} iterations")
print(f"Final inertia (WCSS): {final_kmeans.inertia_:.2f}")
print(f"\nCentroid positions (original scale):")
for i, centroid in enumerate(centroids_original):
    print(f"Cluster {i}: Feature_1={centroid[0]:.2f}, Feature_2={centroid[1]:.2f}")
```

### Step 5: Evaluation
```python
# What's happening: Evaluating clustering quality using multiple metrics
# How to interpret results:
# - Silhouette score > 0.5 indicates good clustering
# - ARI close to 1.0 means perfect agreement with true clusters
# - Higher ARI indicates better recovery of true cluster structure

silhouette_avg = silhouette_score(X_scaled, cluster_labels)
ari_score = adjusted_rand_score(y_true, cluster_labels)

print(f"Clustering Evaluation Metrics:")
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"  Interpretation: {silhouette_avg:.3f} > 0.5 indicates good clustering")

print(f"Adjusted Rand Index: {ari_score:.3f}")
print(f"  Interpretation: {ari_score:.3f} closeness to 1.0 shows good recovery of true clusters")

# Cluster size distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
print(f"\nCluster size distribution:")
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"Cluster {cluster_id}: {count} points ({percentage:.1f}%)")
```

### Step 6: Prediction
```python
# What's happening: Using trained model to predict cluster for new data points
# How to use in practice: Save the fitted model and scaler for consistent predictions

# Simulate new data points
new_points = np.array([[2.0, 3.0], [-1.0, -2.0], [0.0, 0.0]])
print("Predicting clusters for new data points:")

# IMPORTANT: Must apply same scaling transformation used during training
new_points_scaled = scaler.transform(new_points)
new_predictions = final_kmeans.predict(new_points_scaled)

for i, (point, prediction) in enumerate(zip(new_points, new_predictions)):
    print(f"Point {i+1}: {point} → Cluster {prediction}")

    # Calculate distance to assigned centroid for confidence assessment
    assigned_centroid = centroids_scaled[prediction]
    distance = np.linalg.norm(new_points_scaled[i] - assigned_centroid)
    print(f"  Distance to centroid: {distance:.3f}")

# Production usage pattern
print(f"\nFor production deployment:")
print(f"1. Save the fitted scaler and kmeans model")
print(f"2. For new predictions: scaler.transform(new_data) → model.predict()")
print(f"3. Monitor prediction distances to detect potential data drift")

# Visualization for understanding
plt.figure(figsize=(12, 4))

# Plot 1: Original data with true clusters
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot 2: K-Means clusters
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title(f'K-Means Clusters (k={optimal_k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot 3: Elbow curve
plt.subplot(1, 3, 3)
plt.plot(k_range, wcss, 'bo-', label='WCSS')
plt.axvline(x=optimal_k, color='red', linestyle='--',
           label=f'Optimal k={optimal_k}')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.legend()

plt.tight_layout()
plt.show()
```

## Summary

**Key Takeaways:**
- K-Means is optimal for spherical, well-separated clusters of similar sizes
- Always scale features and use multiple initializations for stable results
- Determine k using elbow method combined with silhouette analysis
- Evaluate results using multiple metrics and visual inspection
- Consider alternatives (DBSCAN, GMM) for non-spherical or overlapping clusters

**Quick Reference:**
- **Best for**: Customer segmentation, image compression, preprocessing
- **Avoid when**: Clusters are non-spherical, varying densities, or unknown k
- **Key parameters**: n_clusters, init='k-means++', n_init=10
- **Evaluation**: Silhouette score > 0.5, elbow method for k selection
- **Preprocessing**: Feature scaling is essential
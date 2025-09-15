# K-Nearest Neighbors (KNN) Quick Reference

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that makes predictions based on the k closest training examples in the feature space. It's a non-parametric method that doesn't make assumptions about the underlying data distribution and can be used for both classification and regression tasks.

## What the Algorithm Does

KNN stores all training data and makes predictions by finding the k nearest neighbors to a query point, then either taking the majority vote (classification) or averaging the values (regression). The algorithm is "lazy" because it defers computation until prediction time rather than learning a model during training.

**Core concept**: "Similar instances have similar outcomes" - the algorithm assumes that nearby points in feature space should have similar labels or values.

**Algorithm type**: Both classification and regression (lazy learning, instance-based)

The mathematical foundation:
- **Distance calculation**: $d(x_i, x_j) = \sqrt{\sum_{f=1}^{p}(x_{if} - x_{jf})^2}$ (Euclidean distance)
- **Classification**: $\hat{y} = \text{mode}(\{y_i : x_i \in N_k(x)\})$ (majority vote)
- **Regression**: $\hat{y} = \frac{1}{k}\sum_{x_i \in N_k(x)} y_i$ (average of neighbors)
- **Weighted prediction**: $\hat{y} = \frac{\sum_{x_i \in N_k(x)} w_i \cdot y_i}{\sum_{x_i \in N_k(x)} w_i}$ where $w_i = \frac{1}{d(x, x_i)}$

## When to Use It

### Problem Types
- **Classification with irregular boundaries**: When decision boundaries are complex and non-linear
- **Regression with local patterns**: When the relationship varies across different regions
- **Recommendation systems**: Finding similar users or items based on features
- **Anomaly detection**: Identifying outliers based on distance to nearest neighbors

### Data Characteristics
- **Small to medium datasets**: Works best with hundreds to thousands of samples
- **Low to medium dimensionality**: Performance degrades with high-dimensional data (curse of dimensionality)
- **Continuous or ordinal features**: Works well with numerical data that has meaningful distances
- **Local patterns**: When similar inputs tend to have similar outputs

### Business Contexts
- **E-commerce**: Product recommendation based on user similarity
- **Real estate**: Property valuation based on similar properties
- **Healthcare**: Diagnosis based on similar patient profiles
- **Image recognition**: Classification based on visual similarity
- **Quality control**: Defect detection based on similar patterns

### Comparison with Alternatives
- **Choose over Linear Models**: When relationships are highly non-linear and local
- **Choose over Decision Trees**: When you need smooth decision boundaries
- **Choose over Neural Networks**: When you have limited data and need interpretability
- **Choose over SVM**: When you need simple implementation and probabilistic outputs

## Strengths & Weaknesses

### Strengths
- **Simple to understand and implement**: Intuitive concept, easy to explain
- **No assumptions about data**: Non-parametric, works with any data distribution
- **Handles multi-class naturally**: No need for one-vs-rest strategies
- **Local adaptation**: Can capture local patterns and complex boundaries
- **Online learning**: Can easily add new data without retraining
- **Probabilistic output**: Can provide confidence estimates for predictions
- **Interpretable**: Easy to see which neighbors influenced a prediction

### Weaknesses
- **Computationally expensive**: O(n) prediction time, slow for large datasets
- **Memory intensive**: Must store all training data
- **Sensitive to irrelevant features**: All features contribute to distance equally
- **Curse of dimensionality**: Performance degrades rapidly with many features
- **Sensitive to feature scaling**: Features with larger scales dominate distance
- **Poor with sparse data**: Struggles when data points are far apart
- **Sensitive to local noise**: Outliers can significantly affect predictions

## Important Hyperparameters

### Critical Parameters

**Number of Neighbors (k)**
- **Range**: 1 to √n (where n is training set size)
- **Odd values recommended**: Prevents ties in binary classification
- **Lower values**: More sensitive to noise, complex boundaries, may overfit
- **Higher values**: Smoother boundaries, less sensitive to noise, may underfit
- **Tuning strategy**: Use cross-validation, try [1, 3, 5, 7, 9, 11, 15, 21]

**Distance Metric**
- **Euclidean**: Most common, good for continuous features
- **Manhattan**: Better for high-dimensional or categorical data
- **Minkowski**: Generalization (p=1 Manhattan, p=2 Euclidean)
- **Hamming**: For categorical features
- **Custom metrics**: Domain-specific distance functions

**Weighting Scheme**
- **Uniform**: All neighbors have equal weight (default)
- **Distance**: Closer neighbors have more influence, weights = 1/distance
- **Custom**: Domain-specific weighting functions

**Algorithm Optimization**
- **Ball Tree**: Good for high-dimensional data
- **KD Tree**: Good for low-dimensional data (< 20 features)
- **Brute Force**: Exhaustive search, guaranteed accuracy

### Parameter Tuning Examples
```python
# Grid search for optimal parameters
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
```

## Key Assumptions

### Data Assumptions
- **Meaningful distance**: Features should have interpretable distance relationships
- **Local similarity**: Nearby points should have similar target values
- **Adequate density**: Sufficient training points in all regions of feature space
- **Consistent feature importance**: All features should be relevant for similarity

### Statistical Assumptions
- **Smoothness**: Target function should be locally smooth
- **Stationarity**: Relationship between features and target shouldn't change drastically
- **Representative sampling**: Training data should cover the full feature space

### Violations and Consequences
- **High dimensionality**: All points become equidistant (curse of dimensionality)
- **Irrelevant features**: Noise features can dominate distance calculations
- **Different feature scales**: Large-scale features overwhelm small-scale ones
- **Sparse regions**: Poor predictions in areas with few training examples

### Preprocessing Requirements
- **Feature scaling**: Mandatory - use StandardScaler or MinMaxScaler
- **Feature selection**: Remove irrelevant features to improve distance calculations
- **Handle missing values**: Impute or use distance metrics that handle missing data
- **Outlier treatment**: Consider removing or transforming extreme outliers

## Performance Characteristics

### Time Complexity
- **Training**: O(1) - just stores the data
- **Prediction**: O(n × p) for each query (n=samples, p=features)
- **With tree structures**: O(log n × p) average case for prediction

### Space Complexity
- **Memory usage**: O(n × p) - stores all training data
- **High memory requirement**: Not suitable for very large datasets
- **Tree structures**: Additional O(n) space for indexing

### Convergence Properties
- **No training phase**: Instant "training" (just data storage)
- **Deterministic**: Same results for same data and parameters
- **Non-parametric**: No model parameters to learn

### Scalability Characteristics
- **Sample size**: Prediction time increases linearly with training set size
- **Feature size**: Severely affected by dimensionality curse
- **Parallel processing**: Predictions can be parallelized
- **Approximate methods**: LSH, random sampling for large-scale applications

## How to Evaluate & Compare Models

### Appropriate Metrics

**For Classification**
- **Accuracy**: Overall correctness for balanced datasets
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Ranking ability (using predict_proba)
- **Confusion Matrix**: Detailed breakdown of prediction errors

**For Regression**
- **RMSE**: Root Mean Square Error (penalizes large errors)
- **MAE**: Mean Absolute Error (robust to outliers)
- **R²**: Proportion of variance explained
- **MAPE**: Mean Absolute Percentage Error (scale-independent)

**Distance-Specific Metrics**
- **Average neighbor distance**: Measure data sparsity
- **Silhouette score**: Quality of neighborhood structure
- **Local density**: Distribution of neighbors

### Cross-Validation Strategies
- **K-Fold**: Standard approach, be careful with spatial/temporal data
- **Stratified K-Fold**: For classification to maintain class balance
- **Leave-One-Out**: Good for small datasets but computationally expensive
- **Time Series Split**: For temporal data to avoid look-ahead bias

**Recommended approach**:
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Baseline Comparisons
- **1-NN**: Simplest version for comparison
- **Random classifier**: For classification tasks
- **Mean prediction**: For regression tasks
- **Linear models**: Compare with logistic regression or linear regression

### Statistical Significance
- **Paired t-test**: Compare performance across CV folds
- **McNemar's test**: For classification comparison between models
- **Wilcoxon signed-rank test**: Non-parametric alternative to paired t-test

## Practical Usage Guidelines

### Implementation Tips
- **Always scale features**: Use StandardScaler or MinMaxScaler before training
- **Start with k=√n**: Good rule of thumb for initial k value
- **Use odd k values**: Prevents ties in binary classification
- **Consider distance weighting**: Often improves performance over uniform weighting
- **Feature selection**: Remove irrelevant features to improve distance quality
- **Cross-validate k**: Don't rely on default values

### Common Mistakes
- **Forgetting feature scaling**: Most critical preprocessing step for KNN
- **Using too many features**: Curse of dimensionality degrades performance
- **Ignoring data sparsity**: Poor performance in regions with few neighbors
- **Not handling ties**: Can cause inconsistent predictions
- **Using even k for binary classification**: Can lead to tie situations
- **Overlooking computational cost**: Can be prohibitive for large datasets

### Debugging Strategies
- **Visualize neighbors**: Plot query points and their k nearest neighbors
- **Check feature scaling**: Ensure all features have similar ranges
- **Analyze distance distributions**: Look for features dominating distances
- **Examine prediction confidence**: Low confidence may indicate sparse regions
- **Profile performance**: Identify computational bottlenecks
- **Validate distance metric**: Try different metrics for your data type

### Production Considerations
- **Computational optimization**: Use approximate methods (LSH, sampling) for large datasets
- **Memory management**: Consider data compression or feature reduction
- **Real-time predictions**: Pre-compute neighbor structures when possible
- **Model updates**: Efficiently add new data points to existing structure
- **Monitoring**: Track prediction times and neighbor distances over time
- **Caching**: Store frequent query results to improve response times

## Complete Example with Step-by-Step Explanation

Let's build a KNN classifier to predict house prices based on property features, demonstrating both regression and classification capabilities.

### Step 1: Data Preparation
```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# What's happening: Loading California housing dataset for regression and creating classification version
# Why this step: We need realistic data to demonstrate KNN's effectiveness on both regression and classification

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Create DataFrame for easier manipulation
feature_names = housing.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['price'] = y

print("Dataset Overview:")
print(f"Dataset shape: {df.shape}")
print(f"Features: {list(feature_names)}")
print(f"Target (price) range: ${y.min():.1f}k - ${y.max():.1f}k")
print(f"Missing values: {df.isnull().sum().sum()}")

# Display basic statistics
print("\nFeature Statistics:")
print(df.describe())

# Create classification version: convert prices to categories
price_categories = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
y_classification = price_categories.codes

print(f"\nPrice Categories for Classification:")
for i, category in enumerate(['Low', 'Medium', 'High']):
    count = np.sum(y_classification == i)
    price_range = pd.cut(y, bins=3).categories[i]
    print(f"  {category}: {count} houses ({price_range})")
```

### Step 2: Preprocessing
```python
# What's happening: Preparing data for KNN by scaling features and splitting datasets
# Why this step: KNN requires feature scaling and proper train/test separation for accurate evaluation

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split data for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

print("Data Splitting Results:")
print(f"Regression - Training: {X_train_reg.shape[0]}, Testing: {X_test_reg.shape[0]}")
print(f"Classification - Training: {X_train_clf.shape[0]}, Testing: {X_test_clf.shape[0]}")

# Feature scaling - CRITICAL for KNN
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Use same scaler for classification
X_train_clf_scaled = scaler.transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# What's happening: Converting features to have mean=0, std=1
# Why this step: Without scaling, features with larger ranges (like population) would dominate distance

print("\nFeature Scaling Results:")
print("Before scaling:")
print(f"  Feature ranges: {X_train_reg.min(axis=0)} to {X_train_reg.max(axis=0)}")
print("After scaling:")
print(f"  Feature means: {X_train_reg_scaled.mean(axis=0)}")
print(f"  Feature stds: {X_train_reg_scaled.std(axis=0)}")

# Analyze feature importance by correlation
feature_correlation = pd.DataFrame(X_train_reg, columns=feature_names).corrwith(pd.Series(y_train_reg))
print(f"\nFeature Correlation with Price:")
for feature, corr in feature_correlation.sort_values(ascending=False).items():
    print(f"  {feature}: {corr:.3f}")
```

### Step 3: Model Configuration
```python
# What's happening: Setting up KNN models for both regression and classification
# Why these parameters: Starting with moderate k values and distance weighting

# Create KNN regressor
knn_regressor = KNeighborsRegressor(
    n_neighbors=5,          # Start with 5 neighbors
    weights='distance',     # Weight by inverse distance
    algorithm='auto',       # Let sklearn choose best algorithm
    metric='euclidean'      # Standard distance metric
)

# Create KNN classifier
knn_classifier = KNeighborsClassifier(
    n_neighbors=5,          # Start with 5 neighbors
    weights='distance',     # Weight by inverse distance
    algorithm='auto',       # Let sklearn choose best algorithm
    metric='euclidean'      # Standard distance metric
)

print("Model Configuration:")
print(f"Regression Model:")
print(f"  n_neighbors: {knn_regressor.n_neighbors}")
print(f"  weights: {knn_regressor.weights}")
print(f"  algorithm: {knn_regressor.algorithm}")
print(f"  metric: {knn_regressor.metric}")

print(f"\nClassification Model:")
print(f"  n_neighbors: {knn_classifier.n_neighbors}")
print(f"  weights: {knn_classifier.weights}")
print(f"  algorithm: {knn_classifier.algorithm}")
print(f"  metric: {knn_classifier.metric}")

# What the algorithm will store: All training data points for distance calculations
print(f"\nWhat KNN stores:")
print(f"  All {X_train_reg_scaled.shape[0]} training samples")
print(f"  All {X_train_reg_scaled.shape[1]} features per sample")
print(f"  No explicit model parameters (lazy learning)")
```

### Step 4: Training
```python
# What's happening: "Training" KNN models (actually just storing the data)
# What the algorithm is learning: Nothing explicitly - just stores training data for later use

import time

# Train regression model
start_time = time.time()
knn_regressor.fit(X_train_reg_scaled, y_train_reg)
reg_training_time = time.time() - start_time

# Train classification model
start_time = time.time()
knn_classifier.fit(X_train_clf_scaled, y_train_clf)
clf_training_time = time.time() - start_time

print("Model Training Completed!")
print(f"Regression training time: {reg_training_time:.4f} seconds")
print(f"Classification training time: {clf_training_time:.4f} seconds")

# Note: Training is instant because KNN just stores the data
print(f"\nTraining Process:")
print(f"  No parameters learned (non-parametric)")
print(f"  Training data stored for distance calculations")
print(f"  Actual computation happens during prediction")

# Demonstrate neighbor finding for a sample point
sample_point = X_test_reg_scaled[0:1]
distances, indices = knn_regressor.kneighbors(sample_point)

print(f"\nSample Neighbor Analysis:")
print(f"Sample point features: {X_test_reg[0]}")
print(f"Actual price: ${y_test_reg[0]:.1f}k")
print(f"\nNearest neighbors:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    neighbor_price = y_train_reg[idx]
    print(f"  Neighbor {i+1}: Distance={dist:.3f}, Price=${neighbor_price:.1f}k")
```

### Step 5: Evaluation
```python
# What's happening: Evaluating model performance and analyzing neighbor patterns
# How to interpret results: Multiple metrics show different aspects of KNN performance

# Regression predictions
start_time = time.time()
y_pred_reg = knn_regressor.predict(X_test_reg_scaled)
reg_prediction_time = time.time() - start_time

# Classification predictions
start_time = time.time()
y_pred_clf = knn_classifier.predict(X_test_clf_scaled)
y_pred_proba_clf = knn_classifier.predict_proba(X_test_clf_scaled)
clf_prediction_time = time.time() - start_time

print("Prediction Performance:")
print(f"Regression prediction time: {reg_prediction_time:.4f} seconds ({len(y_test_reg)} samples)")
print(f"Classification prediction time: {clf_prediction_time:.4f} seconds ({len(y_test_clf)} samples)")

# Regression evaluation
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)
mae = np.mean(np.abs(y_test_reg - y_pred_reg))

print(f"\nRegression Results:")
print(f"RMSE: ${rmse:.2f}k")
print(f"MAE: ${mae:.2f}k")
print(f"R² Score: {r2:.3f}")

# Classification evaluation
accuracy = knn_classifier.score(X_test_clf_scaled, y_test_clf)
print(f"\nClassification Results:")
print(f"Accuracy: {accuracy:.3f}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test_clf, y_pred_clf, target_names=['Low', 'Medium', 'High']))

# Visualize regression results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($k)')
plt.ylabel('Predicted Price ($k)')
plt.title(f'KNN Regression Results (R² = {r2:.3f})')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test_reg - y_pred_reg
plt.scatter(y_pred_reg, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price ($k)')
plt.ylabel('Residuals ($k)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Confusion matrix for classification
cm = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix - House Price Classification')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.show()
```

### Step 6: Hyperparameter Tuning and Final Predictions
```python
# What's happening: Finding optimal k value and demonstrating predictions on new data
# How to use in practice: Shows hyperparameter tuning and real-world deployment

# Hyperparameter tuning for regression
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

print("Hyperparameter Tuning:")
grid_search = GridSearchCV(
    KNeighborsRegressor(algorithm='auto'),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train_reg_scaled, y_train_reg)

print(f"Best parameters for regression: {grid_search.best_params_}")
print(f"Best CV score (RMSE): ${np.sqrt(-grid_search.best_score_):.2f}k")

# Evaluate best model
best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test_reg_scaled)
best_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_best))
best_r2 = r2_score(y_test_reg, y_pred_best)

print(f"Best model test performance:")
print(f"  RMSE: ${best_rmse:.2f}k")
print(f"  R² Score: {best_r2:.3f}")

# Example predictions on new houses
new_houses = np.array([
    [8.0, 25.0, 5.0, 1.2, 3000.0, 3.0, 37.0, -122.0],  # High-value area
    [4.0, 15.0, 8.0, 1.5, 2000.0, 2.5, 34.0, -118.0],  # Medium-value area
    [2.0, 35.0, 10.0, 2.0, 1500.0, 2.0, 36.0, -119.0]  # Lower-value area
])

print(f"\nNew House Predictions:")
print("House Features: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]")

# Scale new houses using fitted scaler
new_houses_scaled = scaler.transform(new_houses)

# Get predictions and analyze neighbors
predictions = best_knn.predict(new_houses_scaled)
distances, indices = best_knn.kneighbors(new_houses_scaled)

for i, (house, pred) in enumerate(zip(new_houses, predictions)):
    print(f"\nHouse {i+1}:")
    print(f"  Features: {house}")
    print(f"  Predicted Price: ${pred:.1f}k")

    print(f"  Nearest Neighbors Analysis:")
    neighbor_prices = y_train_reg[indices[i]]
    for j, (dist, neighbor_price) in enumerate(zip(distances[i], neighbor_prices)):
        print(f"    Neighbor {j+1}: Distance={dist:.3f}, Price=${neighbor_price:.1f}k")

    print(f"  Neighbor price range: ${neighbor_prices.min():.1f}k - ${neighbor_prices.max():.1f}k")
    print(f"  Neighbor price std: ${neighbor_prices.std():.1f}k")

# Analyze effect of k value
k_values = range(1, 21)
train_errors = []
test_errors = []

for k in k_values:
    knn_temp = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn_temp.fit(X_train_reg_scaled, y_train_reg)

    train_pred = knn_temp.predict(X_train_reg_scaled)
    test_pred = knn_temp.predict(X_test_reg_scaled)

    train_errors.append(np.sqrt(mean_squared_error(y_train_reg, train_pred)))
    test_errors.append(np.sqrt(mean_squared_error(y_test_reg, test_pred)))

# Plot bias-variance tradeoff
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_errors, 'o-', label='Training RMSE', color='blue')
plt.plot(k_values, test_errors, 'o-', label='Testing RMSE', color='red')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('RMSE ($k)')
plt.title('Bias-Variance Tradeoff in KNN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Feature importance analysis
feature_importance = []
baseline_score = r2_score(y_test_reg, best_knn.predict(X_test_reg_scaled))

for i, feature_name in enumerate(feature_names):
    # Permute feature values
    X_test_permuted = X_test_reg_scaled.copy()
    X_test_permuted[:, i] = np.random.permutation(X_test_permuted[:, i])

    # Calculate performance drop
    permuted_pred = best_knn.predict(X_test_permuted)
    permuted_score = r2_score(y_test_reg, permuted_pred)
    importance = baseline_score - permuted_score
    feature_importance.append(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sorted_indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), [feature_importance[i] for i in sorted_indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in sorted_indices], rotation=45)
plt.ylabel('Importance (R² Drop)')
plt.title('Feature Importance in KNN Model')
plt.tight_layout()
plt.show()

print(f"\nFeature Importance Analysis:")
for i in sorted_indices:
    print(f"  {feature_names[i]}: {feature_importance[i]:.3f}")
```

## Summary

**Key Takeaways:**

- **Instance-based learning** - stores all training data, no explicit model parameters
- **Distance-dependent** - requires careful feature scaling and distance metric selection
- **Local patterns** - excellent for capturing complex, non-linear relationships
- **Lazy learning** - fast training but slow prediction, especially for large datasets
- **Hyperparameter sensitive** - k value significantly affects bias-variance tradeoff
- **Curse of dimensionality** - performance degrades with high-dimensional data

**Quick Reference:**
- **Always scale features** using StandardScaler or MinMaxScaler
- Start with **k=√n** and **distance weighting**
- Use **odd k values** for binary classification to avoid ties
- **Cross-validate k** - typical range [1, 3, 5, 7, 9, 11, 15, 21]
- Evaluate with **RMSE/R²** for regression, **F1-score** for classification
- Consider **feature selection** to combat curse of dimensionality
- Use **approximate methods** (LSH, sampling) for large-scale applications
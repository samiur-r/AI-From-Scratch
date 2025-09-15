# Support Vector Machine (SVM) Quick Reference

Support Vector Machine (SVM) is a powerful supervised learning algorithm that finds the optimal hyperplane to separate classes by maximizing the margin between them. SVM can handle both linear and non-linear classification through kernel functions and can be adapted for regression tasks (SVR).

## What the Algorithm Does

SVM finds the decision boundary that maximally separates different classes by identifying support vectors - the data points closest to the decision boundary. The algorithm creates the widest possible "street" between classes, making it robust to new data points. For non-linearly separable data, SVM uses kernel functions to map data into higher dimensions where linear separation becomes possible.

**Core concept**: Find the hyperplane that maximizes the margin (distance) between classes while minimizing classification errors.

**Algorithm type**: Binary classification (extended to multi-class), regression (SVR)

The mathematical foundation:
- **Decision function**: $f(x) = \text{sign}(\sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b)$
- **Margin maximization**: $\max \frac{2}{||w||}$ subject to $y_i(w^T x_i + b) \geq 1$
- **Kernel trick**: $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ (implicit mapping to higher dimensions)
- **Soft margin**: $\min \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$ (allows some misclassification)

## When to Use It

### Problem Types
- **High-dimensional data**: Text classification, gene expression analysis, image recognition
- **Non-linear classification**: When decision boundaries are complex curves
- **Binary classification**: Natural fit, though multi-class extensions exist
- **Robust classification**: When you need models resistant to outliers
- **Small to medium datasets**: Works well with hundreds to thousands of samples

### Data Characteristics
- **High-dimensional sparse data**: Performs well even when features >> samples
- **Clean, well-preprocessed data**: Sensitive to feature scaling and outliers
- **Non-linearly separable data**: Kernel functions handle complex boundaries
- **Binary or multi-class problems**: Natural binary classifier, multi-class possible

### Business Contexts
- **Text classification**: Spam detection, document categorization, sentiment analysis
- **Bioinformatics**: Gene classification, protein structure prediction
- **Image recognition**: Handwriting recognition, face detection, medical imaging
- **Finance**: Credit scoring, fraud detection, risk assessment
- **Marketing**: Customer segmentation, churn prediction

### Comparison with Alternatives
- **Choose over Logistic Regression**: When you need non-linear boundaries and have sufficient data
- **Choose over Decision Trees**: When you want smooth boundaries and better generalization
- **Choose over Neural Networks**: When you have limited data and need theoretical guarantees
- **Choose over KNN**: When you want a parametric model and faster prediction

## Strengths & Weaknesses

### Strengths
- **Maximum margin principle**: Theoretically motivated approach for good generalization
- **Kernel trick**: Can handle non-linear relationships without explicit feature mapping
- **High-dimensional efficiency**: Performs well even when features >> samples
- **Memory efficient**: Only stores support vectors, not all training data
- **Robust to outliers**: Focus on support vectors reduces sensitivity to noise
- **Versatile**: Different kernels for different data types and relationships
- **Global optimum**: Convex optimization guarantees finding global solution

### Weaknesses
- **Sensitive to feature scaling**: Requires careful preprocessing and normalization
- **No probabilistic output**: Provides distances, not calibrated probabilities
- **Computational complexity**: Training time scales poorly with large datasets
- **Hyperparameter sensitive**: Performance highly dependent on C and kernel parameters
- **Black box with non-linear kernels**: RBF and polynomial kernels reduce interpretability
- **Memory intensive for large datasets**: Kernel matrix can become very large
- **Sensitive to class imbalance**: May bias toward majority class

## Important Hyperparameters

### Critical Parameters

**Regularization Parameter (C)**
- **Range**: 0.001 to 1000 (log scale)
- **Lower values**: More regularization, simpler decision boundary, may underfit
- **Higher values**: Less regularization, complex decision boundary, may overfit
- **Default**: C=1.0
- **Tuning strategy**: Use grid search with [0.1, 1, 10, 100, 1000]

**Kernel Function**
- **Linear**: `kernel='linear'` - for linearly separable data
- **RBF (Gaussian)**: `kernel='rbf'` - most common, handles non-linear boundaries
- **Polynomial**: `kernel='poly'` - for polynomial relationships
- **Sigmoid**: `kernel='sigmoid'` - similar to neural networks

**Gamma (for RBF/Polynomial kernels)**
- **Range**: 0.001 to 10 (log scale)
- **Lower values**: Smooth, simple decision boundary (far influence)
- **Higher values**: Complex, tight decision boundary (close influence)
- **Default**: gamma='scale' (1/(n_features * X.var()))
- **Tuning strategy**: Grid search with [0.001, 0.01, 0.1, 1, 10]

**Degree (for polynomial kernel)**
- **Range**: 2 to 5 (integer values)
- **Higher degrees**: More complex boundaries but risk overfitting
- **Default**: degree=3

### Advanced Parameters

**Class Weight**
- **Purpose**: Handle imbalanced datasets
- **Options**: None, 'balanced', or custom dictionary
- **Balanced**: Automatically adjusts weights inversely proportional to class frequencies

**Tolerance**
- **Purpose**: Stopping criterion for optimization
- **Default**: tol=0.001
- **Lower values**: More precise but slower convergence

### Parameter Tuning Examples
```python
# Grid search for optimal parameters
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]},
    {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': [0.01, 0.1, 1]}
]
```

## Key Assumptions

### Data Assumptions
- **Feature scaling**: All features should be on similar scales
- **Independent observations**: Training samples should be independent
- **Sufficient data**: Need enough data relative to feature dimensionality
- **Quality labels**: Accurate class labels for supervised learning

### Statistical Assumptions
- **Margin-based separation**: Classes should be separable with reasonable margin
- **Support vector relevance**: Decision boundary determined by few critical points
- **Kernel appropriateness**: Chosen kernel should match data relationships

### Violations and Consequences
- **Poor feature scaling**: Features with large ranges dominate distance calculations
- **Insufficient data**: May lead to overfitting and poor generalization
- **Wrong kernel choice**: Linear kernel on non-linear data leads to underfitting
- **Extreme class imbalance**: Model may ignore minority class entirely

### Preprocessing Requirements
- **Feature scaling**: Mandatory - use StandardScaler or MinMaxScaler
- **Handle missing values**: Impute missing data before training
- **Encode categorical variables**: Use appropriate encoding for categorical features
- **Outlier treatment**: Consider outlier detection and removal
- **Feature selection**: Remove irrelevant features to improve performance

## Performance Characteristics

### Time Complexity
- **Training**: O(n²) to O(n³) depending on algorithm and kernel
- **Prediction**: O(n_sv × p) where n_sv = number of support vectors
- **Kernel computation**: Additional overhead for non-linear kernels

### Space Complexity
- **Memory usage**: O(n²) for kernel matrix during training
- **Model storage**: O(n_sv × p) - only support vectors stored
- **Scalability**: Challenges with datasets > 10,000 samples

### Convergence Properties
- **Global optimum**: Convex optimization guarantees global solution
- **Convergence time**: Can be slow for large datasets
- **Numerical stability**: Generally stable but sensitive to scaling

### Scalability Characteristics
- **Sample size**: Struggles with datasets > 100,000 samples
- **Feature size**: Handles high-dimensional data well
- **Parallel processing**: Limited parallelization during training
- **Online learning**: Not naturally suited for incremental updates

## How to Evaluate & Compare Models

### Appropriate Metrics

**For Classification**
- **Accuracy**: Overall correctness for balanced datasets
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Ranking ability (requires probability calibration)
- **Precision/Recall**: Especially important for imbalanced data
- **Confusion Matrix**: Detailed breakdown of classification errors

**For Regression (SVR)**
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error (robust to outliers)
- **R²**: Proportion of variance explained
- **Epsilon-insensitive loss**: SVR-specific metric

**SVM-Specific Metrics**
- **Number of support vectors**: Model complexity indicator
- **Margin width**: Distance between support vectors
- **Decision function values**: Distance from hyperplane

### Cross-Validation Strategies
- **Stratified K-Fold**: Maintains class distribution across folds
- **Grid Search CV**: Essential for hyperparameter tuning
- **Nested CV**: For unbiased model selection and evaluation
- **Time Series Split**: For temporal data

**Recommended approach**:
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(svm_model, param_grid, cv=cv, scoring='f1')
```

### Baseline Comparisons
- **Linear SVM**: Compare with non-linear kernels
- **Logistic Regression**: Linear baseline comparison
- **Decision Trees**: Non-linear baseline
- **Dummy classifier**: Sanity check baseline

### Statistical Significance
- **McNemar's test**: Compare classification performance between models
- **Paired t-test**: Compare performance across CV folds
- **Bootstrap confidence intervals**: Estimate performance uncertainty

## Practical Usage Guidelines

### Implementation Tips
- **Always scale features**: Use StandardScaler before training
- **Start with linear kernel**: Check if linear separation works first
- **Grid search hyperparameters**: C and gamma are critical for performance
- **Handle class imbalance**: Use class_weight='balanced' for imbalanced data
- **Monitor support vectors**: High number indicates overfitting
- **Use probability calibration**: For calibrated probability estimates

### Common Mistakes
- **Forgetting feature scaling**: Most critical preprocessing step
- **Using default hyperparameters**: Always tune C and gamma
- **Wrong kernel selection**: Linear vs RBF choice significantly impacts results
- **Ignoring class imbalance**: Can lead to biased models
- **Overfitting with RBF**: High gamma values create overly complex boundaries
- **Not handling outliers**: Outliers can skew support vector selection

### Debugging Strategies
- **Visualize decision boundary**: For 2D data, plot boundary and support vectors
- **Check support vector count**: Too many indicates potential overfitting
- **Analyze feature scaling**: Ensure all features have similar ranges
- **Cross-validate systematically**: Use nested CV for reliable estimates
- **Compare kernel performance**: Try multiple kernels on your data
- **Monitor training time**: Consider algorithms for large datasets

### Production Considerations
- **Model serialization**: Save trained models with joblib or pickle
- **Prediction speed**: Consider support vector count for real-time applications
- **Memory requirements**: Monitor memory usage for large models
- **Feature consistency**: Ensure production features match training preprocessing
- **Model monitoring**: Track prediction confidence and support vector drift
- **Scalability planning**: Consider approximate methods for large-scale deployment

## Complete Example with Step-by-Step Explanation

Let's build an SVM classifier to detect cancer in breast tissue based on cell characteristics, demonstrating both linear and non-linear approaches.

### Step 1: Data Preparation
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns

# What's happening: Loading breast cancer dataset for binary classification
# Why this step: Real medical data demonstrates SVM's effectiveness on high-dimensional problems

# Load breast cancer dataset
cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target

# Create DataFrame for easier analysis
feature_names = cancer_data.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Dataset Overview:")
print(f"Dataset shape: {df.shape}")
print(f"Features: {len(feature_names)} features")
print(f"Classes: {cancer_data.target_names}")
print(f"Class distribution:")
for i, class_name in enumerate(cancer_data.target_names):
    count = np.sum(y == i)
    percentage = count / len(y) * 100
    print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

print(f"\nSample Features (first 5):")
for i, feature in enumerate(feature_names[:5]):
    print(f"  {feature}: {df[feature].mean():.2f} ± {df[feature].std():.2f}")

print(f"\nFeature Ranges (before scaling):")
print(f"  Min values: {X.min(axis=0)[:5]}")
print(f"  Max values: {X.max(axis=0)[:5]}")
print("  -> Shows need for feature scaling!")
```

### Step 2: Preprocessing
```python
# What's happening: Preparing data for SVM with critical feature scaling
# Why this step: SVM is extremely sensitive to feature scales - mandatory preprocessing

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Data Splitting:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Feature scaling - ABSOLUTELY CRITICAL for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# What's happening: Converting all features to have mean=0, std=1
# Why this step: Without scaling, features with large ranges dominate the distance calculations

print(f"\nFeature Scaling Results:")
print("Before scaling:")
print(f"  Mean range: {X_train.mean(axis=0).min():.2f} to {X_train.mean(axis=0).max():.2f}")
print(f"  Std range: {X_train.std(axis=0).min():.2f} to {X_train.std(axis=0).max():.2f}")

print("After scaling:")
print(f"  Mean range: {X_train_scaled.mean(axis=0).min():.3f} to {X_train_scaled.mean(axis=0).max():.3f}")
print(f"  Std range: {X_train_scaled.std(axis=0).min():.3f} to {X_train_scaled.std(axis=0).max():.3f}")

# Analyze feature correlation with target
feature_correlation = pd.DataFrame(X_train_scaled, columns=feature_names).corrwith(pd.Series(y_train))
print(f"\nTop 5 Most Correlated Features:")
top_features = feature_correlation.abs().sort_values(ascending=False).head()
for feature, corr in top_features.items():
    print(f"  {feature}: {corr:.3f}")
```

### Step 3: Model Configuration
```python
# What's happening: Setting up different SVM variants to compare approaches
# Why these parameters: Starting with reasonable defaults for comparison

# Create different SVM models
svm_linear = SVC(
    kernel='linear',        # Linear decision boundary
    C=1.0,                 # Moderate regularization
    random_state=42
)

svm_rbf = SVC(
    kernel='rbf',          # Radial basis function (Gaussian) kernel
    C=1.0,                 # Moderate regularization
    gamma='scale',         # Automatic gamma selection
    random_state=42
)

svm_poly = SVC(
    kernel='poly',         # Polynomial kernel
    degree=3,              # Cubic polynomial
    C=1.0,                 # Moderate regularization
    gamma='scale',         # Automatic gamma selection
    random_state=42
)

print("Model Configurations:")
print("1. Linear SVM:")
print(f"   Kernel: {svm_linear.kernel}")
print(f"   C: {svm_linear.C}")

print("\n2. RBF SVM:")
print(f"   Kernel: {svm_rbf.kernel}")
print(f"   C: {svm_rbf.C}")
print(f"   Gamma: {svm_rbf.gamma}")

print("\n3. Polynomial SVM:")
print(f"   Kernel: {svm_poly.kernel}")
print(f"   Degree: {svm_poly.degree}")
print(f"   C: {svm_poly.C}")
print(f"   Gamma: {svm_poly.gamma}")

# What each kernel does:
print(f"\nKernel Functions:")
print("  Linear: Creates straight line/hyperplane decision boundary")
print("  RBF: Creates smooth, curved decision boundaries")
print("  Polynomial: Creates polynomial-shaped decision boundaries")
```

### Step 4: Training
```python
# What's happening: Training different SVM variants and analyzing their characteristics
# What the algorithm is learning: Optimal hyperplane position and support vectors

import time

models = {
    'Linear SVM': svm_linear,
    'RBF SVM': svm_rbf,
    'Polynomial SVM': svm_poly
}

training_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Analyze model characteristics
    n_support_vectors = model.n_support_
    total_sv = np.sum(n_support_vectors)
    support_vector_ratio = total_sv / len(X_train_scaled)

    training_results[name] = {
        'model': model,
        'training_time': training_time,
        'n_support_vectors': n_support_vectors,
        'total_support_vectors': total_sv,
        'support_vector_ratio': support_vector_ratio
    }

    print(f"  Training time: {training_time:.3f} seconds")
    print(f"  Support vectors per class: {n_support_vectors}")
    print(f"  Total support vectors: {total_sv}")
    print(f"  Support vector ratio: {support_vector_ratio:.3f}")

# What the algorithm learned:
print(f"\nWhat SVM Learned:")
print("  Support vectors: Critical data points that define the decision boundary")
print("  Hyperplane: Optimal separation between classes")
print("  Margin: Maximum distance between classes")

# Analyze decision function values
for name, results in training_results.items():
    model = results['model']
    decision_values = model.decision_function(X_train_scaled)

    print(f"\n{name} Decision Function Analysis:")
    print(f"  Decision values range: {decision_values.min():.3f} to {decision_values.max():.3f}")
    print(f"  Margin violations: {np.sum(np.abs(decision_values) < 1)} samples")
```

### Step 5: Evaluation
```python
# What's happening: Evaluating model performance and comparing different kernels
# How to interpret results: Multiple metrics show different aspects of SVM performance

# Evaluate all models
evaluation_results = {}

for name, results in training_results.items():
    model = results['model']

    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    prediction_time = time.time() - start_time

    # Calculate metrics
    accuracy = model.score(X_test_scaled, y_test)

    # Get decision function values for AUC calculation
    decision_scores = model.decision_function(X_test_scaled)
    auc_score = roc_auc_score(y_test, decision_scores)

    evaluation_results[name] = {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'predictions': y_pred,
        'decision_scores': decision_scores,
        'prediction_time': prediction_time
    }

    print(f"\n{name} Results:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  AUC-ROC: {auc_score:.3f}")
    print(f"  Prediction time: {prediction_time:.4f} seconds")
    print(f"  Support vectors: {results['total_support_vectors']}")

# Detailed classification reports
print(f"\nDetailed Classification Reports:")
for name, eval_results in evaluation_results.items():
    print(f"\n{name}:")
    print(classification_report(y_test, eval_results['predictions'],
                              target_names=cancer_data.target_names))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (name, eval_results) in enumerate(evaluation_results.items()):
    cm = confusion_matrix(y_test, eval_results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=cancer_data.target_names,
                yticklabels=cancer_data.target_names)
    axes[idx].set_title(f'{name}\nAccuracy: {eval_results["accuracy"]:.3f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ROC curves comparison
plt.figure(figsize=(10, 8))
for name, eval_results in evaluation_results.items():
    fpr, tpr, _ = roc_curve(y_test, eval_results['decision_scores'])
    auc = eval_results['auc_score']
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - SVM Kernel Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Decision function analysis
plt.figure(figsize=(15, 4))
for idx, (name, eval_results) in enumerate(evaluation_results.items()):
    plt.subplot(1, 3, idx+1)

    # Separate decision scores by class
    malignant_scores = eval_results['decision_scores'][y_test == 0]
    benign_scores = eval_results['decision_scores'][y_test == 1]

    plt.hist(malignant_scores, bins=20, alpha=0.7, label='Malignant', color='red')
    plt.hist(benign_scores, bins=20, alpha=0.7, label='Benign', color='blue')
    plt.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')
    plt.xlabel('Decision Function Value')
    plt.ylabel('Frequency')
    plt.title(f'{name}\nDecision Function Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 6: Hyperparameter Tuning and Final Model
```python
# What's happening: Finding optimal hyperparameters and building final model
# How to use in practice: Systematic hyperparameter optimization for production deployment

# Define comprehensive parameter grid
param_grid = [
    # Linear kernel
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    },
    # RBF kernel
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    },
    # Polynomial kernel
    {
        'kernel': ['poly'],
        'degree': [2, 3, 4],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1]
    }
]

print("Hyperparameter Tuning:")
print(f"Total parameter combinations: {sum(len(grid['C']) * len(grid.get('gamma', [1])) * len(grid.get('degree', [1])) for grid in param_grid)}")

# Perform grid search with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=cv,
    scoring='roc_auc',  # Use AUC for imbalanced data
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation AUC: {grid_search.best_score_:.3f}")

# Evaluate best model
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
best_accuracy = best_svm.score(X_test_scaled, y_test)
best_decision_scores = best_svm.decision_function(X_test_scaled)
best_auc = roc_auc_score(y_test, best_decision_scores)

print(f"\nBest Model Performance:")
print(f"  Test Accuracy: {best_accuracy:.3f}")
print(f"  Test AUC: {best_auc:.3f}")
print(f"  Support Vectors: {best_svm.n_support_}")
print(f"  Total Support Vectors: {np.sum(best_svm.n_support_)}")

# Probability calibration for better probability estimates
print(f"\nProbability Calibration:")
calibrated_svm = CalibratedClassifierCV(best_svm, method='platt', cv=3)
calibrated_svm.fit(X_train_scaled, y_train)

# Compare calibrated vs uncalibrated probabilities
uncalibrated_proba = best_svm.decision_function(X_test_scaled)
calibrated_proba = calibrated_svm.predict_proba(X_test_scaled)[:, 1]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(uncalibrated_proba, calibrated_proba, alpha=0.6)
plt.xlabel('Uncalibrated Decision Function')
plt.ylabel('Calibrated Probability')
plt.title('Probability Calibration')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(calibrated_proba, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Calibrated Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Calibrated Probabilities')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance analysis (for linear SVM)
if best_svm.kernel == 'linear':
    feature_importance = np.abs(best_svm.coef_[0])
    sorted_indices = np.argsort(feature_importance)[::-1]

    plt.figure(figsize=(12, 8))
    top_features = sorted_indices[:15]  # Top 15 features
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance (Linear SVM)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print(f"\nTop 10 Most Important Features (Linear SVM):")
    for i, idx in enumerate(sorted_indices[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.3f}")

# Example predictions on new patients
new_patients = X_test_scaled[:5]  # Use first 5 test samples as examples
new_predictions = calibrated_svm.predict_proba(new_patients)
new_decisions = best_svm.decision_function(new_patients)

print(f"\nExample Predictions on New Patients:")
print("Patient | Actual | Predicted | Probability | Decision Score | Confidence")
print("-" * 70)

for i in range(len(new_patients)):
    actual = 'Benign' if y_test[i] == 1 else 'Malignant'
    predicted = 'Benign' if new_predictions[i][1] > 0.5 else 'Malignant'
    prob_benign = new_predictions[i][1]
    decision_score = new_decisions[i]
    confidence = 'High' if abs(decision_score) > 1 else 'Low'

    print(f"   {i+1}    | {actual:9} | {predicted:9} | {prob_benign:11.3f} | {decision_score:13.3f} | {confidence}")

# Model interpretation summary
print(f"\nModel Interpretation Summary:")
print(f"  Best kernel: {best_svm.kernel}")
print(f"  Decision boundary: {'Linear' if best_svm.kernel == 'linear' else 'Non-linear'}")
print(f"  Support vector ratio: {np.sum(best_svm.n_support_) / len(X_train_scaled):.3f}")
print(f"  Model complexity: {'Low' if best_svm.kernel == 'linear' else 'High'}")
print(f"  Interpretability: {'High' if best_svm.kernel == 'linear' else 'Limited'}")

# Performance vs complexity analysis
kernels_tested = ['linear', 'rbf', 'poly']
complexity_scores = []
performance_scores = []

for kernel in kernels_tested:
    if kernel in [result['model'].kernel for result in training_results.values()]:
        for name, result in training_results.items():
            if result['model'].kernel == kernel:
                complexity_scores.append(result['support_vector_ratio'])
                performance_scores.append(evaluation_results[name]['auc_score'])
                break

plt.figure(figsize=(8, 6))
for i, kernel in enumerate(kernels_tested):
    if i < len(complexity_scores):
        plt.scatter(complexity_scores[i], performance_scores[i],
                   s=100, label=f'{kernel.upper()} kernel')

plt.xlabel('Model Complexity (Support Vector Ratio)')
plt.ylabel('Performance (AUC Score)')
plt.title('Performance vs Complexity Trade-off')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Summary

**Key Takeaways:**

- **Maximum margin principle** - finds optimal decision boundary that maximizes separation
- **Kernel trick** - handles non-linear relationships without explicit feature mapping
- **Feature scaling critical** - mandatory preprocessing step for proper SVM function
- **Hyperparameter sensitive** - C and gamma significantly affect bias-variance tradeoff
- **Support vector efficiency** - only stores critical boundary points, not all training data
- **Probability calibration needed** - raw decision scores require calibration for probabilities

**Quick Reference:**
- **Always scale features** using StandardScaler before training
- Start with **linear kernel**, then try **RBF** for non-linear patterns
- Use **grid search** for C and gamma: [0.1, 1, 10, 100] and [0.001, 0.01, 0.1, 1]
- Monitor **support vector count** - high numbers indicate potential overfitting
- Use **class_weight='balanced'** for imbalanced datasets
- Apply **probability calibration** (Platt scaling) for reliable probability estimates
- Consider **computational limits** - training time scales poorly beyond 10k samples
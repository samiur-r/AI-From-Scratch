# Decision Trees Quick Reference

A versatile supervised learning algorithm that creates a tree-like model of decisions by recursively splitting data based on feature values to make predictions for both classification and regression tasks.

## What the Algorithm Does

Decision trees work by creating a hierarchical series of if-else conditions that partition the data into increasingly homogeneous subsets. The algorithm learns a tree structure where:

- **Internal nodes** represent feature tests (e.g., "Age < 30?")
- **Branches** represent the outcome of tests (Yes/No)
- **Leaf nodes** contain the final prediction (class or value)

**Core Process:**
1. Start with entire dataset at root
2. Find the best feature and split point that maximizes information gain
3. Create branches for each possible outcome
4. Recursively repeat for each subset until stopping criteria met
5. Assign predictions to leaf nodes based on majority class (classification) or average (regression)

**Mathematical Foundation:**
- **Gini Impurity**: $Gini = 1 - \sum_{i=1}^{c} p_i^2$ (measures node impurity)
- **Entropy**: $H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$ (information theory measure)
- **Information Gain**: $IG = H(parent) - \sum \frac{|S_v|}{|S|} H(S_v)$ (reduction in impurity)

## When to Use It

### Problem Types
- **Classification tasks**: Predicting discrete categories with clear decision boundaries
- **Regression tasks**: Predicting continuous values when relationships are non-linear
- **Feature selection**: Understanding which variables are most important
- **Rule extraction**: Creating interpretable business rules from data
- **Exploratory analysis**: Initial data understanding and pattern discovery

### Data Characteristics
- **Mixed data types**: Handles numerical and categorical features naturally
- **Non-linear relationships**: Captures complex interactions between features
- **Missing values**: Can handle missing data without imputation
- **Irrelevant features**: Automatically ignores non-informative variables
- **Small to medium datasets**: Works well without massive amounts of data

### Business Contexts
- **Medical diagnosis**: Creating diagnostic decision protocols
- **Credit approval**: Loan decision-making with clear criteria
- **Customer segmentation**: Creating actionable customer profiles
- **Fraud detection**: Identifying suspicious transaction patterns
- **Marketing campaigns**: Targeting rules for different customer segments
- **Risk assessment**: Insurance underwriting and pricing decisions

### Comparison with Alternatives
- **Choose Decision Trees over Linear Models** when relationships are non-linear and you need interpretability
- **Choose Decision Trees over Neural Networks** for smaller datasets and when explainability is crucial
- **Choose Random Forest over Single Trees** when you need better performance and can sacrifice some interpretability
- **Choose Decision Trees over SVM** when you need to understand the decision process and handle mixed data types

## Strengths & Weaknesses

### Strengths
- **Highly interpretable**: Easy to visualize and explain decisions to stakeholders
- **No assumptions about data distribution**: Works with any type of data distribution
- **Handles mixed data types**: Numerical and categorical features without preprocessing
- **Feature selection**: Automatically identifies most important variables
- **Non-linear relationships**: Captures complex patterns and interactions
- **Missing value tolerance**: Can work with incomplete data
- **Fast prediction**: O(log n) prediction time with balanced trees
- **Rule extraction**: Can convert to simple if-then rules

### Weaknesses
- **Overfitting prone**: Can create overly complex trees that don't generalize
- **Unstable**: Small data changes can result in very different trees
- **Bias toward features with many levels**: Categorical variables with many categories favored
- **Difficulty with linear relationships**: May create unnecessarily complex trees for simple linear patterns
- **Limited extrapolation**: Cannot predict beyond the range of training data
- **Greedy algorithm**: Local optimization may miss globally optimal trees
- **Class imbalance sensitivity**: Biased toward majority classes

## Important Hyperparameters

### Tree Structure Control
- **max_depth** (default: None)
  - Maximum tree depth to prevent overfitting
  - Range: [3, 20], typical values: 5-10
  - Lower values = simpler model, higher bias
  - None = unlimited depth (prone to overfitting)

- **min_samples_split** (default: 2)
  - Minimum samples required to split internal node
  - Range: [2, 100] or [0.01, 0.2] as fraction
  - Higher values = fewer splits, simpler tree

- **min_samples_leaf** (default: 1)
  - Minimum samples required at leaf node
  - Range: [1, 50] or [0.01, 0.1] as fraction
  - Higher values = smoother boundaries, less overfitting

### Split Quality Control
- **criterion** (default: 'gini' for classification, 'mse' for regression)
  - Split quality measure
  - Classification: 'gini', 'entropy'
  - Regression: 'mse', 'mae', 'friedman_mse'
  - Entropy often provides better results but slower

- **max_features** (default: None)
  - Number of features considered for best split
  - Options: None (all), 'sqrt', 'log2', integer, float
  - 'sqrt' good for Random Forest, None for single trees

### Pruning and Regularization
- **min_impurity_decrease** (default: 0.0)
  - Minimum impurity decrease required for split
  - Range: [0.0, 0.1], typical: 0.01-0.05
  - Higher values = more aggressive pruning

- **max_leaf_nodes** (default: None)
  - Maximum number of leaf nodes
  - Alternative to max_depth for tree size control
  - Range: [10, 1000] depending on problem complexity

### Tuning Strategies
- **Start simple**: Begin with max_depth=5, min_samples_split=20
- **Grid search**: Systematically test combinations of key parameters
- **Validation curves**: Plot performance vs individual parameters
- **Pruning**: Use cost-complexity pruning (ccp_alpha parameter)

### Default Recommendations
- **Beginners**: max_depth=5, min_samples_split=20, min_samples_leaf=10
- **Small datasets**: max_depth=3, min_samples_leaf=5
- **Large datasets**: max_depth=10, min_samples_split=100
- **High bias**: Increase max_depth, decrease min_samples_leaf
- **High variance**: Decrease max_depth, increase min_samples_split

## Key Assumptions

### Data Assumptions
- **Feature relevance**: Assumes informative features exist in the dataset
- **Sufficient samples**: Each class needs adequate representation
- **Feature stability**: Assumes feature relationships remain consistent over time
- **No strong linear relationships**: Works best when relationships are non-linear or rule-based

### Algorithmic Assumptions
- **Greedy optimization**: Assumes locally optimal splits lead to good global trees
- **Independence of splits**: Each split decision is made independently
- **Recursive partitioning**: Problem can be solved by dividing into subproblems
- **Stopping criteria**: Assumes defined stopping rules prevent infinite growth

### Violations and Consequences
- **Insufficient data**: Small leaf nodes lead to unreliable predictions
- **Feature noise**: Irrelevant features can create spurious splits
- **Temporal changes**: Model may become outdated if patterns change
- **Strong linear relationships**: May create unnecessarily complex trees

### Preprocessing Requirements
- **Minimal preprocessing**: Generally works with raw data
- **Missing values**: Can handle NaN or use imputation strategies
- **Categorical encoding**: Ordinal encoding for ordered categories, no dummy variables needed
- **Feature scaling**: Not required (algorithm is scale-invariant)
- **Outlier handling**: Trees are naturally robust to outliers

## Performance Characteristics

### Time Complexity
- **Training**: O(n × m × log n) where n = samples, m = features
- **Prediction**: O(log n) for balanced trees, O(n) for degenerate trees
- **Memory during training**: O(n) for storing data splits
- **Scalability**: Good for small-medium datasets, struggles with very large data

### Space Complexity
- **Model size**: O(nodes) where nodes depend on tree complexity
- **Training memory**: O(n × m) for feature sorting
- **Prediction memory**: O(depth) for traversal stack
- **Very compact models**: Final tree size much smaller than training data

### Scalability
- **Small datasets (< 1K)**: Excellent, may overfit easily
- **Medium datasets (1K-100K)**: Good performance with proper tuning
- **Large datasets (> 100K)**: Consider ensemble methods or pruning
- **High dimensions (> 1000 features)**: May struggle without feature selection

### Convergence Properties
- **Deterministic**: Same tree given same data and parameters
- **Greedy construction**: No guarantee of globally optimal tree
- **Stopping criteria**: Relies on hyperparameters to control complexity
- **Local optimization**: Each split is locally optimal but not globally

## How to Evaluate & Compare Models

### Appropriate Metrics
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Regression**: MSE, RMSE, MAE, R²
- **Tree-specific**: Tree depth, number of nodes, feature importance
- **Overfitting indicators**: Training vs validation performance gap

### Cross-Validation Strategies
- **K-Fold CV**: Standard approach for most problems
- **Stratified K-Fold**: Maintains class distribution in each fold
- **Time series**: TimeSeriesSplit for temporal data
- **Nested CV**: Hyperparameter tuning with unbiased evaluation

### Baseline Comparisons
- **Simple rules**: Domain-specific heuristics
- **Single feature**: Best single feature performance
- **Random predictions**: Random chance baseline
- **Linear models**: Compare complexity vs interpretability trade-off

### Statistical Significance
- **Bootstrap confidence intervals**: For performance metrics
- **Feature importance stability**: Consistency across CV folds
- **Tree structure consistency**: How much trees vary across folds
- **Permutation importance**: Feature significance testing

## Practical Usage Guidelines

### Implementation Tips
- **Visualize trees**: Use plot_tree() or export_graphviz() for interpretation
- **Feature importance**: Analyze feature_importances_ to understand decisions
- **Pruning**: Use cost-complexity pruning to avoid overfitting
- **Ensemble methods**: Combine with Random Forest or Gradient Boosting

### Common Mistakes
- **No depth limit**: Allowing unlimited growth leads to overfitting
- **Ignoring class imbalance**: Majority class bias in splits
- **Over-interpreting single trees**: Small data changes can alter structure significantly
- **Not validating**: Using training accuracy instead of cross-validation
- **Forgetting about instability**: Not testing robustness of tree structure

### Debugging Strategies
- **Tree visualization**: Examine actual decision paths
- **Feature importance plots**: Identify most influential variables
- **Learning curves**: Plot training vs validation scores
- **Prediction paths**: Trace specific predictions through tree
- **Cross-validation consistency**: Check for stable performance

### Production Considerations
- **Model versioning**: Track tree structure changes over time
- **Performance monitoring**: Watch for degradation in accuracy
- **Feature drift**: Monitor for changes in feature distributions
- **Explanation generation**: Provide decision paths for predictions
- **A/B testing**: Compare against existing rule-based systems

## Complete Example

### Step 1: Data Preparation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

# What's happening: Loading the wine dataset for classification demonstration
# Why this step: Wine dataset has multiple classes and mixed feature types,
# perfect for demonstrating decision tree capabilities
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='wine_class')

# Map target values to class names for interpretability
class_names = ['Class_0', 'Class_1', 'Class_2']
y_labels = pd.Series([class_names[i] for i in y], name='wine_type')

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution:\n{pd.Series(y).value_counts()}")

# Explore feature relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Feature distributions by class
X['target'] = y
for i, feature in enumerate(['alcohol', 'flavanoids', 'color_intensity', 'proline']):
    ax = axes[i//2, i%2]
    for class_idx in range(3):
        subset = X[X['target'] == class_idx][feature]
        ax.hist(subset, alpha=0.6, label=f'Class {class_idx}', bins=15)
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{feature} distribution by class')
    ax.legend()

plt.tight_layout()
plt.show()

X = X.drop('target', axis=1)  # Remove target from features
```

### Step 2: Preprocessing
```python
from sklearn.preprocessing import LabelEncoder

# What's happening: Minimal preprocessing since decision trees handle raw data well
# Why this step: Decision trees don't require scaling, but we split data for evaluation

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training class distribution:\n{pd.Series(y_train).value_counts()}")

# Check for missing values (decision trees can handle them)
print(f"\nMissing values in training set: {X_train.isnull().sum().sum()}")

# Examine feature ranges (no scaling needed for decision trees)
print(f"\nFeature ranges:")
print(f"Min values: {X_train.min().min():.2f}")
print(f"Max values: {X_train.max().max():.2f}")
print("Note: Decision trees are scale-invariant, no normalization needed")
```

### Step 3: Model Configuration
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# What's happening: Setting up decision tree with different complexity levels
# Why these parameters: Testing various tree depths and split requirements
# to find optimal balance between bias and variance

# Define models with different complexity levels
models = {
    'Simple Tree': DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    'Medium Tree': DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
    'Complex Tree': DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),
    'Entropy Tree': DecisionTreeClassifier(
        criterion='entropy',
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
}

# Compare models using cross-validation
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = {
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'scores': cv_scores
    }
    print(f"{name}: Accuracy = {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Hyperparameter tuning with GridSearch
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### Step 4: Training
```python
# What's happening: Training the best model and analyzing its structure
# What the algorithm is learning: Creating binary splits that maximize
# information gain at each node to separate classes effectively

# Use the best model from grid search
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Analyze tree structure
print("Tree Structure Analysis:")
print(f"Tree depth: {best_model.tree_.max_depth}")
print(f"Number of nodes: {best_model.tree_.node_count}")
print(f"Number of leaves: {best_model.tree_.n_leaves}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances in Decision Tree')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Interpret the most important splits
print(f"\nTree Decision Interpretation:")
print(f"Most important feature: {feature_importance.iloc[0]['feature']}")
print(f"This feature alone explains {feature_importance.iloc[0]['importance']:.1%} of the decision variance")
```

### Step 5: Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

# What's happening: Comprehensive evaluation of tree performance and structure
# How to interpret results:
# - High training accuracy with lower test accuracy suggests overfitting
# - Confusion matrix shows which classes are confused
# - Tree visualization reveals actual decision logic

# Make predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

# Calculate performance metrics
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print("Model Performance:")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"Overfitting Check: {train_accuracy - test_accuracy:.3f}")
print("(Large difference suggests overfitting)")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Add text annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Visualize the decision tree (simplified view)
plt.figure(figsize=(20, 10))
plot_tree(best_model,
          feature_names=X.columns,
          class_names=class_names,
          filled=True,
          max_depth=3,  # Show only top 3 levels for readability
          fontsize=10)
plt.title('Decision Tree Structure (Top 3 Levels)')
plt.show()

# Analyze prediction confidence
confidence_scores = np.max(y_pred_proba, axis=1)
print(f"\nPrediction Confidence Analysis:")
print(f"Mean confidence: {confidence_scores.mean():.3f}")
print(f"Low confidence predictions (< 0.7): {(confidence_scores < 0.7).sum()}")
```

### Step 6: Prediction
```python
# What's happening: Making predictions and explaining decision paths
# How to use in practice: Trees provide interpretable predictions with clear reasoning

# Simulate new wine sample
new_wine = np.array([[13.2, 2.8, 2.4, 20, 112, 1.48, 0.85, 0.31, 1.46, 7.3, 1.28, 2.88, 1065]])
new_wine_df = pd.DataFrame(new_wine, columns=X.columns)

# Make prediction
prediction = best_model.predict(new_wine)[0]
prediction_proba = best_model.predict_proba(new_wine)[0]
predicted_class = class_names[prediction]

print(f"New Wine Sample Prediction:")
print(f"Predicted Class: {predicted_class}")
print(f"Prediction Probabilities:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: {prediction_proba[i]:.3f}")

# Extract decision path
def get_decision_path(model, sample, feature_names):
    """Extract the decision path for a sample through the tree"""
    tree = model.tree_
    node_indicator = model.decision_path(sample)
    leaf_id = model.apply(sample)

    sample_id = 0
    node_index = node_indicator.toarray()[sample_id].nonzero()[0]

    print(f"\nDecision Path for Sample {sample_id}:")
    print("=" * 50)

    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            print(f"LEAF: Predicted class = {model.classes_[np.argmax(model.tree_.value[node_id])]}")
            print(f"Class probabilities: {model.tree_.value[node_id][0] / model.tree_.value[node_id][0].sum()}")
        else:
            feature = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            sample_value = sample[0, tree.feature[node_id]]

            if sample_value <= threshold:
                direction = "≤"
                next_node = "left"
            else:
                direction = ">"
                next_node = "right"

            print(f"Node {node_id}: {feature} {direction} {threshold:.3f}")
            print(f"  Sample value: {sample_value:.3f} → Go {next_node}")

# Show decision path
get_decision_path(best_model, new_wine, X.columns)

# Feature contribution analysis for this prediction
print(f"\nFeature Values for New Sample:")
feature_analysis = pd.DataFrame({
    'Feature': X.columns,
    'Value': new_wine[0],
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_analysis.head(10).to_string(index=False))

# Compare with typical values from training set
print(f"\nComparison with Training Data:")
important_features = feature_analysis.head(5)['Feature'].values
for feature in important_features:
    train_mean = X_train[feature].mean()
    train_std = X_train[feature].std()
    sample_value = new_wine_df[feature].iloc[0]
    z_score = (sample_value - train_mean) / train_std

    print(f"{feature}:")
    print(f"  Sample: {sample_value:.2f}")
    print(f"  Training mean ± std: {train_mean:.2f} ± {train_std:.2f}")
    print(f"  Z-score: {z_score:.2f}")
    if abs(z_score) > 2:
        print(f"  ⚠️  Unusual value (|z| > 2)")
```

## Summary

Decision trees are powerful, interpretable algorithms that excel at capturing non-linear relationships:

**Key Takeaways:**
- **Best for**: Non-linear patterns, mixed data types, interpretability requirements, feature selection
- **Avoid when**: Strong linear relationships, very large datasets, need for stable predictions
- **Remember**: Control complexity with pruning, validate thoroughly, consider ensemble methods
- **Extensions**: Random Forest for stability, Gradient Boosting for performance

**Quick Reference:**
- **Training time**: Fast to moderate (O(n × m × log n))
- **Prediction time**: Very fast (O(log n))
- **Interpretability**: Excellent (visual tree, clear rules)
- **Scalability**: Good for small-medium data
- **Hyperparameters**: max_depth, min_samples_split, min_samples_leaf

Decision trees serve as the foundation for powerful ensemble methods and provide unmatched interpretability for understanding complex decision processes. Master tree tuning before advancing to ensemble techniques.
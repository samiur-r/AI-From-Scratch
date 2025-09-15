# Logistic Regression Quick Reference

Logistic regression is a statistical method used for binary classification problems. Unlike linear regression which predicts continuous values, logistic regression predicts the probability that an instance belongs to a particular category using the logistic (sigmoid) function to map any real-valued input to a value between 0 and 1.

## What the Algorithm Does

Logistic regression models the probability of a binary outcome using a linear combination of features passed through the sigmoid function. The algorithm learns optimal weights (coefficients) that maximize the likelihood of observing the training data.

**Core concept**: Instead of fitting a straight line to data, logistic regression fits an S-shaped curve (sigmoid function) that asymptotically approaches 0 and 1, making it perfect for probability estimation.

**Algorithm type**: Binary classification (can be extended to multi-class)

The mathematical foundation:
- **Linear combination**: $z = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$
- **Sigmoid function**: $p = \frac{1}{1 + e^{-z}}$
- **Decision boundary**: Classify as positive if $p \geq 0.5$, negative if $p < 0.5$

## When to Use It

### Problem Types
- **Binary classification**: Email spam detection, medical diagnosis, pass/fail prediction
- **Probability estimation**: When you need probability scores, not just classifications
- **Baseline model**: Quick initial model for classification problems
- **Interpretable results**: When you need to understand feature importance and relationships

### Data Characteristics
- **Linear relationships**: Features should have linear relationship with log-odds
- **Medium-sized datasets**: Works well with hundreds to thousands of samples
- **No strict scaling requirements**: Robust to feature scaling (though scaling can improve convergence)
- **Independent observations**: Assumes observations are independent

### Business Contexts
- **Marketing**: Customer conversion prediction, click-through rate modeling
- **Finance**: Credit approval, fraud detection (as baseline)
- **Healthcare**: Disease diagnosis, treatment response prediction
- **Operations**: Quality control, failure prediction

### Comparison with Alternatives
- **Choose over Linear Regression**: When target is categorical, not continuous
- **Choose over Decision Trees**: When you need probability estimates and interpretability
- **Choose over SVM**: When you need probability outputs and faster training
- **Choose over Neural Networks**: When you have limited data and need interpretability

## Strengths & Weaknesses

### Strengths
- **Probabilistic output**: Provides confidence scores, not just binary predictions
- **No tuning required**: No hyperparameters to tune (regularization optional)
- **Fast training**: Computationally efficient, scales well to large datasets
- **Interpretable**: Coefficients directly indicate feature importance and direction
- **No assumptions about distributions**: Doesn't assume features follow specific distributions
- **Robust to outliers**: Less sensitive to outliers than linear regression
- **Well-calibrated probabilities**: Output probabilities often well-calibrated

### Weaknesses
- **Linear decision boundary**: Cannot capture complex non-linear relationships
- **Sensitive to feature scaling**: Large-scale features can dominate (though algorithm still works)
- **Requires large sample sizes**: Needs sufficient data per feature for stable results
- **Multicollinearity issues**: Correlated features can make coefficients unstable
- **Complete separation problems**: Can fail when classes are perfectly separable
- **Assumes linear log-odds**: Relationship between features and log-odds must be linear

## Important Hyperparameters

### Critical Parameters

**Regularization (C parameter in sklearn)**
- **Range**: 0.001 to 1000 (inverse of regularization strength)
- **Default**: C=1.0
- **Lower values**: More regularization, simpler model, may underfit
- **Higher values**: Less regularization, more complex model, may overfit
- **Tuning strategy**: Use cross-validation, try [0.001, 0.01, 0.1, 1, 10, 100]

**Penalty (L1 vs L2 regularization)**
- **L1 (Lasso)**: Promotes sparsity, automatic feature selection
- **L2 (Ridge)**: Shrinks coefficients, handles multicollinearity better
- **Elastic Net**: Combines L1 and L2
- **Default recommendations**: Start with L2, use L1 if you need feature selection

**Solver Algorithm**
- **liblinear**: Good for small datasets, supports L1 penalty
- **lbfgs**: Good for small to medium datasets, only L2 penalty
- **saga**: Good for large datasets, supports all penalties
- **Default recommendations**: Use 'lbfgs' for most cases, 'liblinear' for L1 regularization

**Maximum Iterations**
- **Range**: 100 to 10000
- **Default**: Usually 100-1000
- **Tuning strategy**: Increase if convergence warnings appear

### Parameter Tuning Examples
```python
# Grid search for optimal parameters
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

## Key Assumptions

### Data Assumptions
- **Linear relationship**: Log-odds should be linear function of features
- **Independence**: Observations should be independent of each other
- **No perfect multicollinearity**: Features shouldn't be perfectly correlated
- **Sufficient sample size**: Rule of thumb: 10-15 samples per feature minimum

### Statistical Assumptions
- **Binary outcome**: Target variable should be binary (0/1)
- **No extreme outliers**: Outliers can skew probability estimates
- **Complete cases**: Algorithm handles missing values poorly

### Violations and Consequences
- **Non-linear relationships**: Model will underfit, consider polynomial features
- **Perfect separation**: Algorithm may not converge, use regularization
- **Multicollinearity**: Coefficients become unstable, use L2 regularization
- **Small sample size**: Coefficients unreliable, use stronger regularization

### Preprocessing Requirements
- **Feature scaling**: Recommended but not required (improves convergence)
- **Handle missing values**: Impute or remove before training
- **Encode categorical variables**: Use one-hot encoding or label encoding
- **Remove perfect correlations**: Drop redundant features

## Performance Characteristics

### Time Complexity
- **Training**: O(n × p × i) where n=samples, p=features, i=iterations
- **Prediction**: O(p) per sample - very fast
- **Typical training time**: Seconds to minutes for most datasets

### Space Complexity
- **Memory usage**: O(p) - stores only feature weights
- **Very memory efficient**: Minimal storage requirements
- **Scalability**: Can handle millions of samples with proper solvers

### Convergence Properties
- **Guaranteed convergence**: With proper learning rate and sufficient iterations
- **Iterative optimization**: Uses gradient descent or Newton's method
- **Convergence speed**: Generally fast, usually converges in 100-1000 iterations

### Scalability Characteristics
- **Sample size**: Scales well to millions of samples
- **Feature size**: Handles thousands of features efficiently
- **Parallel processing**: Some solvers support parallel computation
- **Online learning**: Can be updated with new data incrementally

## How to Evaluate & Compare Models

### Appropriate Metrics

**For Balanced Datasets**
- **Accuracy**: Overall correctness, good when classes are balanced
- **AUC-ROC**: Area under ROC curve, measures ranking ability
- **Log-loss**: Measures probability calibration quality

**For Imbalanced Datasets**
- **Precision**: TP/(TP+FP) - "Of predicted positives, how many were correct?"
- **Recall**: TP/(TP+FN) - "Of actual positives, how many were found?"
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-PR**: Area under Precision-Recall curve

**Probability-Specific Metrics**
- **Brier Score**: Mean squared difference between predicted probabilities and actual outcomes
- **Calibration plots**: Visual assessment of probability calibration

### Cross-Validation Strategies
- **Stratified K-Fold**: Maintains class distribution across folds
- **Time Series Split**: For temporal data, train on past, test on future
- **Leave-One-Out**: For very small datasets (< 100 samples)

**Recommended approach**:
```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Baseline Comparisons
- **Dummy classifier**: Predict most frequent class or random prediction
- **Simple rules**: Domain-specific heuristics
- **Linear models**: Compare with Ridge/Lasso regression
- **Tree-based models**: Decision trees, Random Forest

### Statistical Significance
- **McNemar's test**: Compare paired predictions between models
- **Bootstrap confidence intervals**: Estimate metric uncertainty
- **Cross-validation significance tests**: Account for data dependency

## Practical Usage Guidelines

### Implementation Tips
- **Start simple**: Begin with no regularization, add if needed
- **Feature engineering**: Create polynomial features for non-linear relationships
- **Handle class imbalance**: Use class_weight='balanced' or SMOTE
- **Check convergence**: Monitor convergence warnings, increase max_iter if needed
- **Validate assumptions**: Plot residuals and check linearity assumption

### Common Mistakes
- **Ignoring probability calibration**: Check if probabilities are well-calibrated
- **Over-interpreting coefficients**: Be careful with multicollinear features
- **Using default parameters**: Always validate hyperparameters with CV
- **Ignoring feature scaling**: Can slow convergence significantly
- **Perfect separation**: Watch for infinite coefficients, use regularization

### Debugging Strategies
- **Convergence issues**: Increase max_iter, scale features, add regularization
- **Poor performance**: Check for non-linear relationships, try feature engineering
- **Unstable coefficients**: Reduce multicollinearity, add L2 regularization
- **Probability miscalibration**: Use Platt scaling or isotonic regression

### Production Considerations
- **Model monitoring**: Track prediction distribution drift
- **Retraining strategy**: Retrain when performance degrades
- **Computational efficiency**: Very fast for real-time predictions
- **Interpretability**: Coefficients provide business insights
- **Robustness**: Generally stable across different data conditions

## Complete Example with Step-by-Step Explanation

Let's build a logistic regression model to predict whether a customer will purchase a product based on their demographic and behavioral features.

### Step 1: Data Preparation
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# What's happening: Creating a synthetic dataset that mimics customer behavior
# Why this step: We need realistic data to demonstrate the algorithm's capabilities
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.normal(35, 10, n_samples)
income = np.random.normal(50000, 15000, n_samples)
website_visits = np.random.poisson(5, n_samples)
email_opens = np.random.binomial(10, 0.3, n_samples)

# Create realistic relationships for target variable
# Older, higher income customers with more engagement are more likely to purchase
purchase_probability = (
    0.1 +  # Base probability
    0.02 * (age - 25) / 10 +  # Age effect
    0.03 * (income - 30000) / 20000 +  # Income effect
    0.05 * website_visits / 5 +  # Engagement effect
    0.04 * email_opens / 10
)
purchase_probability = np.clip(purchase_probability, 0, 1)  # Keep probabilities valid
purchased = np.random.binomial(1, purchase_probability, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'website_visits': website_visits,
    'email_opens': email_opens,
    'purchased': purchased
})

print("Dataset Overview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Purchase rate: {df['purchased'].mean():.2%}")
print(f"Missing values: {df.isnull().sum().sum()}")
```

### Step 2: Preprocessing
```python
# What's happening: Preparing the data for logistic regression
# Why this step: Proper preprocessing ensures the algorithm converges properly and performs optimally

# Separate features and target
X = df.drop('purchased', axis=1)
y = df['purchased']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# What's happening: 80% for training, 20% for testing, maintaining class balance
# Why this step: We need unseen data to evaluate how well our model generalizes

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set purchase rate: {y_train.mean():.2%}")
print(f"Test set purchase rate: {y_test.mean():.2%}")

# Scale features for better convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# What's happening: Converting features to have mean=0, std=1
# Why this step: Helps gradient descent converge faster and makes coefficients more interpretable
```

### Step 3: Model Configuration
```python
# What's happening: Setting up the logistic regression model with specific parameters
# Why these parameters: Starting with moderate regularization to prevent overfitting

# Create logistic regression model
log_reg = LogisticRegression(
    C=1.0,           # Moderate regularization strength
    penalty='l2',    # L2 regularization to handle multicollinearity
    solver='lbfgs',  # Efficient solver for our dataset size
    max_iter=1000,   # Sufficient iterations for convergence
    random_state=42  # Reproducible results
)

# What the algorithm will learn: Optimal weights (coefficients) for each feature
# that maximize the likelihood of observing our training data
print("Model parameters:")
print(f"Regularization strength (C): {log_reg.C}")
print(f"Penalty type: {log_reg.penalty}")
print(f"Solver: {log_reg.solver}")
```

### Step 4: Training
```python
# What's happening: The algorithm learns the relationship between features and purchase probability
# What the algorithm is learning: Coefficients that define the linear combination in the sigmoid function

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Display learned parameters
print("Model Training Completed!")
print("\nLearned Coefficients:")
feature_names = X.columns
for feature, coef in zip(feature_names, log_reg.coef_[0]):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {log_reg.intercept_[0]:.4f}")

# Interpret coefficients
print("\nCoefficient Interpretation:")
print("Positive coefficients increase probability of purchase")
print("Negative coefficients decrease probability of purchase")
print("Larger absolute values indicate stronger influence")
```

### Step 5: Evaluation
```python
# What's happening: Evaluating model performance on unseen test data
# How to interpret results: Multiple metrics give different perspectives on model quality

# Make predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]  # Probability of purchase

# Classification metrics
print("Classification Results:")
print(f"Accuracy: {log_reg.score(X_test_scaled, y_test):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nMatrix Interpretation:")
print(f"True Negatives (correct non-purchases): {cm[0,0]}")
print(f"False Positives (predicted purchase, didn't buy): {cm[0,1]}")
print(f"False Negatives (predicted no purchase, but bought): {cm[1,0]}")
print(f"True Positives (correct purchase predictions): {cm[1,1]}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Purchase Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Step 6: Prediction
```python
# What's happening: Using the trained model to make predictions on new customer data
# How to use in practice: This shows how to deploy the model for real-world predictions

# Example: New customer data
new_customer = pd.DataFrame({
    'age': [28, 45, 52],
    'income': [35000, 75000, 90000],
    'website_visits': [2, 8, 12],
    'email_opens': [1, 6, 9]
})

print("New Customer Predictions:")
print("Customer Data:")
print(new_customer)

# Scale new data using fitted scaler
new_customer_scaled = scaler.transform(new_customer)

# Get predictions and probabilities
predictions = log_reg.predict(new_customer_scaled)
probabilities = log_reg.predict_proba(new_customer_scaled)[:, 1]

print("\nPrediction Results:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Customer {i+1}: {'Will Purchase' if pred else 'Will Not Purchase'} "
          f"(Probability: {prob:.3f})")

# Business interpretation
print("\nBusiness Insights:")
print("- Customer 1: Low probability (young, lower income, low engagement)")
print("- Customer 2: Moderate probability (middle-aged, good income, high engagement)")
print("- Customer 3: High probability (older, high income, very engaged)")

# Feature importance visualization
plt.figure(figsize=(10, 6))
feature_importance = abs(log_reg.coef_[0])
plt.barh(feature_names, feature_importance)
plt.xlabel('Coefficient Magnitude (Feature Importance)')
plt.title('Logistic Regression Feature Importance')
plt.grid(True, alpha=0.3)
for i, v in enumerate(feature_importance):
    plt.text(v + 0.001, i, f'{v:.3f}', va='center')
plt.tight_layout()
plt.show()

# Model equation
print("\nModel Equation:")
print("log_odds = {:.4f}".format(log_reg.intercept_[0]), end="")
for feature, coef in zip(feature_names, log_reg.coef_[0]):
    print(f" + {coef:.4f}*{feature}", end="")
print(f"\nprobability = 1 / (1 + exp(-log_odds))")
```

## Summary

**Key Takeaways:**

- **Best for binary classification** with interpretable results and probability estimates
- **Linear decision boundary** - use feature engineering for non-linear relationships
- **Fast and efficient** - excellent baseline model for classification problems
- **Probabilistic output** - provides confidence scores, not just classifications
- **Regularization important** - use L2 for stability, L1 for feature selection
- **Requires preprocessing** - scale features and handle missing values properly

**Quick Reference:**
- Use when you need **interpretable probabilities**
- Start with **C=1.0, penalty='l2', solver='lbfgs'**
- Evaluate with **AUC-ROC** for ranking, **F1-score** for imbalanced data
- Monitor **convergence warnings** and **coefficient stability**
- Consider **feature engineering** for non-linear relationships
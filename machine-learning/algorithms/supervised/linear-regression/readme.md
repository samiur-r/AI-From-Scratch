# Linear Regression Quick Reference

A fundamental supervised learning algorithm that models the relationship between a dependent variable and independent variables by fitting a linear equation to observed data.

## What the Algorithm Does

Linear regression finds the best-fitting straight line (or hyperplane in multiple dimensions) through data points to predict continuous numerical values. It establishes a linear relationship between input features (X) and the target variable (y) using the equation:

**Simple Linear Regression:** $y = mx + b$
**Multiple Linear Regression:** $y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε$

Where:
- $y$ = dependent variable (target)
- $x$ = independent variables (features)
- $β$ = coefficients (weights)
- $ε$ = error term

The algorithm minimizes the sum of squared residuals (differences between predicted and actual values) using the least squares method.

## When to Use It

### Problem Types
- **Regression tasks**: Predicting continuous numerical outcomes
- **Baseline modeling**: Often the first algorithm to try for regression problems
- **Feature importance analysis**: Understanding which variables impact the target most
- **Trend analysis**: Identifying linear relationships in data

### Data Characteristics
- **Small to medium datasets**: Works well with limited data
- **Linear relationships**: Best when features have linear correlation with target
- **Continuous target variables**: Designed for numerical predictions
- **Clean data**: Sensitive to outliers and requires preprocessing

### Business Contexts
- **Sales forecasting**: Predicting revenue based on marketing spend
- **Risk assessment**: Credit scoring, insurance pricing
- **Real estate**: House price prediction based on features
- **Economics**: GDP prediction, stock price modeling
- **Scientific research**: Hypothesis testing, controlled experiments

### Comparison with Alternatives
- **Choose Linear Regression over Decision Trees** when you need interpretability and have linear relationships
- **Choose Linear Regression over Neural Networks** for simple problems with limited data
- **Choose Polynomial Regression over Linear** when relationships are non-linear but still relatively simple
- **Choose Ridge/Lasso over Linear** when dealing with multicollinearity or feature selection

## Strengths & Weaknesses

### Strengths
- **Highly interpretable**: Easy to understand coefficient meanings and feature importance
- **Fast training and prediction**: Computationally efficient, closed-form solution available
- **No hyperparameter tuning required**: Simple implementation with default parameters
- **Statistical foundation**: Well-established theory with confidence intervals and significance tests
- **Baseline performance**: Provides good starting point for comparison with complex models
- **Small data friendly**: Works well even with limited training samples
- **Probabilistic output**: Can provide prediction intervals and uncertainty estimates

### Weaknesses
- **Assumes linear relationships**: Cannot capture complex non-linear patterns
- **Sensitive to outliers**: Extreme values can significantly skew the model
- **Requires feature scaling**: Performance affected by different feature scales
- **Multicollinearity issues**: Correlated features can make coefficients unstable
- **Overfitting with many features**: Poor performance when features >> samples
- **Assumes residual normality**: Violations can affect statistical inference
- **No automatic feature selection**: Requires manual feature engineering

## Important Hyperparameters

### Core Parameters
- **fit_intercept** (default: True)
  - Whether to calculate the intercept term (β₀)
  - Set to False if data is already centered
  - Range: True/False

- **normalize** (deprecated, use StandardScaler instead)
  - Previously controlled feature normalization
  - Always use preprocessing pipelines for scaling

### Regularized Variants
- **alpha** (Ridge/Lasso): Controls regularization strength
  - Ridge: Typical range [0.1, 100], default often 1.0
  - Lasso: Typical range [0.001, 1.0], default often 0.1
  - Higher values = more regularization = simpler model

- **l1_ratio** (Elastic Net): Balance between Ridge and Lasso
  - Range: [0, 1], where 0 = Ridge, 1 = Lasso
  - Default: 0.5 for balanced regularization

### Tuning Strategies
- **Cross-validation**: Use GridSearchCV or cross_val_score for alpha selection
- **Regularization path**: Plot validation curves to find optimal alpha
- **Start simple**: Begin with basic Linear Regression, add regularization if needed

### Default Recommendations
- **Beginners**: Start with default LinearRegression()
- **Many features**: Try Ridge with alpha=1.0
- **Feature selection needed**: Try Lasso with alpha=0.1
- **Best of both**: Try ElasticNet with l1_ratio=0.5

## Key Assumptions

### Data Assumptions
- **Linearity**: The relationship between X and y is linear
- **Independence**: Observations are independent of each other
- **Homoscedasticity**: Constant variance of residuals across all levels of X
- **No multicollinearity**: Independent variables are not highly correlated with each other

### Statistical Assumptions
- **Normality of residuals**: Errors are normally distributed (for inference)
- **Zero mean errors**: Residuals have mean of zero
- **No autocorrelation**: Residuals are not correlated with each other (important for time series)

### Violations and Consequences
- **Non-linearity**: Use polynomial features or different algorithms
- **Heteroscedasticity**: Consider weighted least squares or robust standard errors
- **Multicollinearity**: Use Ridge regression or remove correlated features
- **Non-normal residuals**: Affects confidence intervals, not predictions

### Preprocessing Requirements
- **Feature scaling**: Standardize features for regularized versions
- **Outlier handling**: Remove or cap extreme values
- **Missing value imputation**: Linear regression cannot handle NaN values
- **Categorical encoding**: Convert categorical variables to numerical

## Performance Characteristics

### Time Complexity
- **Training**: O(n × p²) where n = samples, p = features
- **Prediction**: O(p) per sample
- **Closed-form solution**: Can solve directly without iteration
- **Large datasets**: Consider stochastic gradient descent for scalability

### Space Complexity
- **Memory**: O(p²) for normal equation, O(p) for gradient descent
- **Model size**: Stores only p coefficients plus intercept
- **Very memory efficient**: Minimal storage requirements

### Scalability
- **Small datasets (< 1K samples)**: Excellent performance
- **Medium datasets (1K-100K)**: Good performance with proper preprocessing
- **Large datasets (> 100K)**: Consider SGDRegressor for online learning
- **High dimensions**: Regularization becomes essential

### Convergence Properties
- **Closed-form solution**: Guaranteed global optimum
- **No convergence issues**: Unlike iterative methods
- **Numerical stability**: Can be unstable with multicollinearity
- **SGD variant**: Requires learning rate tuning and convergence monitoring

## How to Evaluate & Compare Models

### Appropriate Metrics
- **Mean Squared Error (MSE)**: Primary optimization target, sensitive to outliers
- **Root Mean Squared Error (RMSE)**: Same units as target variable
- **Mean Absolute Error (MAE)**: Robust to outliers, easier to interpret
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **Adjusted R²**: Penalizes for additional features

### Cross-Validation Strategies
- **K-Fold CV**: Standard approach for most datasets
- **Stratified sampling**: For regression, ensure balanced target distribution
- **Time series split**: For temporal data, use TimeSeriesSplit
- **Leave-one-out**: For very small datasets

### Baseline Comparisons
- **Mean baseline**: Always predict the mean of training targets
- **Simple rules**: Domain-specific heuristics
- **Previous model**: Compare against existing production model
- **More complex models**: Decision trees, random forest, neural networks

### Statistical Significance
- **Confidence intervals**: For coefficient estimates
- **P-values**: Test statistical significance of features
- **F-statistic**: Overall model significance
- **Cross-validation confidence**: Use multiple CV runs to assess variance

## Practical Usage Guidelines

### Implementation Tips
- **Always visualize data**: Plot features vs target to check linearity
- **Check assumptions**: Use residual plots and statistical tests
- **Feature engineering**: Create polynomial features for non-linear relationships
- **Pipeline usage**: Always use scikit-learn pipelines for preprocessing

### Common Mistakes
- **Not scaling features**: Especially important for regularized versions
- **Ignoring outliers**: Can severely impact model performance
- **Multicollinearity**: Check correlation matrix and VIF scores
- **Overfitting**: Too many features relative to samples
- **Wrong evaluation**: Using training error instead of validation error

### Debugging Strategies
- **Residual analysis**: Plot residuals vs fitted values and features
- **Feature importance**: Examine coefficient magnitudes and signs
- **Learning curves**: Plot training vs validation scores
- **Cross-validation**: Use multiple folds to check consistency

### Production Considerations
- **Model monitoring**: Track prediction accuracy over time
- **Feature drift**: Monitor for changes in feature distributions
- **Retraining schedule**: Retrain when performance degrades
- **Interpretability**: Document coefficient meanings for stakeholders

## Complete Example

### Step 1: Data Preparation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# What's happening: Loading the Boston housing dataset for demonstration
# Why this step: We need a real dataset with continuous target variable to demonstrate linear regression
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MEDV')

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: mean={y.mean():.2f}, std={y.std():.2f}")

# Explore the relationship between features and target
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X['RM'], y, alpha=0.6)
plt.xlabel('Average number of rooms (RM)')
plt.ylabel('Median home value')
plt.title('Linear relationship example')

plt.subplot(1, 3, 2)
plt.scatter(X['LSTAT'], y, alpha=0.6)
plt.xlabel('% lower status population (LSTAT)')
plt.ylabel('Median home value')
plt.title('Negative correlation example')

plt.subplot(1, 3, 3)
plt.hist(y, bins=20, alpha=0.7)
plt.xlabel('Median home value')
plt.ylabel('Frequency')
plt.title('Target distribution')
plt.tight_layout()
plt.show()
```

### Step 2: Preprocessing
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# What's happening: Creating preprocessing pipeline with feature scaling
# Why this step: Linear regression is sensitive to feature scales, and standardization
# ensures all features contribute equally to the distance calculations
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Check for multicollinearity
correlation_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((correlation_matrix.columns[i],
                                  correlation_matrix.columns[j],
                                  correlation_matrix.iloc[i, j]))

if high_corr_pairs:
    print("High correlation pairs (>0.8):")
    for pair in high_corr_pairs:
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
```

### Step 3: Model Configuration
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

# What's happening: Setting up different linear regression variants
# Why these parameters: Starting with basic linear regression, then adding regularization
# to handle potential overfitting and multicollinearity

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),  # L2 regularization for multicollinearity
    'Lasso (L1)': Lasso(alpha=0.1)   # L1 regularization for feature selection
}

# Compare models using cross-validation
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                               scoring='neg_mean_squared_error')
    cv_results[name] = {
        'mean_mse': -cv_scores.mean(),
        'std_mse': cv_scores.std(),
        'mean_rmse': np.sqrt(-cv_scores.mean())
    }
    print(f"{name}: RMSE = {cv_results[name]['mean_rmse']:.3f} (+/- {cv_scores.std():.3f})")
```

### Step 4: Training
```python
# What's happening: Training the best performing model on full training set
# What the algorithm is learning: Finding optimal coefficients that minimize
# the sum of squared differences between predicted and actual values

# Select best model (let's assume Ridge performed best)
best_model = Ridge(alpha=1.0)
best_model.fit(X_train, y_train)

# What the algorithm learned
print("Model Coefficients:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': best_model.coef_,
    'abs_coefficient': np.abs(best_model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(feature_importance.head(10))
print(f"\nIntercept: {best_model.intercept_:.3f}")

# Interpret coefficients
print("\nTop 3 most important features:")
for idx, row in feature_importance.head(3).iterrows():
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    print(f"- {row['feature']}: 1 std increase {direction} price by ${abs(row['coefficient']):.3f}k")
```

### Step 5: Evaluation
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

# What's happening: Evaluating model performance using multiple metrics
# How to interpret results:
# - R² close to 1 indicates good fit
# - RMSE should be small relative to target range
# - Residuals should be randomly distributed

y_pred = best_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: ${rmse:.3f}k")
print(f"MAE: ${mae:.3f}k")
print(f"Target range: ${y_test.min():.1f}k - ${y_test.max():.1f}k")

# Residual analysis
residuals = y_test - y_pred

plt.figure(figsize=(15, 5))

# Predicted vs Actual
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual')

# Residuals vs Predicted
plt.subplot(1, 3, 2)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

# QQ plot for normality of residuals
plt.subplot(1, 3, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()

# Check assumptions
print(f"\nAssumption Checks:")
print(f"Mean of residuals: {residuals.mean():.6f} (should be ~0)")
print(f"Shapiro-Wilk test p-value: {stats.shapiro(residuals)[1]:.6f}")
print("(p > 0.05 suggests residuals are normally distributed)")
```

### Step 6: Prediction
```python
# What's happening: Making predictions on new data and providing uncertainty estimates
# How to use in practice: Always preprocess new data the same way as training data

# Simulate new data point
new_data = np.array([[0.1, 15.0, 10.0, 0, 0.6, 6.5, 70.0, 3.0, 4, 300, 16.0, 350, 8.0]])
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = best_model.predict(new_data_scaled)[0]
print(f"Predicted house price: ${prediction:.1f}k")

# Prediction interval (approximate)
residual_std = np.std(residuals)
confidence_interval = 1.96 * residual_std  # 95% confidence interval
print(f"95% Prediction Interval: ${prediction - confidence_interval:.1f}k - ${prediction + confidence_interval:.1f}k")

# Feature contribution analysis
feature_contributions = new_data_scaled[0] * best_model.coef_
contribution_df = pd.DataFrame({
    'feature': X.columns,
    'value': new_data[0],
    'scaled_value': new_data_scaled[0],
    'contribution': feature_contributions
}).sort_values('contribution', key=abs, ascending=False)

print(f"\nTop 5 features contributing to this prediction:")
for idx, row in contribution_df.head(5).iterrows():
    sign = "+" if row['contribution'] > 0 else ""
    print(f"- {row['feature']}: {sign}{row['contribution']:.3f}")
```

## Summary

Linear regression is the foundation of machine learning, offering:

**Key Takeaways:**
- **Best for**: Linear relationships, interpretability needs, small datasets, baseline modeling
- **Avoid when**: Complex non-linear patterns, many irrelevant features, very large datasets
- **Remember**: Always check assumptions, handle outliers, and use proper preprocessing
- **Extensions**: Ridge/Lasso for regularization, polynomial features for non-linearity

**Quick Reference:**
- **Training time**: Very fast (closed-form solution)
- **Prediction time**: Extremely fast
- **Interpretability**: Excellent (coefficient meanings)
- **Scalability**: Good for small-medium data
- **Hyperparameters**: Minimal (mainly regularization strength)

Linear regression serves as both a powerful standalone algorithm and the building block for more advanced techniques. Master it first before moving to complex models.
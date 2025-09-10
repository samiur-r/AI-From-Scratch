# Scikit-learn Quick Reference

Scikit-learn is the most popular machine learning library for Python, providing simple and efficient tools for data mining and analysis. It features various classification, regression, clustering algorithms, and tools for model selection, preprocessing, and evaluation. Built on NumPy, SciPy, and matplotlib.

### Installation
```bash
pip install scikit-learn
```

### Importing Scikit-learn

```
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
```

* * * * *

2\. Loading Data
----------------

```
# Built-in datasets
from sklearn.datasets import load_iris, load_boston, load_wine, make_classification

# Load classic datasets
iris = load_iris()
X, y = iris.data, iris.target  # Features and targets
print(X.shape)  # (150, 4) - 150 samples, 4 features

# Generate synthetic data
X_synth, y_synth = make_classification(n_samples=1000, n_features=4, n_classes=2)

# Load from pandas DataFrame
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)  # Features
y = df['target']               # Target variable

# Common dataset types
# Classification: iris, wine, breast_cancer
# Regression: boston, california_housing, diabetes
# Clustering: make_blobs, make_circles

```

* * * * *

3\. Data Preprocessing
----------------------

```
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Feature scaling
scaler = StandardScaler()  # Mean=0, Std=1
X_scaled = scaler.fit_transform(X)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data (don't fit!)

# Min-Max scaling (0 to 1)
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

# Label encoding for categorical variables
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(['cat', 'dog', 'cat', 'bird'])  # [0, 1, 0, 2]

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse=False)
categories = [['red'], ['blue'], ['red'], ['green']]
encoded = onehot.fit_transform(categories)  # Binary columns for each category

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Can use 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)

```

* * * * *

4\. Train-Test Split
--------------------

```
from sklearn.model_selection import train_test_split

# Basic split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stratified split (maintains class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Multiple splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Results in 60% train, 20% validation, 20% test

```

* * * * *

5\. Classification
------------------

```
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)  # Probabilities for each class

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Support Vector Machine
svm = SVC(kernel='rbf', probability=True)  # Enable probability for predict_proba
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Feature importance (for tree-based models)
importances = rf.feature_importances_
print(f"Most important feature: {np.argmax(importances)}")

```

* * * * *

6\. Regression
--------------

```
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print(f"Coefficients: {lin_reg.coef_}")
print(f"Intercept: {lin_reg.intercept_}")

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0)  # Higher alpha = more regularization
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Lasso Regression (L1 regularization, feature selection)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

```

* * * * *

7\. Clustering
--------------

```
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Determine optimal number of clusters (Elbow method)
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# DBSCAN (density-based clustering)
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)
print(f"Number of clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}")

# Hierarchical Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = agg_clustering.fit_predict(X)

# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
clusters = gmm.fit_predict(X)
probabilities = gmm.predict_proba(X)  # Soft clustering

```

* * * * *

8\. Model Evaluation
--------------------

```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Classification Metrics
accuracy = accuracy_score(y_test, y_pred)  # Correct predictions / Total
precision = precision_score(y_test, y_pred, average='macro')  # TP / (TP + FP)
recall = recall_score(y_test, y_pred, average='macro')     # TP / (TP + FN)
f1 = f1_score(y_test, y_pred, average='macro')            # Harmonic mean of precision/recall

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)  # [[True_Neg, False_Pos], [False_Neg, True_Pos]]

# Detailed classification report
report = classification_report(y_test, y_pred, target_names=['Class_0', 'Class_1'])
print(report)

# ROC Curve and AUC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])  # For binary classification
roc_auc = auc(fpr, tpr)

# Regression Metrics
mse = mean_squared_error(y_test, y_pred)      # Mean Squared Error
rmse = np.sqrt(mse)                           # Root Mean Squared Error  
r2 = r2_score(y_test, y_pred)                 # R-squared (coefficient of determination)
mae = mean_absolute_error(y_test, y_pred)     # Mean Absolute Error

```

* * * * *

9\. Cross-Validation
--------------------

```
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

# Basic cross-validation
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Stratified K-Fold (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)

# Cross-validation with custom scoring
scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')

# Multiple metrics
from sklearn.model_selection import cross_validate
scoring = ['accuracy', 'precision_macro', 'recall_macro']
scores = cross_validate(model, X, y, cv=5, scoring=scoring)

# Leave-One-Out Cross-Validation
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)

```

* * * * *

10\. Hyperparameter Tuning
---------------------------

```
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_}")
best_model = grid_search.best_estimator_

# Randomized Search (faster for large parameter spaces)
from scipy.stats import randint, uniform
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, None],
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, 
                                 scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Bayesian Optimization (requires scikit-optimize)
# from skopt import BayesSearchCV

```

* * * * *

11\. Pipeline and Feature Selection
-----------------------------------

```
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Create a pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),           # Step 1: Scale features
    ('selector', SelectKBest(k=10)),        # Step 2: Select top 10 features  
    ('classifier', LogisticRegression())    # Step 3: Train classifier
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Pipeline with grid search
param_grid = {
    'selector__k': [5, 10, 15],
    'classifier__C': [0.1, 1, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

# Feature selection methods
from sklearn.feature_selection import RFE, SelectFromModel

# Recursive Feature Elimination
rfe = RFE(LogisticRegression(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X_train, y_train)

# Select features based on model importance
selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
X_selected = selector.fit_transform(X_train, y_train)

```

* * * * *

12\. Dimensionality Reduction
-----------------------------

```
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Principal Component Analysis
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(f"Variance explained: {explained_variance}")

# Choose number of components
pca_full = PCA()
pca_full.fit(X)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1  # 95% variance retained

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Linear Discriminant Analysis (supervised)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Independent Component Analysis
from sklearn.decomposition import FastICA
ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X)

```

* * * * *

13\. Ensemble Methods
---------------------

```
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Voting Classifier (combines multiple algorithms)
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
    voting='soft'  # Use probabilities (requires probability=True for SVM)
)
voting_clf.fit(X_train, y_train)

# Bagging (Bootstrap Aggregating)
bagging = BaggingClassifier(
    base_estimator=LogisticRegression(),
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train, y_train)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

# XGBoost (requires xgboost package)
# import xgboost as xgb
# xgb_model = xgb.XGBClassifier()

```

* * * * *

14\. Model Persistence
----------------------

```
import joblib
import pickle

# Save model using joblib (recommended)
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

# Load model
loaded_model = joblib.load('model.pkl')
predictions = loaded_model.predict(X_test)

# Save with pickle
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load with pickle
with open('model_pickle.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Save entire pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
pipe.fit(X_train, y_train)
joblib.dump(pipe, 'pipeline.pkl')

# Version control for models
model_info = {
    'model': model,
    'scaler': scaler,
    'feature_names': X_train.columns.tolist(),
    'model_version': '1.0',
    'training_date': '2024-01-01'
}
joblib.dump(model_info, 'model_with_metadata.pkl')

```

* * * * *

15\. Advanced Topics
--------------------

```
# Imbalanced datasets
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Simple resampling
majority = df[df.target == 0]
minority = df[df.target == 1]
minority_upsampled = resample(minority, n_samples=len(majority), random_state=42)
balanced_df = pd.concat([majority, minority_upsampled])

# SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Class weights for imbalanced data
clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

# Calibration
from sklearn.calibration import CalibratedClassifierCV
calibrated_clf = CalibratedClassifierCV(SVC(), cv=3)
calibrated_clf.fit(X_train, y_train)
calibrated_probs = calibrated_clf.predict_proba(X_test)

# Multi-label classification
from sklearn.multioutput import MultiOutputClassifier
multi_clf = MultiOutputClassifier(RandomForestClassifier())
# y should be 2D array where each column is a different label

```

* * * * *

Summary
=======

-   **Preprocessing** is crucial: scale features, handle missing values, encode categories.

-   Use **pipelines** to chain preprocessing and modeling steps together.

-   **Cross-validation** provides reliable performance estimates and prevents overfitting.

-   **Hyperparameter tuning** with GridSearchCV/RandomizedSearchCV improves model performance.

-   **Ensemble methods** often outperform single models by combining multiple approaches.

-   Always **evaluate** models properly using appropriate metrics for your problem type.
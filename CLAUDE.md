# Claude Code Guidelines for AI-From-Scratch Repository

## README Generation Guidelines for Machine Learning Algorithms

When generating README files for machine learning algorithms in this repository, ensure each algorithm documentation includes the following comprehensive sections:

### 1. Algorithm Overview
- **What the algorithm does**: Clear, concise explanation of the algorithm's purpose and functionality
- **Core concept**: The mathematical or logical foundation behind the algorithm
- **Algorithm type**: Classification, regression, clustering, etc.

### 2. When to Use It (Applicability)
- **Problem types**: Specific scenarios where this algorithm excels
- **Data characteristics**: Size, dimensionality, distribution requirements
- **Business contexts**: Real-world applications and use cases
- **Comparison with alternatives**: When to choose this over other algorithms

### 3. Strengths & Weaknesses
#### Strengths
- Performance advantages
- Interpretability
- Computational efficiency
- Robustness to certain data conditions

#### Weaknesses
- Limitations and failure modes
- Computational bottlenecks
- Data requirements and assumptions
- Common pitfalls

### 4. Important Hyperparameters
- **Critical parameters**: Those that significantly impact performance
- **Parameter ranges**: Typical values and bounds
- **Tuning strategies**: How to optimize each parameter
- **Default recommendations**: Good starting points for beginners

### 5. Key Assumptions
- **Data assumptions**: Distribution, independence, linearity, etc.
- **Statistical assumptions**: What the algorithm assumes about the underlying data
- **Violations**: What happens when assumptions are broken
- **Preprocessing requirements**: Data preparation needed

### 6. Performance Characteristics
- **Time complexity**: Training and prediction time
- **Space complexity**: Memory requirements
- **Scalability**: Performance with increasing data size/features
- **Convergence properties**: How the algorithm reaches solutions

### 7. How to Evaluate & Compare Models
- **Appropriate metrics**: Which evaluation metrics work best
- **Cross-validation strategies**: Recommended validation approaches
- **Baseline comparisons**: What to compare against
- **Statistical significance**: How to ensure results are meaningful

### 8. Practical Usage Guidelines
- **Implementation tips**: Best practices for coding
- **Common mistakes**: What to avoid
- **Debugging strategies**: How to troubleshoot issues
- **Production considerations**: Deployment and monitoring

### 9. Complete Example with Step-by-Step Explanation

Each algorithm README must include one comprehensive example that demonstrates:

#### Step 1: Data Preparation
```python
# What's happening: [Explain the data loading and initial exploration]
# Why this step: [Explain why this preparation is necessary]
```

#### Step 2: Preprocessing
```python
# What's happening: [Explain each preprocessing step]
# Why this step: [Explain the rationale for each transformation]
```

#### Step 3: Model Configuration
```python
# What's happening: [Explain parameter choices]
# Why these parameters: [Justify the hyperparameter selection]
```

#### Step 4: Training
```python
# What's happening: [Explain the training process]
# What the algorithm is learning: [Describe what's being optimized]
```

#### Step 5: Evaluation
```python
# What's happening: [Explain evaluation metrics and process]
# How to interpret results: [Guide for understanding output]
```

#### Step 6: Prediction
```python
# What's happening: [Explain prediction process]
# How to use in practice: [Real-world application guidance]
```

## Code Style Guidelines

### Code Blocks
- Include complete, runnable examples
- Add comments explaining each major step
- Use meaningful variable names
- Include import statements

### Mathematical Notation
- Use LaTeX for complex formulas: `$\sum_{i=1}^{n} x_i$`
- Explain mathematical concepts in plain language
- Provide intuitive interpretations of formulas

### Visual Elements
- Include algorithm flowcharts when helpful
- Add decision trees or visual representations
- Use tables for parameter comparisons
- Include performance comparison charts

## Structure Template

```markdown
# [Algorithm Name] Quick Reference

[Brief description of what the algorithm does]

## What the Algorithm Does
[Detailed explanation]

## When to Use It
[Applicability section]

## Strengths & Weaknesses
### Strengths
- [List strengths]

### Weaknesses
- [List weaknesses]

## Important Hyperparameters
[Parameter details with explanations]

## Key Assumptions
[List and explain assumptions]

## Performance Characteristics
[Time/space complexity and scalability]

## Evaluation & Comparison
[Metrics and validation strategies]

## Practical Usage
[Implementation tips and best practices]

## Complete Example
[Step-by-step implementation with explanations]

## Summary
[Key takeaways and quick reference points]
```

## Quality Standards

- **Completeness**: All 9 sections must be present and comprehensive
- **Clarity**: Explanations should be accessible to beginners but detailed enough for practitioners
- **Accuracy**: Technical details must be correct and up-to-date
- **Practicality**: Focus on actionable guidance and real-world applicability
- **Consistency**: Follow the same structure and style across all algorithm READMEs

## Examples of Good Explanations

### What's Happening Explanations
```python
# What's happening: We're splitting the dataset into 80% training and 20% testing
# Why this step: This allows us to evaluate how well our model generalizes to unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Algorithm Process Explanations
```python
# What the algorithm is learning: Finding the optimal hyperplane that maximizes
# the margin between different classes while minimizing classification errors
# The support vectors are the critical data points that define this boundary
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
```

### Interpretation Guidance
```python
# How to interpret results:
# - Accuracy above 0.8 indicates good performance for this dataset
# - Precision tells us how many predicted positives were actually positive
# - Recall shows how many actual positives we successfully identified
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
```

## Repository Integration

- Link to related algorithms and concepts
- Reference other sections of the repository when relevant
- Maintain consistency with the overall repository structure and style
- Include cross-references to practical projects that use the algorithm

---

**Note**: These guidelines ensure that every algorithm README provides comprehensive, practical guidance that helps users understand not just how to implement the algorithm, but when and why to use it effectively.
# Claude Code Guidelines for AI-From-Scratch Repository

This repository contains documentation for both **machine learning algorithms** and **machine learning frameworks**. Each type requires a different documentation approach:

- **Algorithms** (e.g., Support Vector Machines, Random Forest, K-Means): Focus on mathematical concepts, hyperparameters, and when to use the algorithm
- **Frameworks** (e.g., PyTorch, TensorFlow, scikit-learn): Focus on practical usage, installation, and how to use the framework's APIs

## README Generation Guidelines for Machine Learning Algorithms

When generating README files for **individual machine learning algorithms** in this repository, ensure each algorithm documentation includes the following comprehensive sections:

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

## README Generation Guidelines for Machine Learning Frameworks

When generating README files for **machine learning frameworks** (such as PyTorch, TensorFlow, Lightning, scikit-learn), use a different approach focused on practical usage and framework capabilities:

### Framework Documentation Structure

#### 1. Installation and Setup
- **Installation commands**: pip/conda installation with GPU support options
- **Verification**: How to verify the installation works correctly
- **Basic imports**: Essential imports needed to get started

#### 2. Core Framework Concepts
- **Basic operations**: Fundamental operations (tensors, data structures, etc.)
- **Key abstractions**: Main classes and concepts (e.g., nn.Module, DataLoader)
- **Simple examples**: Getting started with basic usage

#### 3. Data Handling
- **Data loading**: How to load and preprocess data using framework tools
- **Data pipelines**: Framework-specific data pipeline patterns
- **Built-in datasets**: Using framework's built-in datasets

#### 4. Model Building
- **Model definition**: How to define models using the framework's APIs
- **Layer types**: Available layers and components
- **Model compilation**: How to configure models for training

#### 5. Training and Optimization
- **Training loops**: Framework's training patterns and best practices
- **Optimizers**: Available optimizers and how to configure them
- **Loss functions**: Built-in loss functions and custom losses

#### 6. Advanced Features
- **Multi-GPU/distributed training**: Scaling to multiple devices
- **Callbacks/hooks**: Extending functionality with framework-specific features
- **Custom components**: Creating custom layers, losses, or training logic

#### 7. Model Evaluation and Metrics
- **Evaluation patterns**: How to evaluate models using framework tools
- **Built-in metrics**: Available metrics and how to use them
- **Visualization**: Integration with visualization tools

#### 8. Model Deployment
- **Saving/loading**: Model serialization and persistence
- **Export formats**: Converting models for deployment (ONNX, TorchScript, etc.)
- **Production inference**: Optimizing models for production use

#### 9. Ecosystem Integration
- **Related libraries**: How the framework integrates with other tools
- **Extensions**: Popular extensions and add-ons
- **Community resources**: Where to find additional help and resources

### Framework Documentation Template

```markdown
# [Framework Name] Quick Reference

[Brief description of the framework and its main use cases]

### Installation
```bash
# Installation commands with different options
```

### Importing [Framework]
```python
# Essential imports
```

* * * * *

## 1. [Core Concept 1]
```python
# Practical examples showing basic usage
```

## 2. [Core Concept 2]
```python
# More examples building on previous concepts
```

[Continue with 8-12 sections covering all major framework capabilities]

* * * * *

Summary
=======

- **Key feature 1** brief description
- **Key feature 2** brief description
- **Key feature 3** brief description
[List main framework strengths and capabilities]
```

### Key Differences from Algorithm Documentation

**Algorithm READMEs focus on:**
- Mathematical concepts and theory
- Hyperparameters and their effects
- When to use the algorithm vs alternatives
- Statistical assumptions and requirements
- Performance characteristics and complexity

**Framework READMEs focus on:**
- Practical usage and code examples
- Installation and setup procedures
- Framework-specific APIs and patterns
- Integration with other tools and libraries
- Production deployment considerations

### Framework Documentation Quality Standards

- **Practical focus**: Emphasize how to use the framework, not theoretical concepts
- **Complete examples**: Include working code examples for all major features
- **Progressive complexity**: Start simple and build to advanced features
- **Current best practices**: Use modern, recommended patterns and APIs
- **Cross-references**: Link to official documentation and related frameworks

### Examples of Framework vs Algorithm Documentation

**Algorithm Example (SVM):**
- Focuses on margin maximization, kernel trick, hyperparameters (C, gamma)
- Explains mathematical foundations and when to use vs other classifiers
- Discusses computational complexity and assumptions about data

**Framework Example (PyTorch):**
- Focuses on tensors, autograd, nn.Module, DataLoader usage
- Shows practical code for building, training, and deploying models
- Demonstrates framework-specific features like dynamic graphs and TorchScript

---

**Important**: Always identify whether you're documenting an **algorithm** or a **framework** before choosing the appropriate documentation pattern. This ensures consistency and provides users with the most relevant information for their needs.
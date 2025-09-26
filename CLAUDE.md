# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

AI-From-Scratch is a comprehensive educational repository containing quick references, implementations, and guides for artificial intelligence, machine learning, and data science technologies. The repository serves as a structured learning path from fundamental concepts to advanced AI applications.

## Repository Structure

This repository follows a clear hierarchical organization with two main types of content:

### Algorithm Documentation (machine-learning/algorithms/, computer-vision/ algorithms)
- Mathematical concepts, hyperparameters, and when to use specific algorithms
- Focus on theoretical understanding and practical application guidelines
- Each algorithm includes comprehensive examples with step-by-step explanations

### Framework Documentation (deep-learning/frameworks/, data-processing/, etc.)
- Practical usage guides for ML/AI frameworks and libraries
- Installation instructions, API usage, and integration examples
- Focus on implementation patterns and best practices

## Key Architecture Patterns

### Documentation Standards
The repository follows specific documentation guidelines defined in the existing CLAUDE.md:

**For ML Algorithms**: Include 9 comprehensive sections covering algorithm overview, applicability, strengths/weaknesses, hyperparameters, assumptions, performance characteristics, evaluation methods, practical usage, and complete examples.

**For Frameworks**: Focus on installation, core concepts, data handling, model building, training, advanced features, evaluation, deployment, and ecosystem integration.

### Code Organization
- **Self-contained examples**: Most directories contain standalone Python scripts and Jupyter notebooks
- **Practical implementations**: Real-world examples like the self-driving car system (`computer-vision/yolo/self_driving_car.py`)
- **Mixed framework usage**: Code uses various frameworks (TensorFlow, PyTorch, OpenCV, scikit-learn) depending on the specific technology being demonstrated

### File Structure Patterns
- Each topic directory contains a README.md with comprehensive quick reference
- Python files contain complete, runnable examples with detailed comments
- Jupyter notebooks provide interactive exploration of concepts
- No centralized dependency management - each example is self-contained

## Development Guidelines

### Creating New Documentation
When adding new algorithm or framework documentation:

1. **Identify the type**: Determine if you're documenting an algorithm (mathematical concept) or framework (practical tool)
2. **Follow the appropriate template**: Use the algorithm template for ML concepts, framework template for libraries/tools
3. **Include complete examples**: All code should be runnable and well-commented
4. **Add step-by-step explanations**: Each code block should explain "what's happening" and "why this step"

### Code Style
- Include detailed comments explaining each major step
- Use meaningful variable names
- Add docstrings for functions and classes
- Include import statements in examples
- Provide both theoretical explanation and practical implementation

### Content Standards
- All examples should be educational and self-contained
- Focus on practical applicability over theoretical depth
- Include performance considerations and best practices
- Provide real-world context and use cases

## Common Patterns in the Codebase

### Example Structure
Most implementations follow this pattern:
1. Imports and setup
2. Data preparation/loading
3. Model/algorithm configuration
4. Training/processing
5. Evaluation and visualization
6. Practical usage functions

### Educational Focus
- Code includes extensive comments explaining concepts
- Examples progress from simple to complex
- Real-world applications are provided (e.g., self-driving car, image generation)
- Multiple approaches are often shown for comparison

## File Types and Usage

- **Python scripts (.py)**: Complete implementations with example usage
- **Jupyter notebooks (.ipynb)**: Interactive exploration and experimentation
- **README.md files**: Comprehensive quick reference guides
- **No test files**: This is an educational repository focused on examples rather than production code

## Important Notes

- This repository prioritizes educational clarity over production-ready code
- Examples are designed to be self-contained and immediately runnable
- The focus is on understanding concepts through practical implementation
- Documentation should be accessible to beginners but detailed enough for practitioners
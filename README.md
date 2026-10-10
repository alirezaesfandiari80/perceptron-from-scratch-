# Perceptron from Scratch

## Overview

This project explores the Perceptron, a classic supervised learning algorithm for binary classification, through a from-scratch implementation.

## Theoretical Background

### 1. Linear Decision Function

For an input vector `x`, the Perceptron computes:

$$
z = w^T x + b
$$

Here, `w` is the weight vector and `b` is the bias.

For labels encoded as `-1` and `+1`, a standard prediction rule is:

$$
\hat{y} =
\begin{cases}
+1 & \text{if } w^T x + b \geq 0 \\
-1 & \text{otherwise}
\end{cases}
$$

The decision boundary is the hyperplane:

$$
w^T x + b = 0
$$

**Note:** This is the standard formulation. Check that the notebook uses the same label encoding and prediction convention.

### 2. Perceptron Learning Rule

For a misclassified example `(x_i, y_i)`, where `y_i` is either `-1` or `+1`, the standard update is:

$$
w \leftarrow w + \eta y_i x_i
$$

$$
b \leftarrow b + \eta y_i
$$

Here, `η > 0` is the learning rate. The update changes the decision boundary based on the misclassified example.

### 3. Convergence and Limitations

Under the standard assumptions, the Perceptron Convergence Theorem guarantees a finite number of updates when the training data are linearly separable.

If the data are not linearly separable, the algorithm is not guaranteed to converge to a perfect classifier.

## What This Project Explores

- The mechanics of a linear binary classifier
- Updating weights and bias using training examples
- Learning and visualizing a linear decision boundary

## How to Run

1. Open the notebook in Jupyter Notebook or Google Colab.
2. Install the libraries imported by the notebook.
3. Run the cells in order.

## Learning Objective

The goal is to understand how a linear classifier learns from labeled examples and how parameter updates affect its decision boundary.

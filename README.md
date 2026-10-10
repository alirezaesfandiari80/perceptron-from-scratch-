# Perceptron from Scratch

## Overview

This project explores the Perceptron, a classic supervised learning algorithm for binary classification, through a from-scratch implementation.

## Theoretical Background

### 1. Linear decision function

Given an input vector \(x \in \mathbb{R}^d\), the Perceptron computes

\[
z = w^\top x + b,
\]

where \(w\) is the weight vector and \(b\) is the bias.

For labels encoded as \(\{-1,+1\}\), a common prediction rule is

\[
\hat y =
\begin{cases}
+1 & \text{if } w^\top x+b \geq 0,\\
-1 & \text{otherwise}.
\end{cases}
\]

The decision boundary is the hyperplane \(w^\top x+b=0\).  
**Note:** The prediction rule and label encoding above describe the standard formulation. Check the notebook to confirm its exact convention.

### 2. Perceptron learning rule

For a misclassified training example \((x_i,y_i)\), with \(y_i\in\{-1,+1\}\), the standard update is

\[
w \leftarrow w+\eta y_i x_i,\qquad
b \leftarrow b+\eta y_i,
\]

where \(\eta>0\) is the learning rate. The update moves the decision boundary to improve classification of the misclassified example.

### 3. Convergence and limitations

The Perceptron Convergence Theorem states that the standard algorithm makes a finite number of updates and converges to a separating hyperplane when the training data are linearly separable, under the theorem's assumptions.

If the data are not linearly separable, the algorithm is not guaranteed to converge to a perfect classifier.

## What This Project Explores

- The mechanics of a linear binary classifier
- Updating model parameters from training examples
- Learning a linear decision boundary
- Visualizing classification behavior

## Notebook

Open the Jupyter Notebook in this repository and run its cells in order. If your notebook has a different filename, update this section to match it.

## Requirements

Check the notebook imports for the exact dependencies. A typical implementation may use Python, NumPy, and Matplotlib.

## Learning Objective

The goal is to understand how a simple linear classifier learns from labeled examples and how its parameter updates affect the decision boundary.

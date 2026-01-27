# Linear Regression — Mathematical Foundations (From First Principles)

This document explains **Simple Linear Regression** step by step, with intuition, math, and clear links to implementation. The goal is not memorization, but **understanding why the algorithm works**.

---

## 1. Problem Setting

We are given a dataset:

$`
{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)}
`$

where:

* $`(x_i \in \mathbb{R})`$ is a single input feature
* $`(y_i \in \mathbb{R})`$ is the target value

Our task is to learn a function that maps inputs to outputs **as accurately as possible**.

---

## 2. Model Assumption

We assume a **linear relationship** between input and output:

$`
\hat{y} = wx + b
`$

Where:

* (w) = slope (weight)
* (b) = intercept (bias)

> This does **not** mean the data must be linear — it means the model is linear **in its parameters**.

---

## 3. What Does “Best Fit” Mean?

We need a way to measure how bad a prediction is.

### Error for one data point

$`
\text{error}_i = y_i - \hat{y}_i = y_i - (wx_i + b)
`$

Errors can be positive or negative, so we square them.

---

## 4. Loss Function (Mean Squared Error)

We define the loss over the full dataset as:

$`
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2
`$

Why MSE?

* Penalizes large errors heavily
* Differentiable (important for optimization)
* Convex for linear regression

Because the loss is **convex**, it has **one global minimum**.

---

## 5. Optimization Objective

Our goal:

$`
\min_{w, b} J(w, b)
`$

We want the values of (w) and (b) that minimize the loss.

---

## 6. Gradient Descent (Core Idea)

Instead of guessing parameters, we **iteratively improve** them.

Gradient descent updates parameters in the direction of **steepest decrease** of the loss.

---

## 7. Computing the Gradients

### Partial derivative with respect to (w):

$`
\frac{\partial J}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)
`$

### Partial derivative with respect to (b):

$`
\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)
`$

These gradients tell us:

* How sensitive the loss is to (w)
* How sensitive the loss is to (b)

---

## 8. Parameter Update Rules

Using learning rate (\alpha):

$`
w := w - \alpha \frac{\partial J}{\partial w}
`$

$`
b := b - \alpha \frac{\partial J}{\partial b}
`$

Interpretation:

* $`Large (\alpha)`$ → fast but unstable
* $`Small (\alpha)`$ → slow but stable

---

## 9. Convergence

Because the loss function is convex:

* Gradient descent **will converge**
* Final parameters correspond to the global minimum

Stopping criteria:

* Max iterations reached
* Loss change below threshold

---

## 10. Closed-Form Solution (Normal Equation)

Linear regression also has an analytical solution:

$`
w = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
`$

$`
b = \bar{y} - w\bar{x}
`$

Scikit-learn uses optimized solvers based on this idea.

Gradient descent is preferred when:

* Data is large
* Model extends beyond simple regression

---

## 11. Prediction

Once trained:

$`
\hat{y} = wx + b
`$

This is used during inference.

---

## 12. Model Evaluation (R² Score)

We evaluate model quality using **coefficient of determination**:

$`
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
`$

Interpretation:

* $`(R^2 = 1)`$ → perfect prediction
* $`(R^2 = 0)`$ → same as predicting mean
* $`(R^2 < 0)`$ → worse than mean

---

## 13. Assumptions of Linear Regression

* Linearity
* Independence of errors
* Homoscedasticity
* No strong outliers

Violating these assumptions degrades performance.

---

## 14. Failure Cases

Linear regression fails when:

* Relationship is highly non-linear
* Strong multicollinearity (in multivariate case)
* Outliers dominate the loss

---

## 15. Key Takeaway

Linear regression is not "simple" because it is weak — it is simple because it is **mathematically elegant**.

Understanding this model deeply provides:

* Foundation for all supervised learning
* Insight into optimization
* Intuition for bias–variance tradeoff

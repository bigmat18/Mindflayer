---
Data: 2025-11-14T14:40:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Introduction to Machine Learning]]"
Area: "[[Master's degree]]"
---
# Learning Algorithms

A learning algorithm is based on [[Data in ML|Data]], [[Tasks in ML|Task]] and [[Models in ML|Model]]. We use a **heuristic** that means search through the hypothesis space **H** of the **best hypothesis**
- ie the best approximation to the (unknown) target function
- typically searching for the $h$ with the minimum error
- e.g. free parameters of the model are fitted to the task at hand
- examples: best $w$ in linear models, best rules for symbolic models, ...

**H** may not coincide with the set of all possible functions and the search can not be exhaustive: we need to make assumption that we call **inductive bias**

![[Screenshot 2025-11-14 at 12.12.24.png | 500]]


We are going to introduce 2 Learning Algorithms for the [[Regression Models]] and for the [[Classification Models]] task using a linear model both based on LMS.
1. A direct approach based on **normal equation** solution
2. An iterative approach based on **gradient descent**

We start **redefining the learning problem** and the loss for them (for l data and multidimensional inputs)

### The learning problem: [[Classification Models|Classification Tasks]]
- **Given** a set $l$ training examples $(x_p, y_p)$ and loss function (measure) L $y_p = \{0,1\}$ or $y_p = \{-1, +1\}$
- **Find**: the weight vector $w$ that minimizes the expected loss on the traingin data.
$$
R_{emp} = \frac{1}{l}\sum_{p=1}^l L(h(x_p), y_p)
$$
**For classification**: Using a piecewise constant (over $sign(w^Tx)$ ) for the loss can make this a difficult problem. Assume we still use the [[Least Squares]] (as for the regression case)
$$
E(w) = \sum_{p=1}^l (y_p - x_p^T w)^2 = \sum_{p=1}^l (y_p - w^T x_p)^2
$$
Initially, we can make the optimization problem easier by replacing the original objective function L (0/1 loss) by a **smooth, differentiable function**. For example, consider the popular mean squared error (MSE loss).

![[Pasted image 20260406124051.png]]

Find optimal values for $w$ (for fitting of training (TR) data) by least squares:
- **Given**: a set of $l$ training examples $(x_p, y_p)$, find $w$ to minimize the residual sum of squares:
$$
E(w) = \sum_{p=1}^l (y_p - x_p^T w)^2 = ||y - Xw||^2
$$
Where $x_p$ is p-th input vector, $y_p$ the output for $p, w$ free par., l num. of examples, n input dim.

Min error: 
- if $y_p = 1$ then $x_i^T w$ go toward -> no class Error
- if $y_p = -1$ then $x_p^Tw$ go toward $-1$ -> no class Error

**Note**: in $E(w)$ we do not use $h(x)$ as for regression, to hold a continuos **differentiable** loss. (because $h(w) = sign(w^Tw)$ for classification)
- This is a quadrati function -> minimum always exists (but may be not unique)
- $X$ is a matrix $l$ $x$ $n$ with a row is used of regression problem

**Note**: The same approash is used for [[Regression Models]] problems

### [[Learning Algorithms using Normal Equation]]

### [[Learning Algorithms using Gradient Descent]]

# References
---
Data: 2026-04-06T11:29:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Linear Models]]"
Area: "[[Master's degree]]"
---
# Introduction Linear Model
The linear model has been the mainstay of statistics:

> Despite the great inroads made by modern nonparametric regression
> techniques, linear models remain important, and so we need to understand
> them well

### Notation for Multidimensional Inputs
Assuming column vector for $x$ and $w$ (in bold). Number of data $l$, dimension of input vector $n, y_p$ (targets), $p=1, \dots, l$ 
$$
w^Tx + w_0 = w_0 + w_1x_1 + w_2 x_2 + \dots + w_n x_n = w_0 + \sum_{i=1}^n w_i x_i
$$
Note that sometimes (in NN) the transponse notations $T$ in $w^T$ is omitted. $w_0$ is the intercept, threshold, bias, offset ...

Often it is convenient to include the constant $x_0=1$ so that we can write the equation above with:
$$
w^T x = x^Tw \:\:\:\:\: x^T = [1, x_1, x_2, \dots, x_n] \:\:\:\:\: w^T= [w_0, w_1, w_2, \dots, w_n]
$$
So, the “linear “model can be written as a function that for each $x_p$ compute:
$$
h(x_p) = x_p^T w = \sum_{i=0}^n x_{p,i}w_i
$$
$w$: continues (free) parameters are called weights

### [[Regression Models]]

### [[Classification Models]]

### Good or bad approximation?
Possible scenarios (we know the true target function):
- **Scenario 1**: The data in each class are generated a Gaussian distribution with uncorrelated components, same variances, and different means.
	- In this scenario the linear regression rule is almost **optimal** (is the best on can do). The region of overlap is inevitable (due to errors in the input data)
- **Scenarion 2**: The data in each class are generated from a mixture of 10 gaussians in each class.
	- fot this scenario it is fat too rigid

### Inductive Bias 
- **Language bias**: the H is a set of linear functions (may be very restrictive and rigid)
- **Search bias**: ordered search guided by the Least Square minimization goal
	- For instance, we could prefer a different method to obtain a restriction on the values of parameters, achieving a different solutions with other properties (in particular to consider the generalization issue), ...

It shows that even for a “simple” model there are many possibilities. We need a principled approach! (see theory of ML)…

### Limitations
In geometry, two set of points in a two-dimensional plot are **linearly separable** when the two sets of points can be completely separated by a single line. 

In general, two groups are linearly separable in n-dimensional space if they can be separated by an (n − 1)-dimensional hyperplane.

![[Pasted image 20260406230714.png | 350]]

The linear decision boundary can provide exact solutions only for linearly separable sets of point. 
##### Example: Conjuctions
We can represent conjunctions by the linear models, e.g.: **Conjuctions**

![[Pasted image 20260406230812.png]]

##### Example: Classification tasks
Given 3 points, can we always find a separation plane for every assignment of $f(x)$ ?
- No, 3 aligned points with 0 in the middle and others 1; yes if they are not aligned (existence!).

![[Pasted image 20260406230924.png]]

Given 4 points, can we always find a separation plane for every assignment of $f(x)$ ?
- No (XOR). We can find a labeling such that the linear classifier fails to be perfect.

![[Pasted image 20260406231026.png | 350]]


# References
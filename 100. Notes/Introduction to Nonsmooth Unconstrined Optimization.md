---
Data: 2026-05-04T23:59:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Introduction to Nonsmooth Unconstrined Optimization

Convex nondifferentiable (or "nonsmooth") optimization deals with minimizing functions that have "kinky" points, where the classic derivative is undefined. This discipline is not just a theoretical whim, but a practical necessity driven by fundamental problems, especially in the field of Machine Learning (ML).

### 1. Motivation I: Incremental (Stochastic) Gradient in Machine Learning

The first reason we encounter problems akin to nonsmooth optimization stems from the very nature of training models on massive amounts of data.

**The Fitting Problem**

In supervised Machine Learning, we start with a dataset:

- **Inputs**: $X=[x^{i}\in\mathbb{R}^{h}]_{i\in I}$ where $I=\{1,...,m\}$ is the set of $m$ samples.
    
- **Outputs**: $y=[y^{i}\in\mathbb{R}^{k}]_{i\in I}$.
    

The goal is to train a predictor $\pi(x;w):\mathbb{R}^{h}\rightarrow\mathbb{R}^{k}$, parameterized by a weight vector $w\in\mathbb{R}^{n}$, so that the error made on the data is as small as possible. This is formulated by minimizing a global loss function $f(w)$:

$$min\{f(w)=\sum_{i\in I}[f^{i}(w)=\mathcal{L}(y^{i},\pi(x^{i};w))]:w\in\mathbb{R}^{n}\}$$

.

**The Computational Cost of the Exact Gradient** To find the minimum using classic gradient methods, we must compute the derivative of $f(w)$, which is the sum of the gradients for each individual data point: $\nabla f(w)=\sum_{i\in I}\nabla f^{i}(w)$.

- _Example (Linear Least Squares)_: If we use a linear model $\pi(x;w)=\langle x,w\rangle$ and a squared loss $\mathcal{L}=(y-z)^{2}/2$, the function on a single data point is $f^{i}(w)=(y^{i}-\langle x^{i},w\rangle)^{2}/2$, and its gradient is $\nabla f^{i}(w)=-x^{i}(y^{i}-\langle x^{i},w\rangle)$. Although computing each individual $\nabla f^{i}$ is cheap, in real-world scenarios the number of samples $m$ is so large that computing the "full" gradient at every iteration is already too costly.
    

**The Solution: Stochastic Gradient Descent (SGD)** The intuition behind SGD is that, if the data is independent and identically distributed (i.i.d.), many of the gradients will "cancel out", making it redundant to compute all of them. A small sample is enough to compute a close approximation to the "true" gradient.

- By taking a "small" subset $K\subset I$, we define the **incremental gradient**: $\nabla f^{K}(w)=\sum_{i\in K}\nabla f^{i}(w)$.
    
- It is called a _batch_ if $K=I$, a _mini batch_ if $\#K<m$ (often a single data point is used, $\#K=1$), and an _on-line_ version if observations keep coming fast and have to be used immediately one by one and then discarded (no memory).
    

**The Link to Nonsmooth Optimization** The use of the incremental gradient creates a chaotic dynamic. The negative direction $-\nabla f^{K}$ computed on a subset **is not a descent direction** for the global function $f(w)$. A different mathematical analysis is needed, with results often given in terms of expected value $\mathbb{E}(\cdot)$ and the "mean of iterates" (the Cesáro average $\overline{x}^{i}=(\sum_{k=0}^{i}x^{k})/i$). The convergence rates for SGD with $\#K=1$ are rather worse than in the deterministic case: for example, to guarantee an error $\mathbb{E}(f(\overline{x}^{i})-f_{*})\le\epsilon$ on a smooth ($C^{1}$) and convex function, you need a number of iterations $i\ge O(1/\epsilon^{2})$. This intrinsic slowness exactly mirrors the difficulties encountered when analyzing nondifferentiable functions.

---

### 2. Motivation II: Nondifferentiable Regularization (Lasso)

The second motivation arises from the mathematical architecture of predictive models. A good model must not only minimize the error on the training data but must also "generalize" well on unseen data. To achieve this, ML experts add a regularization term $\Omega(w)$ weighted by a hyper-parameter $\mu$:

$$min\{\sum_{i\in I}\mathcal{L}(y^{i},\pi(x^{i};w))+\mu\Omega(w):w\in\mathbb{R}^{n}\}$$

.

**Feature Selection and the L1 Norm**

- **Ridge Regularization (L2 Norm)**: This is the standard choice, defined as $\Omega(w)=||w||_{2}^{2}/2$. It is a "smooth" and infinitely differentiable function ($\in C^{\infty}$), with a simple gradient $\nabla\Omega(w)=w$.
    
- **Feature Selection (L0 Norm)**: Another way to simplify a model is to decrease $n$, completely "turning off" irrelevant variables. One could try using the L0 norm (which counts the number of non-zero parameters, $\Omega=||\cdot||_{0}$), but it is a very nasty function for optimization: it is discontinuous ($\notin C^{0}$) and can be written as a complex Mixed-Integer Nonlinear Problem.
    
- **Lasso (L1 Norm)**: A workable alternative is to use the Lasso, $\Omega=||\cdot||_{1}$, which is the best convex approximation of the L0 norm. It increases sparsity in practice, is convex, and is continuous ($\in C^{0}$), but it is **nondifferentiable** ($\notin C^{1}$) at zero.
    

---

### 3. Why classic smooth methods fail miserably

What happens if we try to use a standard smooth method on a function regularized with L1? The text illustrates a specific example:

$$f(w_{1},w_{2})=(3w_{1}+2w_{2}-2)^{2} +10(|w_{1}|+|w_{2}|)$$

(with $\mu=10$ and data points set at $x^{1}=[3,2]$ and $y^{1}=2$).

**The Problem of "Kinky" Points** Because of the absolute value, wherever $w_{1}=0$ or $w_{2}=0$, the sublevel sets of the function $S(f, \cdot)$ are "kinky" (they have sharp edges). At these exact points, the derivative of the absolute value $[|\cdot|]'(0)$ is undefined: mathematically, it could be -1, 1, or 0.

If an optimization algorithm lands exactly on one of these kinks and has to "choose arbitrarily" a value for the derivative, there is a high probability that the computed gradient will generate a direction of movement $-g$ that **points outside** the sublevel set $S(f, f(w))$.

In other words, instead of pointing "downhill" toward the minimum ($\exists(-)g\approx\nabla f(w)$ "pointing inside $S(f,f(w))$"), many other directions point uphill (meaning there is no descent direction). Faced with a direction that increases the error, a descent method will refuse the movement (setting the step size $\alpha^{i}=0$). The result is that the algorithm completely stops ($w^{i+1}=w^{i}$).

This unequivocally demonstrates that **methods need not be of descent** when dealing with these scenarios, and entirely new mathematical approaches are needed to handle nonsmooth functions effectively.
# References
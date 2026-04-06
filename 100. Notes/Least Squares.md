---
Data: 2026-04-05T11:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Least Squares
**The abstract goal is:** given vectors $a_{1},a_{2},...,a_{n}\in\mathbb{R}^{m}$ and a "target vector" $y\in\mathbb{R}^{m}$, we look for coefficients $x_{1},x_{2},...,x_{n}$ such that:
$$a_{1}x_{1}+\dots+a_{n}x_{n}=y$$
###### Example
A certain food is a mixture of ingredient A, which contains 10 grams of sugars, 20 of protein and 3 of fats, and ingredient B, which contains 5 grams of sugars, 1 of protein and 1 of fats. A lab analysis reveals that the mixture contains 40 grams of sugars, 30 grams of protein and 20 grams of fats. What is the amount of each ingredient?

$$\begin{bmatrix}10\\ 20\\ 3\end{bmatrix}x_{1}+\begin{bmatrix}5\\ 1\\ 1\end{bmatrix}x_{2}=\begin{bmatrix}40\\ 30\\ 20\end{bmatrix}$$

### Solvability
We can write the system as:
$$Ax=y \text{ where } A=[\begin{matrix}a_{1}&a_{2}&...&a_{n}\end{matrix}]$$
The system is solvable for each $y$ when we have $n=m$ **linearly independent vectors** (invertibility, from linear algebra).

This is not always the case: sometimes the vectors are too few, or they are not linearly independent.

###### Example of non-solvability
The system $\begin{bmatrix}2\\ 1\\ 0\end{bmatrix}x_{1}+\begin{bmatrix}1\\ 3\\ 0\end{bmatrix}x_{2}=\begin{bmatrix}5\\ 5\\ 1\end{bmatrix}$ is not solvable. Geometric interpretation: the target vector lies outside the spanning plane. Not even if I add $\begin{bmatrix}4\\ 3\\ 0\end{bmatrix}$, $\begin{bmatrix}12\\ -8\\ 0\end{bmatrix}$ ...

### Linear Least Squares Problems
When a system cannot be solved exactly, we ask: what is the closest I can get? Even if I cannot get $\begin{bmatrix}5\\ 5\\ 1\end{bmatrix}$ maybe I can get $\begin{bmatrix}5\\ 5\\ 0\end{bmatrix}$...
$$\min_{x\in\mathbb{R}^{n}}||Ax-y||$$

Here, we use the Euclidean norm: $||v||_{2}=v_{1}^{2}+v_{2}^{2}+\dots+v_{n}^{2}$.

**Geometric interpretation:** We seek the closest vector to $y$ inside the hyperplane $Im(A)$, which is obtained by **orthogonal** projection.

Obstructions are not always visible; for instance, all columns of $A$ may have a zero sum instead of a zero component.

### Matlab Implementation
Matlab provides two division operators to solve these problems:
- **Forward slash (`/`):** `5/2` yields `2.5000e+00`.
- **Backslash (`\`):** `5\2` yields `4.0000e-01`.
    
**Mnemonic**: One divides the number above the bar by the number below.

##### Solving Systems and Least Squares
`A \ y`: Finds the vector $x$ such that $Ax=y$.

It is functionally equivalent to $A^{-1}y$, but implemented using faster and more stable methods than `inv(A)*y`.

There is also `X/A`, which computes $XA^{-1}$, when the product makes sense, e.g., when $X=v^{\top}$ is a row vector.

The same operators solve linear systems:
```
>> [1 2; 3 4] \ [5; 6]
ans =
   -4.0000e+00
    4.5000e+00
```

The backslash operator also solves least squares problems, such as:
```
>> [2 1; 1 3; 0 0] \ [5; 5; 1]
ans =
    2.0000e+00
    1.0000e+00
```

##### Applications
###### 0. Linear Regression in machine learning
Apart from notation change,
$$\min_{x}||Ax-y||^{2}\iff \min_{w}||Xw-y||^{2}$$

###### 1. Salary Estimation
Contains number of points made, rebounds taken, fouls committed by 399 NBA players in season 2015-2016, and the salaries they earn. (Source: basketball-reference.com)

Is it true that the best-performing players are paid more? Which of these statistics has a larger impact?

Using NBA player data, we can create a linear model to estimate salary:

$$(salary)\approx(rebounds)x_{1}+(fouls)x_{2}+(points)x_{3}$$

We minimize the sum of squared differences for all players:

$$\sum_{p\in players}(x_{1}(rebounds)_{p}+x_{2}(fouls)_{p}+x_{3}(points)_{p}-(salary)_{p})^{2}$$

Our intuition suggests that $x_{1}$ and $x_{3}$ should be positive, and $x_{2}$ may be negative.

```
% separator: ','; skip 1 row, 1 column.
>> M = dlmread('salaries.csv', ',', 1, 1)
>> A = M(:, 1:3);
>> y = M(:, 4);
>> x = A \ y
ans =
	1.3285e+04
	-2.6578e+04
	9.5162e+03
>> [value, location] = min(A*x-y)
value = -1.8864e+07
location = 271
```

Player #271 is paid 18M$ more than he would deserve...

###### 2. Polynomial Fitting
Given pairs $(x_{i},y_{j})$, we can recover unknown coefficients $a, b, c, d$ for a polynomial $ax_{i}^{3}+bx_{i}^{2}+cx_{i}+d$. This is a linear problem where $a, b, c, d$ are the unknowns:
$$\min \left\| \begin{bmatrix} x_1^3 & x_1^2 & x_1 & 1 \\ \vdots & \vdots & \vdots & \vdots \\ x_m^3 & x_m^2 & x_m & 1 \end{bmatrix} \begin{bmatrix} a \\ b \\ c \\ d \end{bmatrix} - \begin{bmatrix} y_1 \\ \vdots \\ y_m \end{bmatrix} \right\|$$

The last column of ones represents the bias in machine learning terms.

```
% 1000 random points in [-10, 10], sorted
>> x = sort(20*rand(1000,1) - 10);
% degree-3 polynomial plus random noise
>> y = 0.02*x.^3 - x + 1 + randn(1000,1);
>> plot(x, y)
```

This is not too different from the values we started with; and actually these numbers give a lower error than the ones we used to construct the example, $\begin{bmatrix}0.02 & 0 & -1 & 1\end{bmatrix}$

```
>> A = [x.^3, x.^2, x, ones(size(x))];
>> p = A \ y
p =
   1.9842e-02
  -5.9348e-04
  -9.9320e-01
   1.0230e+00
>> plot(x, y, x, A*p)
```

Now with 100x as much noise... General idea: the signal-to-noise ratio is related to the accuracy we can get.

```
>> y = 0.02*x.^3 - x + 1 + 100*randn(1000,1);
>> p = A \ y
p =
   1.5762e-03
   5.1916e-02
   4.7983e-01
  -7.1315e+00
>> plot(x, y, x, A*p)
```

### The statistics behind it
Statistical problem: given observations $y_i$ , what are the values of $a, b, c, d$ that ‘most likely’ produced it?

If noise = random Gaussian with same variance for each $i$, ‘most likely’ (maximum likelihood) means minimizing$$\sum_{i=1}^m (ax_i^3 + bx_i^2 + cx_i + d - y_i)^2$$i.e., the squared Euclidean norm. 

Remark This works because the variance of the added noise is the same on each entry. If they are different, e.g.,

```
>> y(1) = 0.02*x(1)^3 - x(1) + 1 + randn();
>> y(2) = 0.02*x(2)^3 - x(2) + 1 + 5*randn();
```

we should rescale rows to have more accuracy. (Ask a statistician for more detail.)


### Solvability of least squares problems
- **Linear systems:** $Ax=y$ with $A$ square: **unique solution if $A$ is nonsingular.** 
- **Linear least squares problems:** $\min||Ax-y||$ with $A$ tall thin: unique solution if...?.
###### Example:
$$\min||Ax-y|| \text{, } A=\begin{bmatrix}1&-1&0\\ 2&1&3\\ 1&0&1\\ 0&0&0\end{bmatrix} \:\:\:\:\:\: y=\begin{bmatrix}0\\ 3\\ 1\\ 2\end{bmatrix}$$
**Solution**: We can 'match' the first three entries (but not the 4th). $x=\begin{bmatrix}0\\ 0\\ 1\end{bmatrix}$ solves the problem. But also $x=\begin{bmatrix}1\\ 1\\ 0\end{bmatrix}$ or $x=\begin{bmatrix}\frac{1}{8}\\ \frac{1}{8}\\ \frac{1}{2}\end{bmatrix}...$.

### Full column rank definition
What is going on: there is a vector $z\ne0$ n $\ker A$: $A\begin{bmatrix}1\\ 1\\ -1\end{bmatrix}=0$. If $x$ is a solution, then so is $x+z, x+2z, x-37z...$.

> **Definition** We say that $A\in\mathbb{R}^{m\times n}$ has **full column rank** if $\ker A=\{0\}$, or, equivalently: $\text{rank } A=n$ or, equivalently: there is no $z\in\mathbb{R}^{n}$, $z\ne0$ such that $Az=0$.

We shall see, via several equivalent conditions, that the least squares problem $\min||Ax-y||$ has a unique solution if and only if $A$ has full column rank.

### Criterion for full column rank

> **Theorem**
> $A$ has full column rank if and only if $A^{T}A$ is positive definite.

We already saw (lecture on orthogonal matrices) that $A^{T}A$ is symmetric and positive semidefinite. For each $z\ne0$, $z^{T}A^{T}Az=||Az||^{2}\ge0$. 

> **Proof:** $A$ full column rank $\iff Az\ne0$ for all $z\ne0$ $\iff z^{T}A^{T}Az=||Az||^{2}\ne0$ for all $z\ne0$.

We can test the matrix from our earlier example, using eigenvalues.

```
>> A = [1 -1 0; 2 1 3; 1 0 1; 0 0 0];
>> eig(A’*A)
ans =
	2.6232e-16
	2.0718e+00
	1.5928e+01
```

### Least squares problems - solution
Suppose $A$ has full column rank. Then $\min||Ax-y||$ can also be written as:

$$\min_{x\in\mathbb{R}^{n}}\frac{1}{2}||Ax-y||^{2}=\min_{x\in\mathbb{R}^{n}}\frac{1}{2}(Ax-y)^{T}(Ax-y)$$
$$=\min_{x\in\mathbb{R}^{n}}\frac{1}{2}(x^{T}A^{T}Ax-y^{T}Ay-x^{T}A^{T}y+y^{T}y)$$
$$=\min_{x\in\mathbb{R}^{n}}\frac{1}{2}x^{T}A^{T}Ax-y^{T}Ax+\frac{1}{2}y^{T}y$$

We have transformed the problem into the one of finding the minimum of a quadratic function $f(x)$ - sounds familiar?

### Some optimization
$$\min_{x\in\mathbb{R}^{n}}\frac{1}{2}x^{T}A^{T}Ax-y^{T}Ax+\frac{1}{2}y^{T}y$$
- **Gradient:** $A^{T}Ax-A^{T}y$ 
- **Hessian**: $A^{T}A>0 \rightarrow \text{strictly convex!}$

The minimum exists unique, and can be found with
$$
0 = \text{gradient} = A^{T}Ax - A^{T}y
$$
or:$$A^{T}Ax=A^{T}y$$$A^{T}A$ is **square** invertible (because it's positive definite), so this linear system has a unique solution. Can be solved with many methods: Gaussian elimination, LU factorization, QR (you'll see it soon),....

### Computational cost
If done naively: (for $A\in\mathbb{R}^{m\times n}$, $m>n$, ignoring lower-order terms)

1. Computing $A^{T}A$: $2mn^{2}$.
2. Computing $A^{T}y$: $2mn$ (lower-order).
3. Solving $A^{T}Ax=A^{T}y$ with Gaussian elimination / LU factorization: $\frac{2}{3}n^{3}$.
    

- **Trick 1:** using symmetry, we can skip half of the entries of $A^{T}A$. 

- **Trick 2:** a better way to solve linear systems with posdef matrices, Cholesky factorization, $A^{T}A=R^{T}R$ (we'll see it later).
	1. Computing $A^{T}A$: $mn^{2}$.
	2. Computing $A^{T}y$: $2mn$ (lower-order).
	3. Solving $A^{T}Ax=A^{T}y$ with Cholesky: $\frac{1}{3}n^{3}$.
    

### Geometric idea
Can't solve $Ax=y$? Multiply both sides by $A^{T}$ and try again! 

**Geometric idea:** The residual $Ax-y$ is orthogonal to any vector $Av \in \text{span } A$: $(Av)^{T}(Ax-y)=0$. 

This method to solve LS problems is known as method of normal equations ('normal' is a fancy word for 'perpendicular/orthogonal').

### Pseudoinverse
We showed that the solution of $\min||Ax-y||$ is given by

$$x_{*}=(A^{T}A)^{-1}A^{T}y$$

(if $A$ has full column rank) .

> **Definition** The (Moore-Penrose) **pseudoinverse** of a matrix $A$ with full column rank is $A^{+}:=(A^{T}A)^{-1}A^{T}$.

So we can write $x=A^{+}y$ for the solution of a LS problem. This generalizes the concept of inverse $A^{-1}$ to a non-square $A$. 

**Non-obvious consequence:** the solution is always obtained by multiplying $y$ by a certain matrix. In particular, the solution of $\min||Ax-(y_{1}+y_{2})||$ is the sum of the two solutions of $\min||Ax_{1}-y_{1}||$ and $\min||Ax_{2}-y_{2}||$. Note that $A^{+}A=I_{n}$, but $AA^{+}\ne I_{m}$ (there is no matrix such that $AA^{+}=I_{m}$, for rank reasons).

### The other side
Sometimes in ML the same problem is formulated with multiplications on the other side: $w\in\mathbb{R}^{1\times n}$ row vector of unknown weights, $X\in\mathbb{R}^{n\times m}$ matrix with each "feature" as a **row**, $y\in\mathbb{R}^{1\times m}$ target (row) vector:
$$\min_{w}||wX-y||_{2}$$
This is the same problem, apart from notation. If $X\in\mathbb{R}^{n\times m}$ is short-fat ($n\le m$) with linearly independent **rows**, then its pseudoinverse is defined as
$$X^{+}=X^{T}(XX^{T})^{-1}$$
(Mnemonic: you must invert a matrix with the **small** dimension as its side) .

# References
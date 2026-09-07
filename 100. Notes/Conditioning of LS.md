---
Data: 2026-07-19T16:09:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Conditioning and Sensitivity
When solving least squares problems, different algorithms have varying computational costs depending on the dimensions of the matrix ($m$ rows, $n$ columns):

- For $m \approx n$: Normal equations require $\frac{4}{3}n^3$ operations, while QR and SVD require $\approx 13n^3$ operations.
- For $m \gg n$: Normal equations require $mn^2$ operations, while QR and SVD require $2mn^2$ operations.

TL;DR: Normal equations are faster than QR, which is faster than SVD.

**Sensitivity issues:** Despite being the fastest, normal equations can be much less accurate. When a matrix $A$ is at a distance of $10^{-8}$ from a non-full-rank matrix, the algorithm attempts to compute a solution for a matrix that is close to singular. In such cases, solving $x=(A'*A)\backslash(A'*y)$ yields a warning that the matrix is close to singular or badly scaled, resulting in an inaccurate $x_3$ solution. This highlights a common trend: problems close to unsolvable are numerically troublesome.

Furthermore, evaluating the residuals ($norm(A*x - y)$) doesn't tell us much about the true accuracy of these computed solutions. Even the placement of parentheses (order of operations) matters due to sensitivity issues, as seen when comparing the SVD solutions `x = V*(pinv(S)*(U'*y))` and `x = V*pinv(S)*U'*y`.

### Sensitivity of a problem
Computational problems map an input to an output.
- Example: solving a linear system maps the inputs $A, y$ to the output $x=A^{-1}y$.
- Example: training a neural network maps training data $x_i, y_i$ to output weights $w$.

The basic question of sensitivity is: how does the output of a problem change when we change its input? A good metaphor is a shower tap: if I turn the shower tap by 10 degrees, how much does the water temperature change? For instance, to compute $f(x)=x^2$, if we change $x$ to $\tilde{x}=x+\delta$, the change in input is $\vert{}\tilde{x}-x\vert{}=\vert{}\delta\vert{}$, and the change in output becomes $\vert{}\tilde{x}^2-x^2\vert{}=\vert{}2\delta x+\delta^2\vert{}$.

##### Definition: (absolute) condition number
The (absolute) condition number of a function $f:\mathbb{R}\rightarrow\mathbb{R}$ is the best bound $K$ of the form:
$$\vert{}f(x+\delta)-f(x)\vert{}\le K\vert{}\delta\vert{}+ o(\delta)$$
where $o(\delta)$ represents higher-order terms like $\delta^2, \delta^3$.

More formally, it is defined as a limit:
$$\kappa_{abs}(f,x)=\lim_{\delta\rightarrow0}\frac{\vert{}f(x+\delta)-f(x)\vert{}}{\vert{}\delta\vert{}}$$

For a scalar-valued function, this is essentially the norm of the derivative (when it exists): $\kappa_{abs}(f,x)=\left\vert{}\frac{df}{dx}\right\vert{}$.

We can generalize the definition to problems with multiple inputs. For computing $f(x,y)=x^2y$, if we change $x$ by $\delta$, the absolute condition number with respect to $x$ is $2\vert{}xy\vert{}$. Analogously, the condition number with respect to $y$ is $\frac{\partial f}{\partial y}(x,y)$.

##### Functions of vectors/matrices
For vector and matrix arguments, we make two changes to the definition:
1. We use **norms** rather than absolute values.
2. We take the largest change over all possible directions $d\in\mathbb{R}^n$, because functions of several variables can change faster in some directions than in others.

The formal definition becomes:
$$\kappa_{abs}(f,x)=\lim_{\delta\rightarrow0}\sup_{\vert{}\vert{}d\vert{}\vert{}\le\delta}\frac{\vert{}\vert{}f(x+d)-f(x)\vert{}\vert{}}{\vert{}\vert{}d\vert{}\vert{}}$$
- For differentiable real-valued functions, this is the norm of the gradient (at least for norm-2): $\vert{}\vert{}\nabla f_x\vert{}\vert{}$.
- For a general norm and $f:\mathbb{R}^m\rightarrow\mathbb{R}^n$, $\kappa_{abs}(f,x)$ is the norm of the Jacobian matrics

### Relative condition number

##### Why relative errors?
Absolute errors are useless without a reference point. For example, if a neural network computes an optimal price $\tilde{x}$ with an absolute error of $\vert{}\tilde{x}-x\vert{}=0.823\$$, we cannot evaluate its accuracy without context. If $x$ is the salary of an NBA player ($10^7\$$), it's a great estimate; but if $x$ is the optimal price of a nail ($0.001\$$), it's a terrible one.

Therefore, it is better to measure input/output changes as relative errors $\frac{\vert{}\tilde{x}-x\vert{}}{\vert{}x\vert{}}$:
- $\approx 1$: very bad accuracy; it's just a number with the same order of magnitude.
- $\approx 10^{-3}$: about 3 correct significant digits.
- $\approx 10^{-16}$: about 16 correct digits, which is typically the best we can do with double precision arithmetic.

**Important Rule:** I cannot stress it enough: use relative errors whenever you have to measure if something is small or large, including thresholds in algorithms, error measures, and stability checks.

##### Definition
Because $f(x)$ and $f(\tilde{x})$ should be compared relative to their own magnitudes, the relative condition number of a function $f$ is defined as:

$$\kappa_{rel}(f,x) = \lim_{\delta\rightarrow0}\sup_{\vert{}\vert{}d\vert{}\vert{}\le\delta} \frac{ \frac{\vert{}\vert{}f(x+d)-f(x)\vert{}\vert{}}{\vert{}\vert{}f(x)\vert{}\vert{}} }{ \frac{\vert{}\vert{}d\vert{}\vert{}}{\vert{}\vert{}x\vert{}\vert{}} } = \kappa_{abs}(f,x)\frac{\vert{}\vert{}x\vert{}\vert{}}{\vert{}\vert{}f(x)\vert{}\vert{}}$$

## Conditioning and Least Squares

### Condition Number of Solving Linear Equations
Let $A$ be a fixed square invertible matrix. We want to understand the variation in the output of the function:
$$f(A,y)= \text{(the solution of } Ax=y) = A^{-1}y$$
with respect to its input $y$. Consider two systems $Ax=y$ and $A\tilde{x}=\tilde{y}$ with $\tilde{y}\ne y$. Let $x$ and $\tilde{x}$ be their respective solutions. Then we can establish the absolute error bound:

$$\vert{}\vert{}\tilde{x}-x\vert{}\vert{}=\vert{}\vert{}A^{-1}\tilde{y}-A^{-1}y\vert{}\vert{}=\vert{}\vert{}A^{-1}(\tilde{y}-y)\vert{}\vert{}\le\vert{}\vert{}A^{-1}\vert{}\vert{}\vert{}\vert{}\tilde{y}-y\vert{}\vert{}$$

and from the original system, we have the property:

$$\vert{}\vert{}y\vert{}\vert{}=\vert{}\vert{}Ax\vert{}\vert{}\le\vert{}\vert{}A\vert{}\vert{}\vert{}\vert{}x\vert{}\vert{}$$

Combining these two inequalities, one gets the relative error bound formulation:

$$\frac{\vert{}\vert{}\tilde{x}-x\vert{}\vert{}}{\vert{}\vert{}x\vert{}\vert{}}\le\frac{\vert{}\vert{}A^{-1}\vert{}\vert{}\vert{}\vert{}\tilde{y}-y\vert{}\vert{}}{\vert{}\vert{}A\vert{}\vert{}}=\vert{}\vert{}A\vert{}\vert{}\vert{}\vert{}A\vert{}\vert{}^{-1}\frac{\vert{}\vert{}\tilde{y}-y\vert{}\vert{}}{\vert{}\vert{}y\vert{}\vert{}}$$

This bound holds for all $\tilde{y}$ hence also in the limit $\vert{}\vert{}\tilde{y}-y\vert{}\vert{}\rightarrow0$.

### Condition Number of a Matrix

**Theorem:** The relative condition number of solving linear equations (with $A$ fixed and $y$ as input) is defined as:
$$\kappa(A)=\vert{}\vert{}A\vert{}\vert{}\vert{}\vert{}A^{-1}\vert{}\vert{}$$
This quantity appears often; it is called the **'condition number of the matrix A'**. (Note: As the text explains, this is a slight abuse of terminology, since we should technically speak of the 'condition number of a problem', not 'of a matrix'.)
###### Examples
**Ill-conditioned matrices:** An ill-conditioned problem has a large condition number. What constitutes 'large' is subjective, but for instance, $\kappa(A)\approx10^{6}$ usually is considered large. This means that tiny variations in the input data (like a $10^{-6}$ perturbation) can cause massive percentage errors in the final solution.
    

### Condition Number with Respect to A

What if one changes $A$ and keeps $y$ fixed? The relative condition number of the problem $Ax=y$ with respect to its input $A$ is, again, $\kappa(A)=\vert{}\vert{}A\vert{}\vert{}\vert{}\vert{}A^{-1}\vert{}\vert{}$.

Using a slightly different notation, assume $A$ is perturbed to $A+\Delta A$, and $x$ is perturbed to $x+\Delta x$:

$$Ax=y, \quad (A+\Delta A)(x+\Delta x)=y$$

Expanding this expression, we can ignore the second-order term $\Delta A\Delta x$ since the perturbations are assumed to be infinitesimally small, getting:

$$y+\Delta Ax+A\Delta x+O(\vert{}\vert{}\Delta x\vert{}\vert{})=y$$

Rearranging the terms to isolate $\Delta x$ yields:

$$\Delta x=-A^{-1}\Delta Ax$$

Taking the norms of both sides gives us the relative variation limit:

$$\frac{\vert{}\vert{}\Delta x\vert{}\vert{}}{\vert{}\vert{}x\vert{}\vert{}}\le\vert{}\vert{}A^{-1}\vert{}\vert{}\vert{}\vert{}A\vert{}\vert{}\frac{\vert{}\vert{}\Delta A\vert{}\vert{}}{\vert{}\vert{}A\vert{}\vert{}}$$

### Condition Number and SVD
Recall that the 2-norm of a matrix is equal to its largest singular value: $\vert{}\vert{}A\vert{}\vert{}=\sigma_{1}$ (with norm-2). For a matrix $A\in\mathbb{R}^{n\times n}$, with singular values $\sigma_{1}\ge...\ge\sigma_{n}$, the condition number can be directly extracted using the SVD:
$$\kappa(A)=\frac{\sigma_{1}}{\sigma_{n}}$$
Indeed, using the decomposition $A=U\Sigma V^{T}$:
$$\vert{}\vert{}A\vert{}\vert{}=\vert{}\vert{}U\Sigma V^{T}\vert{}\vert{}=\vert{}\vert{}\Sigma\vert{}\vert{}=\sigma_{1}$$

Moreover, the inverse is given by $A^{-1}=V\Sigma^{-1}U^{T}$, and its norm is:
$$\vert{}\vert{}\Sigma^{-1}\vert{}\vert{}=\max_{i}\frac{1}{\sigma_{i}}=\frac{1}{\sigma_{n}}$$
Another property tells us that matrices with a high condition number are those that are almost singular.

### Condition Number and Distance to Singularity

The inverse of the condition number represents the "relative distance to singularity":

$$\frac{1}{\kappa(A)}=\min_{\tilde{A} \text{ singular}}\frac{\vert{}\vert{}A-\tilde{A}\vert{}\vert{}}{\vert{}\vert{}A\vert{}\vert{}}$$

Recall that the best rank-k approximation is given by the truncated SVD. The closest singular matrix (meaning the closest matrix with rank $n-1$) to $A=U\Sigma V^{T}$ is achieved by zeroing out the smallest singular value:
$$\overline{A}=U\begin{bmatrix} \sigma_1 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & \sigma_{n-1} \end{bmatrix}V^T$$

_(Note: Repairing the OCR formatting error from the original text, the norm of the difference between the original matrix and this closest singular matrix evaluates exactly to the smallest singular value)_:
$$\vert{}\vert{}A - \overline{A}\vert{}\vert{} = \sigma_n$$
### Conditioning of Least Squares Problems
Conditioning of linear least squares is a more complicated problem than the one for linear systems. We consider the linear least squares problem:
$$\min\vert{}\vert{}Ax-y\vert{}\vert{}$$
with $A\in\mathbb{R}^{m\times n}$ with full column rank.

> **Theorem (Trefethen, Bau, Theorem 18.1):** Its relative condition number with respect to the input $y$ is bounded by:
> $$\kappa_{rel,y\rightarrow x}\le\frac{\kappa(A)}{\cos \theta}$$
> and with respect to $A$ it is bounded by:
> $$\kappa_{rel,A\rightarrow x}\le\kappa(A)+\kappa(A)^{2}\tan \theta$$
> where $\theta$ is the angle such that $\cos \theta=\frac{\vert{}\vert{}Ax\vert{}\vert{}}{\vert{}\vert{}y\vert{}\vert{}}$.

### The Geometric Picture
In least squares, the vector $y$ is 'split' into two orthogonal components: the projection onto the image of $A$ (which is $Ax$) and the residual ($y-Ax$).
- **$\text{image}(A)$:** The space spanned by the columns of the matrix $A$.
- **$\theta$:** The angle between the vector $y$ and its projection $Ax$.

QR and SVD factorizations reveal their norms. If $A=QR$, where $Q=[Q_{0} \quad Q_{c}]$ or $A=U\Sigma V^{T}$ , where $U=[U_{0} \quad U_{c}]$ (as in their thin versions), then:
$$\vert{}\vert{}Ax\vert{}\vert{}=\vert{}\vert{}Q_{0}^{T}y\vert{}\vert{}=\vert{}\vert{}U_{0}^{T}y\vert{}\vert{}=\vert{}\vert{}y\vert{}\vert{}\cos \theta$$
$$\vert{}\vert{}y-Ax\vert{}\vert{}=\vert{}\vert{}Q_{c}^{T}y\vert{}\vert{}=\vert{}\vert{}U_{c}^{T}y\vert{}\vert{}=\vert{}\vert{}y\vert{}\vert{}\sin \theta$$

###### Some Intuition
- **$\theta\approx90^{\circ}$:** $y$ is almost orthogonal to $\text{Im } A$. A small (relative) change in $y$ causes a large (relative) change in the solution.
- **$\theta\approx0^{\circ}$:** This gives more well-behaved problems: the condition number is $\approx\kappa(A)$ instead of $\approx\kappa(A)^{2}$.

$\kappa(A)$ tells us how well we can extract $\text{Im } A$ from $A$. For instance, two matrices can have the exact same image space, but a small (relative) perturbation to one can alter its Image much more significantly. Actually, $\kappa_{2}(A)$ is the relative distance to the nearest matrix $\tilde{A}$ without full column rank, generalizing the square case.

# References
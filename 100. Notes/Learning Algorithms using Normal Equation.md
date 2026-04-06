---
Data: 2026-04-06T17:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Linear Models]]"
Area: "[[Master's degree]]"
---
# Learning Algorithms using Normal Equation

Differentiate $E(w)$ with respect to $w$. In the derivation we find that:
$$
\frac{\partial E(w)}{\partial w_j} = -2\sum_{p=1}^l (y_p - x_p^T w) x_{p,j}
$$
We can get the **normal equation** (point with gradient of $E$ w.r.t. $w=0$)
$$
(X^T X)w = X^Ty
$$
if $X^TX$ si not singular the unique solution is given by
$$
w = (X^TX)^{-1} X^T y = X^{+}y
$$
The '+' is [[Least Squares#Pseudoinverse|Moore-Penrose Pseudoinverse]]. Else the solution are infinite (satisfying the normal equation) we can choose the min norm (w) solution.

### Direct approach by SVD
The [[Singular Value Decomposition (SVD)]] can be used for computing the pseudoinverse of matrix  $X^+$
$$
X = U\Sigma V^T \Longrightarrow X^+ = V\Sigma U^T
$$
where $\Sigma$ is diagonal and $\Sigma^+$ by replacing every nonzero entry but its reciprocal.

Moreover we can apply directly SVD to compute $w = X^+y$ obtaining the minimal norm (on $w$) solution of least squares problem.

**Note**: THIS IS the learning alg. for the direct approach solution on w (or say in closed form). Many algorithms addressing the problems of efficiency and stability

### To find the normal equation
![[Pasted image 20260406180639.png]]

mposing this =0, we can easily obtain the normal equation (first by “sums”, then in matrix notations). And we also obtained the gradient of E rewritten as:

![[Pasted image 20260406180718.png]]


# References
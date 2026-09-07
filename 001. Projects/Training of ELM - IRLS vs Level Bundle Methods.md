---
Area: "[[Computational mathematics for learning and data analysis]]"
Github:
Other Link:
---
# Efficient Training of ETM - IRLS vs Level Bundle Methods

**(M)** is so-called **[[Extreme learning machine]]**, i.e., a neural network with one hidden layer, $y=w\sigma(W_1 x)$ , where the weight matrix for the hidden layer $W_1$ is a fixed random matrix, $\sigma(\cdot)$ is an element wise activation function of your choice, and the output weight vector is chosen by solving a least-squares problem with $L_1$ [[Tikhonov Regularization (Ridge Regression)|regularization]]
$$\arg\min_w f(w) \:\: \text{ with } \:\:f(w) = ||Xw - y||^2_2 + \lambda ||w||_1$$

**(A1)** is **iteratively reweighted [[Least Squares|least squares]]**: i.e., an iteration where you solve at each step the least-squares problem
$$
w_{k+1} = \arg \min_w f_k(w), \quad
f_k(w) = \frac12 \|X w - y \|_2^2 + \lambda \|W_k w\|_2^2
$$
where $W_k$ is the diagonal matrix with entries $(W_k)_{ii} = |(w_k)_i|^{-1/2}$. (You can check that in the limit when $w_k = w_{k+1} = w_*$, $f_k$ and $f$ have the same gradient). Use a threshold so that the values $W$ do not get too large if $(w_k)_i \approx 0$

**(A2)** an algorithm of the class of [level bundle methods](https://pages.di.unipi.it/frangio/abstracts.html#NDOB18)

### (M) [[Extreme Learning Machine]]
### (A1) [[Iteratively Reweighted Least Square]]
### (A2) [[Bundle Methods]]


# References
- [[Iterative_Reweighted_Least_Squares.pdf]]
- [Robust Regularized Extreme Learning Machine for Regression Using Iteratively Reweighted Least Squares](https://www.researchgate.net/publication/311625178_Robust_Regularized_Extreme_Learning_Machine_for_Regression_Using_Iteratively_Reweighted_Least_Squares)

- [[Extreme_Learning_Machine_Theory_Applications.pdf]]
- [[Standard_Bundle_Methods.pdf]]
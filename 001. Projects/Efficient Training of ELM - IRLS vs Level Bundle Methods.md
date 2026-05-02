---
Area: "[[Master's degree]]"
Github:
Other Link:
---
# Efficient Training of ETM - IRLS vs Level Bundle Methods

**(M)** is so-called **[[Extreme learning machine ELM]]**, i.e., a neural network with one hidden layer, $y=w\sigma(W_1 x)$ , where the weight matrix for the hidden layer $W_1$ is a fixed random matrix, $\sigma(\cdot)$ is an elementwise activation function of your choice, and the output weight vector is chosen by solving a least-squares problem with $L_1$ regularization
$$\arg\min_w f(w) \:\: \text{ with } \:\:f(w) = ||Xw - y||^2_2 + \lambda ||w||_1$$

**(A1)** is **iteratively reweighted least squares**: i.e., an iteration where you solve at each step the least-squares problem
$$
w_{k+1} = \arg \min_w f_k(w), \quad
f_k(w) = \frac12 \|X w - y \|_2^2 + \lambda \|W_k w\|_2^2
$$
where $W_k$ is the diagonal matrix with entries $(W_k)_{ii} = |(w_k)_i|^{-1/2}$. (You can check that that in the limit when $w_k = w_{k+1} = w_*$, $f_k$ and $f$ have the same gradient). Use a threshold so that the values $W$ do not get too large if $(w_k)_i \approx 0$

**(A2)** an algorithm of the class of [level bundle methods](https://pages.di.unipi.it/frangio/abstracts.html#NDOB18)


# References
- [[Extreme_learning_machine_and_its_applications.pdf]]
- [Iteratively Reweighted Least Squares for Basis Pursuit with Global Linear Convergence Rate](https://arxiv.org/pdf/2012.12250)
- [Iteratively Re-weighted Least Squares Minimization for Sparse Recovery, Ingrid Daubechies](https://sites.math.duke.edu/~ingrid/publications/DDFG.pdf)
- [Standard Bundle Methods](https://arpi.unipi.it/retrieve/e0d6c92e-be34-fcf8-e053-d805fe0aa794/StandardBundle.pdf)
---
Data: 2026-05-03T00:32:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
Area: "[[Master's degree]]"
---
# Regularisations

Regularisation is a common scalarization method used to solve the [[Least Squares]]. One form of the regularisation is to minimise the weighted sum of the objectives
$$
\arg\min ||Ax-y|| + \lambda||x||
$$
where $\lambda > 0$ is a problem parameter. Remember that $||.||$ is the [[Vector Norms|norm of a vector]]

Regularisation is used in several contexts:
- In an estimation setting, the extra term pealing large $||x||$ can be interpreted as out prior knowledge that $||x||$ is no too large 
- In a optimal design setting, the extra term adds the cost of using large value of the design variables to the cost of missing the target specifications
### [[Tikhonov Regularization (Ridge Regression)]]
### [[Smoothing Regularization]]
### [[L1-norm Regularization]] 
# References
- [Convex Optimization: Regularizations](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=320)
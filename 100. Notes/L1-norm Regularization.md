---
Data: 2026-05-03T21:57:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Linear Models]]"
Area: "[[Master's degree]]"
---
# L1-norm Regularization

Regularitation with a $L_1$-[[Vector Norms|norm]] can be used as a heuristic for finding a sparse solutions. For example consider the problem:
$$
\arg\min ||Ax - y|| + \lambda ||x||_1
$$
in which the residual is measured with the [[Orthogonality#Vector Norms|Euclidean norm]] and the regularization si done with $L_1-norm$. By varing the parameter $\lambda$ we can sweep out the optimal trad-off curve between $||Ax-y||$ and $||x||_1$ wich serves as an approximation of the optimal trafe-off curve between $||Ax-y||$ and the sparsity of cardinality of the vector x, ie, the number of non-zero elements
# References
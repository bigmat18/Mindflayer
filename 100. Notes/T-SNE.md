---
Data: 2026-07-19T14:54:00
Tags:
  - note
  - master
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# T-SNE

Most techniques for dimensionality reduction **are not able to retain both the local and the global structure of the data in a single map.** This is a usefull to see non linear structures ([[Representing real-world surfaces#Manifoldness|manifold]], cluster ...)  Simple tests on handwritten digits demonstrate this (Song et al. 2007).

Similarities between high- and low- dimensional data points is modeled with c**onditional probabilities**. 
- Conditional probability that the point $x_i$ would peak $x_j$ as its neighbor:

![[Pasted image 20260210231432.png | 450]]

- [[Gaussian Curvature|Gaussian]] centered in $x_i$
- $\sigma_i$ scarto per avere una certa "perplexity" per controllar i vicini effettivi

We are interested only in pairwise distance
$$
p_{i|i} = 0
$$
For the low-dimensional points an analogous conditional probability is used:

![[Pasted image 20260210231928.png]]

- x is the input dimension
- y is the foal dimension

###### Kullback-Leibler Divergence
**Coding theory:** expected number of extra bits required to code samples from the distribution P if the current code is optimize for the distribution Q.

**Bayesian view:** a measure of the information gained when one revises one's beliefs from the prior distribution Q to the posterior distribution P.

It is also called relative entropy.
- Definition for discrete distributions:
![[Pasted image 20260210232055.png | 450]]

- Definition for continuos distributions:
![[Pasted image 20260210232113.png | 450]]

In the SNE (Stochastic Neighbor Embedding) The goal is to minimizes the mismatch between $p_{j|i}$ (x) and $q_{j|i}$ (y). Using the Kullback-Leibler divergence this goal can be achieved by minimizing the function:

![[Pasted image 20260210232226.png | 450]]

Problem of SNE:
- The cost function is difficult to optimize.
- SNE suffers, as other dimensionality reduction techniques, of **the crowding problem**.

The solution in **T-SNE**. SNE is made symmetric: It employs a Student-t distribution instead of
a Gaussian distribution to evaluate the similarity between points in low dimension.

![[Pasted image 20260210232337.png]]

- The crowding problem is alleviated.
- Optimization is made simpler.

# References
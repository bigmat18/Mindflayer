---
Data: 2026-07-19T14:52:00
Tags:
  - note
  - master
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Locally Linear Embedding (LLE)

LLE attempts to discover nonlinear structure in high dimension by exploiting local linear approximation. We use this reduction when own data are near and manifold with low dimension

![[Pasted image 20260210225313.png]]

- INTUITION: assuming that there is sufficient data (well-sampled manifold) we expect **each data point** and its neighbors **can be approximated by a local linear patch**.
- The patch is represented by a **weighted sum of the local data points**.

To compute local patch we need:
1. Choose a set of data points close to a given one (ball-radius or K-nearest neighbours).
2. Solve $W_{ih}$

![[Pasted image 20260210225902.png | 500]]

We can do a LLE mapping where we define a new function with $Y$ and we need to find $\vec{Y}_i$ which minimizes the embedding cost function:
![[Pasted image 20260210230039.png| 500]]

The LLE algorithms is the following:
1. Compute the neighbors of each data point, $\vec{X}_i$
2. Compute the weights $\vec{W}_{ij}$ that best reconstruct $\vec{X}_i$
3. Compute the vectors $\vec{Y}_i$ that minimizes the cost function.

###### Example
![[Pasted image 20260210230220.png | 500]]

![[Pasted image 20260210230246.png | 350]]

# References
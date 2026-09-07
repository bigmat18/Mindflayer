---
Data: 2026-04-06T23:12:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Linear Models]]"
Area: "[[Master's degree]]"
---
# Linear basis expansion (LBE)

Note that in: 
$$
h_w(w) = w_1x + w_0 \text{ or } h_w= w^T \cdot x
$$
A statistical parametric models: **"linear**" does not refer to this (**red**) straight line, but rather to the way in which the regression coefficients w occur in the regression equation

![[Pasted image 20260406231619.png | 400]]

Hence, we can use also transformed inputs, such are $x, x^2, x^3, x^4,$ …. with non-linear relationship inputs and output, holding the learning machinery (Least Square solution) developed so far…
$$
h_w (x) = w_0 + w_1 x + w_2 x^2 + \dots + w_M x^M = \sum_{j=0}^M w_jx^j
$$
this is call **poynomial regression**

### Linear basis expansion (LBE)
We view call a poynomial regression as a basis trasformation: **linear basis expansion (LBE)**:
$$
h_w (x) = \sum_{k=0}^K w_k \phi_k (x)
$$
Augment the input vector with additional variables which are transformations of x according to a function phi ($\phi_k: \mathbb{R}^n \to \mathbb{R}$)

Some examples:
- Polynomail represetation of $\phi(x)=x_j^2$ or $\phi(x) = x_jx_i$ or
- Non-linear transformation of single inputs: $\phi(x) = \log(x_j)$, $\phi(x)=root(x_j)$
- Non-linear transformation of multiple input: $\phi(x) = ||x||$

Typically: Number parameters $K>n$ before it was $n$. The model is linear in the parameters (also in phi, not in x): we can use the **same learning alg. as before**.

**Note**: it can be applied for regression (here) or classification

### LBE Criticism
Which $\phi$ we should choose? Toward the so called **dictionary** approaches.
- **Pros**: Can model more complicated relationships (than linear) w.r.t. the inputs: it is more expressive.
- **Cons**: With large basis of functions, we easily risk overfitting, hence we require methods for controlling the complexity (Whereas **complexity** is not for computational cost but a measure of the flexibility of the model to fit the data) of the model.

### [[Tikhonov Regularization (Ridge Regression)]]

# References
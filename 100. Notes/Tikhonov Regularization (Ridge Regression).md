---
Data: 2026-04-06T23:33:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
Area: "[[Master's degree]]"
---
# Tikhonov Regularization (Ridge Regression)

There are many approach to **control model complexity** where complexity is not for the computational cost but a measure of the flexibility of the model to fit the data.

One of this is **Ridge regression: (Tikhonov regularization)**: smoothed model → possible to add constraints to the sum of value of $|w_j|$ penalizing models with high values of $|w|$ , i.e. favoring "sparse" models using less terms due to weights $w_j= 0$ (or close to 0) (it means a less complex model)

![[Pasted image 20260406233443.png]]

**Note** that for the **objective function** we use here the name Loss (used for the model training cost function) to distinguish from the Error E (useful to evaluate the model error and used for the data term inside this Loss)

### Solving it
$$
Loss(w) = \sum_{p=1}^l (y_p - x_p^Tw)^2 + \lambda||w||^2
$$
##### For the direct approach

![[Pasted image 20260406233615.png | 500]]

To understand how this decides which features are important and shrinks the others, we can imagine this formula as a division (or a "tug-of-war") between two opposing forces: the **Signal** ($X^T\mathbf{y}$) and the **Resistance** ($X^T X + \lambda I$).
###### Force 1: The Signal ($X^T\mathbf{y}$)
This part pulls the weights upward. It represents the raw correlation between your data and the target.
- **What it is:** This is the dot product (scalar product) between the columns of your features ($X$) and the actual target values ($\mathbf{y}$).
- **High Correlation (True Signal):** If a feature (like square meters) grows alongside the target price, their dot product will result in a **huge positive number**. This creates a massive upward "pull" for that specific weight.
- **Low Correlation (Noise):** If a feature is random (like pigeons on the roof), it will randomly pair with high and low prices. In the dot product, these random positive and negative multiplications cancel each other out, resulting in a **very small number**. The upward pull is weak.

_In short:_ This force tries to maximize all weights, but it naturally pulls much harder on weights that have a genuine positive correlation with $\mathbf{y}$.

###### Force 2: The Resistance & The Tax ($X^T X + \lambda I$)
This part acts as the denominator (the resistance) that the weights must overcome to grow.
- **If $\lambda = 0$ (No Regularization):** The resistance relies entirely on the data ($X^T X$). The model will eagerly use even the tiniest bit of correlation from the noise parameters to minimize the error, leading to overfitting.
- **If $\lambda \neq 0$ (With Regularization):** We are adding a fixed "tax" or resistance that every single weight must overcome.
- **The Result of the Tug-of-War:**
    - A strong signal (e.g., 1,000,000) easily overpowers the $\lambda$ tax. The math allows that weight to remain large.
    - A weak signal/noise (e.g., 50) is weaker than the $\lambda$ tax. Because it cannot overcome the resistance, the mathematical inversion brutally crushes that weight down toward zero.

###### The Mathematical Magic of $\lambda I$
Adding the identity matrix multiplied by $\lambda$ solves the mathematical and geometric flaws of the standard Least Squares approach.
1. **Solving the Singular Matrix Problem (Algebraic Fix):** In the standard formula $(X^T X)^{-1}$, if two columns (features) are highly correlated or very similar, the matrix becomes _singular_ (or nearly singular), meaning it is mathematically impossible to invert. The computer crashes. By adding $\lambda$ strictly to the diagonal of the matrix ($\lambda I$), we artificially alter the matrix, guaranteeing it is always invertible and numerically stable.
2. **Bending the Parallel Lines (Geometric Fix):** When features are highly correlated, the linear equations they form are nearly parallel. Parallel lines intersect extremely far away from the origin, causing the weights ($w_1, w_2$) to explode to massive numbers (e.g., $w_1 = 50000, w_2 = -49000$). By adding $\lambda$ to the main coefficients, we force the parallel lines to rotate and intersect much closer to the origin $(0,0)$. The equations are forced to balance out using smaller, safer numbers, thus preventing overfitting.

##### For the gradient approach
we still apply –gradient of the Loss. If we compute the gradient of the two terms (error and penalty terms) of the $Loss(w)$ w.r.t. weights $w_i$ separately, using eta only for term E. We obtaion:

![[Pasted image 20260406233803.png | 350]]

That is a **weight decay** technique (basically add $2\lambda w$ to the gradient). E.g. with 0 gradient, it decreases the value of each w with a fraction of the old w

### Trade of
Note the balancing (trade-off) between the two terms:
- **Small lambda ($\lambda$) value** → minimizing the loss the focus is on obtaining a small error data term (first term, minimize just the training error) with a too complex model (high norm of the weights), the risk is of **overfitting**,
- **High lambda ($\lambda$)** → minimizing the loss the focus is on the second term, hence the data error (first term) could grow too much, i.e. moving to **underfitting**

The trade-off is ruled by the value of **lambda** ($\lambda$). The main advantage is that we have a concrete realization of the control of model complexity, easy to be implemented and of general applicability.

### Tikonow [[Statistical Learning Theory (SLT)]]
The penalty term penalizes high value of the weights and tends to drive all the weights to smaller values. E.g. Some weights values can go even to zero

It implements a control of the model complexity. his leads to a model with **less (or proper)**
**VC-Dim**, with a trade-off obtained through **just a (1) parameter** that you can control: the $\lambda$

**0th Order Polynomial**
![[Screenshot 2025-11-18 at 11.50.53.png | 400]]
**Underfitting**: too simple model (red line) w.r.t. to the target function

**1st Order Polynomial**
![[Screenshot 2025-11-18 at 11.51.51.png | 400]]
Still poor solution (due to **underfitting**)

**3rd Order Polynomial**
![[Screenshot 2025-11-18 at 11.55.43.png|400]]
This case more **flexibility** is useful

**9th Order Polynomial**
![[Screenshot 2025-11-18 at 11.56.31.png|400]]

### Regularization $E_{RMS}$ vs $ln\lambda$

![[Pasted image 20260406234309.png | 550]]

![[Pasted image 20260406234326.png | 550]]

# References
[TIKHONOV REGULARIZATION AND TOTAL LEAST SQUARES](https://www.cs.umd.edu/users/oleary/reprints/j51.pdf)
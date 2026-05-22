---
Data: 2026-04-06T23:33:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
Area: "[[Master's degree]]"
---
# Tikhonov regularization

There are many approach to **control model complexity** where complexity is not for the computational cost but a measure of the flexibility of the model to fit the data.

One of this is **Ridge regression: (Tikhonov regularization)**: smoothed model → possible to add constraints to the sum of value of $|w_j|$ penalizing models with high values of $|w|$ , i.e. favoring "sparse" models using less terms due to weights $w_j= 0$ (or close to 0) (it means a less complex model)

![[Pasted image 20260406233443.png]]

**Note** that for the **objective function** we use here the name Loss (used for the model training cost function) to distinguish from the Error E (useful to evaluate the model error and used for the data term inside this Loss)

### Solving it
$$
Loss(w) = \sum_{p=1}^l (y_p - x_p^Tw)^2 + \lambda||w||^2
$$
**For the direct approach:**

![[Pasted image 20260406233615.png | 500]]

**For the gradient approach:**
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
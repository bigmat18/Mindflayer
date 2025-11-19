---
Data: 2025-11-14T15:34:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Introduction to Machine Learning]]"
Area: "[[Master's degree]]"
---
# Generalisation 
There are some ML issues:
- Inferring the general function from know data: an ill posed problem (for example in principle the solution is not unique)
	- with finite data we cannot expect to find the exact solution
- Work with a restricted hypothesis space
	- see also the inductive bias concept

**Learning phase**: to build the model (including training)
**Prediction phase**: evaluate the learned function over novel samples of data (generalisation capability)
**Inductive learning hypothesis**: any $h$ that approximates $f$ well on training examples will also approximate $f$ well on new (unseen) instances x

**Def Overfitting**: A learner overfits the data if it outputs a hypothesis $h() \in H$ having true/generalisation error (risk) R and empirical (training) error E, but there is another $h'() \in H$ having $E' > E$ and $R' < R$ (so that $h'()$ is the better one, despite a worst fitting)

Critical aspect: accuracy/perfomance estimation
- theoretical
- empirical (trining, test) and cross-validation techniques

###### Example
An example on a parametric model for regression:
- the set of functions is assumed as polynomials with degree M
- the **complexity** of the hypothesis increases with the degree M, $l =$ number of examples

Target = $\sin(2x\pi) +$ raandom noise (gaussian) 
![[Screenshot 2025-11-14 at 15.54.36.png | 400]]

Samples affected by noise (not always on the green “true” line)

The soluction is minimize the $E(w)$ (squared error) to find the best $w$ (fitting)
$$
E(w) = \sum_{p=1}^l (y_p - h_w(x_p))^2
$$
- $p$ is the example,
- $y_p$ the target for $p$ 
- $l$ the total number of examples
- $h_w(x_p)$ is the model output at the point $x_p$ ($x$ is the single variable, $n=1$)

![[Screenshot 2025-11-18 at 11.47.18.png | 400]]

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

In this case we have $E(w) = 0$ on training data but an high error on test set. Too complex model (in this case it fits even the noise). Poor representation on the (green) true function (due to **overfitting**)

###### Underfitting and Overfitting with the complexity (M)
![[Screenshot 2025-11-18 at 12.01.02.png | 400]]

Root-Mean-Square (RMS) Error:
$$
E_{RMS} = \sqrt{2E(w^*)/l}
$$
where $E(w^*)$ is the error for the trained model

![[Screenshot 2025-11-18 at 12.02.28.png | 450]]

But if in the previous example we maintain 9th order polynomial ad we change the size of the data set to $l=15$ (previous was 10)

![[Screenshot 2025-11-18 at 12.03.30.png|400]]

And we even more data $l=100$
![[Screenshot 2025-11-18 at 12.04.18.png|400]]
We can use higher M with a higher number of data

### [[Statistical Learning Theory (SLT)]]

# References
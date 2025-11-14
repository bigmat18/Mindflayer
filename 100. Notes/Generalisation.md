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
# References
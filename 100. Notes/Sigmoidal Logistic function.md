---
Data: 2026-04-18T23:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Neural Networks (NN)]]"
Area:
---
# Sigmoidal Logistic function

A non-linear **squashing** function like the sigmoidal logistic function: assumes a continuous range of values in the **bounded** interval `[0,1]`

![[Pasted image 20260419003938.png]]

The sigmoidal-logistic function has the property to be a **smoothed differentiable threshold function**. $a$ is the slope parameter of the sigmoid function.

![[Pasted image 20260419004033.png]]

These functions provide continues outputs but"
- For the Logistic function an output value
	- $\geq0.5$ (**threshold**) correspond to the **positive class**
	- <0.5 correspond to the **zero or negative class**
- t is possible to change this threshold (e.g. studying the effect on FP/FN etc, or by a ROC),
- and even to consider a rejection zone in an interval around the threshold value (to avoid fragile decisions)

For the TanH the **threshold is in 0** (with the analogues possibilities)
![[Pasted image 20260419004248.png]]

### Derivatives of Activation functions
- The derivative of the identity function is 1.
- The derivative of the **step** (threshold) **function** is not defined, which is exactly why it isn't used with LMS 
- **Sigmoids**: for asymmetric and symmetric case we have (a=1):

![[Pasted image 20260419011909.png]]

### LMS with $f_{\sigma}$
The sigmoidal-logistic function has the property to be a smoothed **differentiable** threshold function. Hence we can derive a [[Least Squares]] algorithm by computing the gradient of the meas quare loss function.

From $o(x) = x^T w$ to $o(x) = f_{\sigma}(x^Tw)$ where $f_{\sigma}$ is a logistic function. Find $w$ to minimize the residual sum of squares:
$$
E(w) = \sum_p (d_p - o(x_p))^2 = \sum_p (d_p - f_{\sigma}(x_p^Tw))^2
$$
###  [[Learning Algorithms using Gradient Descent|Gradiant]] with Sigmoidal
![[Pasted image 20260419122824.png]]

We can use this algorithm as the same for linear using the new delta rule (**batch/on-line**)

![[Pasted image 20260419122916.png | 500]]

Again, an **error correction rule**. Moreover:
- The parameters $a$ (slope of $f$) can affect the step of gradient descent
- Max of $f_{\sigma}$ for input (net) close to 0 (linear unit). Here high value of the $\sigma$ are possible.
- Minimum $f_{\sigma}$ are for **saturated cases** (where $f$ go to 0 or 1 asymptotically) -> better to avoid a premature saturation, small $\sigma$, and hence very slow changes of $w$, at the beginning (starting with small weights) or later

![[Pasted image 20260419123634.png | 350]]

**Insight**: we see also a bridge between the two alg.s: toward no corrections for correct outputs also for LMS alg. (as it was for the perceptron learn. alg.). I.e. Using $f_{\sigma}$ we are approximating not only the LTU but also the perceptron learn. alg.


# Reference 
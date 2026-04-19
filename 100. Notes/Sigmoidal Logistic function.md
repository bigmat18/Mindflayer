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


# References
---
Data: 2026-05-20T23:59:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Level Stabilization (PLBM)

The idea of level stabilitazion is in some sense opposite to that of the prevous approaches. In gereral $f_{\mathcal{B}}$  **is too optimisitc a model of** $f$, in that it **underestimates the true value of $f$** in a large part of the space.

So with the PLBM we do the opposite, it fix beforehadn **how much descent the model should attain**, so the algorithms will work in the **sublevel set** $lev(f_{\mathcal{B}}, l)$ for some given level parameter $l<f(\bar{x})$ 

We need to change the Mater Problem to select the right point:
$$
x^* = \arg\min \{ ||x-x^*_i|| : f_i(x) \leq l_i \}
$$

The **advantage** of PLBM approach in that the stabilization parameter $l$ has the scale of function valurs, which may make it easier to choose.

### Choose the $l_i$ value
##### $f_*$ is know
This is the simple case. The actual target $l_i$ must be between $[f(x^*_i), f_i]$ that means between the actual record ($f(x_i^*)$) and the global minimum. The simple strategy is to use a parameter $\lambda \in (0,1]$ 
$$
l_i = \lambda f(x_i^*) + (1 - \lambda) f_*
$$
##### $f_*$ is unknown
This is own case, the most common one. We can use the same formulaton above replacing $f_*$ with its lower bound $v_i$ that we could obtain computer the base master problem.

This means that we need to solve **two times the Master Problem at each iteration**. This could be a good strategy only if the oracle computation is an eavier computation compered to master problem.

##### $l_i$ arbitrarily
The alternative to the above problem is to choose $l_i$ arbitrarily. The possible troubling consegune is that **we will be too much optimistic and we choose $l_i$ impossible to achive**, this means $l_i < f_*$

In optimiziation the Master Problem may be empty but this is not an issues because it does not bring to a crash, it is a discovery, we now know that the $l_i$ value is a valid lower limit for $f_*$. So the algortihm can update $l_{i+i} > l_{i}$ and iterate


# References
- [[Standard_Bundle_Methods.pdf]]
- [Bundle methods for stochastic programs](https://svan2016.sciencesconf.org/conference/svan2016/BASLecture26.pdf)

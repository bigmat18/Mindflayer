---
Data: 2025-11-18T12:39:00
Tags:
  - note
  - youngling
Connection:
  - "[[Introduction to Machine Learning]]"
  - "[[Machine Learning]]"
  - "[[Introduction to Artificial Intelligence]]"
Area: "[[Master's degree]]"
---
# Statistical Learning Theory (SLT)

Putting all together:
- We want to investigate on a **generalisation** capability of a model (measured as a risk or test error)
	- with respect to the trining error
	- overfitting and underfitting zones
- The role of **model complexity**
- The role of the **number of data**

From all this things we can define the **Statistical Learning Theory (SLT)** that is a general theory relating such tops. Some formal settings:
- Approximation unknown $f(x)$ and $d$ is the target ($d=true f+noise$)
- Minimize [[Loss in ML|risk function]] (the true error over all the data domain)
$$
R = \int L(d, h(x))dP(x, d)
$$
Given:
- value from teacher ($d$) and the probability distribution $P(x,d)$
- a loss (or cost) function, for example $L(h(x), d) = (d-h(x))^2$

We need to search $h\in H$ that minimize the risk function $R$. But we have only the finite data set $TR=(x_p, d_p) p=1\dots l$ 

To search $h$: minimise empirical risk (**training error E**) finding the best values for the model free parameters
$$
R_{emp} = R_{emp} (h, TR) = \frac{1}{l}\sum_{p=1}^l (d_p - h(x_p))^2
$$
we call it **Empirical Risk Minimisation (ERM) inductive principle**. We can use $R_{emp}$ to approximate $R$

![[Screenshot 2025-11-18 at 12.19.42.png | 450]]
#### Vapnik-Chervonenkis-dim and SLT
Given the **VC-dim** (VC), a measure **complexity** of H (flexibility to fit data), for example the number of parameters for linear models/polynomial

**Definition:** VC-bounds in the form: it holds with probability $1-\delta$ that 
$$
R \leq R_{emp} + \epsilon (1/l, VC, 1/\delta)
$$
 - $\epsilon$ is a function that grows with VC (VC-dim) and that decreases with (higher) $l$ and $\delta$ (i.e. $l$ and $\delta$ are in the denominator)
 - We know that $R_{emp}$ decreases suing complex models (with high VC-dim) 
 - $\delta$ is the confidence, it rules the probability that the bound holds (for example low delta 0.01 $\to$ the bound holds with probability $0.99$)

Now we can see how it can "explain" the underfitting and overfitting and the aspect that control them.

Intuition:
- Higher $l$ (data) $\to$ lower VC-confidence and a bound close to R
- Too simple models (low VC-dim) can be not suff. due to high $R_{emp}$ (**underfitting**)
- Higher VC-dim (fix $l$) $\to$ lower $R_{emp}$ but VC-conf, and hence R, may increase (**overfitting**)

**Structural risk minimisation**: minimize the bound
![[Screenshot 2025-11-18 at 12.35.56.png | 550]]
Concept of control of the model complexity (flexibility): trade-off between model complexity (VC-dim) and TR accuracy (fitting)

###### Example
It is possible to derive an upper bound of the ideal error which is valid with probability (1-delta), delta being arbitrary small, of the form:
- **General**:
$$
R \leq R_{emp} + \epsilon(1/l, VC, 1/\delta)
$$
- **Example**:
$$
R \leq R_{emp} + \epsilon (VC/l, -ln(\delta / l))
$$
There are different bounds formulations according to different classes of $f$, tasks etc.
More in general, in other words (simplifying): we can make a good approximation of $f$ from examples, provided we have a good number of data, and the complexity of the model is suitable for the task at hand.
- Fit data as mush as possible to avoid underfitting (high $R_{emp}$), but not too mush so to avoid overfitting (due to increase of VC-confidence term)
- Duple descent/over-parametrization/besign overfit phenomena will be discussed later

#### Complexity control
- It allows formal framing of the problem of generalization and overfitting, providing analytic upper-bound to the risk R for the prediction over all the data, regardless to the type of learning algorithm or details of the model
- The ML is well founded: the Learning risk can be analytically limited and only few concepts are fundamentals
- It leads to new models (SVM) (and other methods that directly consider the control of the complexity in the construction of the model)
- It bases one of the inductive principles on the control of the complexity
- It explains the main difference with respect to supporting methods from CM (providing the techniques to perform fitting), apart from modelling aspects

# References
---
Data: 2025-11-14T14:41:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Introduction to Machine Learning]]"
Area: "[[Master's degree]]"
---
# Models in ML

Aim: to capture/describes the relationships among the data (on the basis of the task) by a “language” (numerical, symbolic, …). The “language” is related to the representation used to get knowledge. The model defines the class of functions that the learning machine can implement (hypotheses space)

**Example**: set of functions $h(x, w)$ where $w$ is the (abstract) parameter.

- **Training example** (superv.): An example of the form $(x, f(x)+noise)$, $x$ is usually an input vector of features, ($d$ or $t$ or) $y=f(x)+noise$ is called the target value
- **Target function**: the true function $f$
- **Hypothesis**: A proposed function $h$ believed to be similar to $f$. An expression in a given language that describes the relationships among data.
- **Hypothesis space** H: The space of all hypotheses (specific models) that can, in principle be output by the learning algorithm

###### Examples
- **Linear models**: representation of H defines a **continuously** parametrised space of potential hypothesis; each assignment of $w$ is a different hypothesis

![[Screenshot 2025-11-14 at 11.56.21.png]]
- **Symbolic Rules**: hypothesis is based on **discrete** representations; different rules are possible eg:
![[Screenshot 2025-11-14 at 11.57.23.png]]

- **Neural Nettworks**: as a computational model for the treatment of data, capable of approximation complex (non-linear) relationships between input and outputs

###### Paradigms and methods (languages of H)
- Symbolic and rule-based (or discrete H)
	- Conjuction of literals, [[Alberi Decisionali|decision tree]]
	- Inductive grammars, evolutionary algorithms
	- [[Logica di Primo Ordine (FOL)|inductive logic programming]]
- Sub-symbolic
	- Lineear discriminant analysis, Multiple linear regresion, LTU
	- Neural networks
	- Kernel method ([[Support Vector Machiens (SVM)|SVM]], gaussian kernels, spectral kernels, etc.)
- Probabilistic/Generative
	- Traditional parametric models
	- Graphics models
- Instance-based
	- Nearest neighbor

> **Theory (No Free Lunch Theorem)** : there is no universal “best” learning method (without any knowledge, for any problems,…): if an algorithm achieves superior results on some problems, it must pay with inferiority on other problems. 
> 
> In this sense there is no free lunch. E.g. Devroye (1982), Wolpert and Macready (1997), and others
# References
---
Data: 2026-04-19T13:23:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Neural Networks (NN)]]"
Area: "[[Master's degree]]"
---
# Why NN is a flexible model?

### Which tasks?
**Hypothesis space**: continuous space of all the functions that can be represented by assigning the weight values of the given architecture.

Note that, depending on the class of values produced by the network output units, discrete or continuous, the model can deal, respectively, with 
- **classification** tasks ([[Sigmoidal Logistic function]] output $f$) 
- **regression** tasks ([[Introduction Linear Model]] output $f$) tasks

![[Pasted image 20260419132505.png | 550]]

Also **multi-regression** or **multi-classes classifier** can be obtained by using multiple output units

![[Pasted image 20260419132537.png | 350]]
##### NN as a function
![[Pasted image 20260419135455.png | 400]]

- This is the function computed by a two-layer feedforward neural network
- Units and architecture just a graphical representation (of the data flow process)
- Each
![[Pasted image 20260419135544.png]]

can be seen as computed by an independent **processing element (unit, a hidden unit)**, or a **special kind of phi** ($\phi$) of LBE. Also, NN is a function non linear in the parameters $w$.

##### NN as a dictionary approach
- **[[Linear basis expansion (LBE)]]**: (Recall) **fixed** linear basis functions
![[Pasted image 20260419135726.png]]

- **Neural Network**: Adaptive (flexible) basis functions approach
![[Pasted image 20260419135845.png]]

the basis functions themselves are **adapted** to data (by fitting the w in $\phi$). **Note** that we fix the same type of basis functions for all the terms in the basis expansion (given by the activation function)

![[Pasted image 20260419140110.png | 350]]

$h(x)$ as (nonlinear function of weighted) sums of nonlinearly transformed linear models + the important enhancement of adaptivity

### Hidden Layer Relevance and Interpretation
Each basis function (**hidden unit**) computes a **new** nonlinear derived features, **adaptively** (by learning, according to the training data)

I.e. the parameters of the basis function w are learned from data by learning.
![[Pasted image 20260419140422.png | 400]]

In other words:
- The representational capacity of the model is related to the presence of a hidden layer of units, with the use of non-linear activation function, that transforms the input pattern into the **internal representation** of the network.
- The learning process can define a **suitable internal representation**, also visible as new hidden features of data, allowing the model to extract from data the higher-order statistic that are relevant to approximate the target function.

##### Note: Non-Linear Hidden Layer is need
Non-linear units are essential: MLP with linear units = 1 layer NN!

![[Pasted image 20260419140635.png | 350]]

Anyway for the learning algorithm this introduces an issue: The model is **non-linear in the parameters**, i.e. w.r.t. to $w$ → we have a non linear optimization problem


# References
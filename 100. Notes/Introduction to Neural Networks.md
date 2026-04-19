---
Data: 2026-04-18T22:09:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Neural Networks (NN)]]"
Area: "[[Master's degree]]"
---
# Introduction to Neural Networks

An **artificial Neural Nerworks** is based on the concept of nervous system and neurons. In a NN there are artificial neuron, that are the processing unit.
- Node, neuron or unit
- Input: from extern source or other units (R)
- Input Connections: weights w: free parameters: these can be modified by learning (synaptic strength)

![[Pasted image 20260418221221.png]]

The Weighted sum $net_i$ is called the **net input** to unit i. Note that $w_{ij}$ refers to the weight of the unit $i$, i.e.from unit/input $j$ to unit $i$. The function $f$ is the unit's **activation function** (e.g linear, LTU, …).

### Notation for $w_{ij}$
Some media e.g. Wikpedia (backprogation section) and some NN simulators/libraries use a different notation, whereas, for example $w_{2j}$ (in the blue box) is the weight from input 2 to unit $j$, instead of using as us the traditional $w_{j2}$ (because $w_j$ belong to unit $j$)

![[Pasted image 20260418221516.png]]

### Activation functions

![[Pasted image 20260418221604.png]]

#### [[Introduction Linear Model|Linear Activation Function]]
#### [[Perceptron]]
#### [[Sigmoidal Logistic function]]

#### Radial Basis Functions (RBF Networs)
![[Pasted image 20260419005839.png| 550]]
Stochastic neurons: output is +1 with probability $P(net)$ or –1 with $l-P(net)$→ Boltzmann machines and other models rooted in statistical mechanics
#### Tahn-Like 
Piecewise linear approximation for efficeint computation
![[Pasted image 20260419010129.png]]

#### ReLU (Rectified Linear Unit)
It has become a default choice for Deep models, so it (and its variants) deserves more attention discussing Deep Learning

![[Pasted image 20260419010004.png | 500]]

### Derivatives of Activation functions
- The derivative of the identity function is 1.
- The derivative of the **step** (threshold) **function** is not defined, which is exactly why it isn't used with LMS 
- **Sigmoids**: for asymmetric and symmetric case we have (a=1):

![[Pasted image 20260419011909.png]]


# References
---
Data: 2026-04-19T14:08:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Neural Networks (NN)]]"
Area:
---
# Is the flexibility theoretically grounded?

The Universal approzimation is very important. Many early results (Cybenko 1989, Hornik et al. 1993, etc). Shortly:
- A single hidden-layer network (with logistic activation functions) can approximate (arbitrarily well) every continuous (on hyper cubes) function (provided enough units in the hidden layer
- A MLP network can approximate (arbitrarily well) every input-output mapping (provided enough units in the hidden layers)

![[Pasted image 20260419142036.png]]


After this fundamental result (MLP is able to represent any function), **two issues will deserve our attention:**
1. How to learn by NN
2. How to decide a NN architecture

### NN Expressive power
The **expressive power** of NN is strongly influenced by two aspects: the number of units and their configuration (architecture)

The number of units can be related to the discussion of the **VC-dimension** of the model. Specifically, the network capabilities are influenced by the **number of parameters $w$**, that is proportional to the **number of units**
$$
Number\_parameters = \#input\_units \times \#hidden\_units \times \# output\_units
$$
and further studies report also the dependencies on their value sizes, example:
- Weights = 0 → minimal VC-dim
- Small weights → linear part of the activation function (small VC-dim)
- Higher weights values → more complex mode

### How many layers?
A look ahead (toward deep learning):
- The univ. approx. theorem is a fundamental contribution
- It show that 1 hidden layer is sufficient in general, but it does not assure that a “small number” of units could be sufficient (it does not provide a limit on such number)
- It is possible to find boundaries on such number (for many f. families)
- But also to find “no flattening” results (on efficiency, not expressivity)
	- cases for which the implementation by a single hidden layer would require an exponential number of units (w.r.t input dim.), or non-zero weights,
	- while more layers can help (it can help for the number of units/weights and/or for learning such approximation)
	- But is it easy to optimize (training) a MLP with many layers?

### On the inductive bias of N
NN with backprograpation learning algorithm: generally related to the **smoothness** properties of functions:
- Small input variations -> small output variations
- E.g. a locally limited value of the first derivative
- A very common assumption in ML

Why make sense? A target function that is **Non-smooth**: random number generator -> generalization cannot be achived


# References
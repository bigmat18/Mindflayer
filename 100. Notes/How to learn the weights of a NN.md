---
Data: 2026-04-19T19:45:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Neural Networks (NN)]]"
Area: "[[Master's degree]]"
---
# How to learn the weights of a NN?

The [[Learning Algorithms]] allows to adapting the free-parameters $w$ of the model, ie the values of the connection-weights, in order to obtain the best approximation fo the target function.

As usual, we will realize this in terms of minimisation of an error (or loss) function on the training data set. If wee see the **[[Perceptron]]** 
- We have [[Perceptron#Perceptron Learning Algoritm|Perceptron learning algorithm]]
- Cannot represent all the boolean functions

A network of Perceptrons can represent every Boolean function, but there is a problem, define a **learning algorithm for the network (MLP)**

What is different? Credit assigment problem:
- Which credit to the **hidden units**
- not easy (as for single Percetron) when errors signal is not directly measurable: we don’t know the error(**delta**)/desired response for the **hidden units** (useful to change their weights).

Non-linear wrt to w → non linear optimization problem. Supposed too difficult by Minsky-Papert (1969), while faced by many researchers (see historical notes) with the **backpropagation algorithm** popularized by Rumelhart, Hinton, Williams in the PDP book (1986) → renaissance of NN

### The Loading Problem
Note that in general, it is indeed a difficult problem: **Loading problem** (loading a given set ot TR data into the free par. of the NN)
- given a network and a set of examples
- answer yes/no: is there a set of weights so that the network will be consistent with the examples?

The loading problem is NP-complete (Judd, 1990), it is not known a polynomial alg. to solve it. 

In practice networks can be trained in a reasonable amount of time (see the back prop alg.) although optimal solution is not guaranteed.

### How to solve?
The key steps are:
- Credit assignment problem: how to change the hidden layer weights?
- The [[Learning Algorithms using Gradient Descent|gradient descend approach]], minimizing a loss function, can be extended to MLP, provided that the loss function andthe activation functions are differentiable functions

So, to find the **delta** for every units in the network
![[Pasted image 20260419200209.png]]


# References
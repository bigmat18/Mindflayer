---
Data: 2026-04-19T13:07:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Neural Networks (NN)]]"
Area: "[[Master's degree]]"
---
# Standard Feedforward NN

There are two view:
- A **network** of interconnected units
![[Pasted image 20260419130827.png]]

- A flexible **function** $h(x)$ (as nested non-linear functions)
![[Pasted image 20260419131057.png]]

In a **MLP (Multi Layer Perceptron)** architecture:
- the units are connected by **weighted links** and they are organized in the form of layers
- the **input layer** is simply the source of the input x that projects onto the hidden layer of units: it loads (copy) the (extern) input patterns x (the input units i do not compute the net and f)
- the **hidden layer** projects onto the **output layer** (feedforward computation of this two-layers network in the fig.) or to another hidden layer

The NN model traditionally presented by the type of:
- **Unit**: net, activation functions
- **Architecture**: number of units, topology (e.g. also number of layers)
- **[[Learning Algorithms]]**

### Units

![[Pasted image 20260419131311.png]]

Importants notations:
- The index $t$ denotes a generic unit, it can be either $j$ or $k$
- The index $u$ denotes a generic input component, it can be either $i$ or $j$
- In the unit, $x$ is a generic input from an external source (input vector) or from other units according to the position of the unit in the network.
- If we **load the pattern $x$ in the input layer**, we can use the notation with $o$ for both the inputs and the hidden units outputs. Hence, inside the network, the input to each unit $t$ from any source $u$ (through the connection $w_{tu}$) is **typically denoted as $o_u$** (we will use this style in the back-prop derivation).

### MLP Architecture
The architecture of a NN defines the topology of the connections among the units. The **two-layer feedforward neural network** described in Equation $h(x)$ corresponds to the well-know **MLP (Multi Layer Perceptron) architecture**

![[Pasted image 20260419131803.png]]
##### Feedforward processing
The processing of a pattern for feedforward NN precedes as in the following (from the input layer to the output layer). For each input pattern $x$
1. the input pattern is load in the input layer
2. we compute the output of all the units of the 1st hidden layer
3. we compute the output of all the units of the 2st hidden layer and so on for all the hidden layers
4. we compute the output of all the units of the output layer (NN output $h(x)$)
5. we can now compute the error (delta) at the output level

##### Feedforward versus Recurrent
In the **feedforward** we have a function $direction: in \to out$

Instead in the **recurrent** neural networks: A different category of architecture, based on the addition of feedback loops connections in the network topology,
- The presence of **self-loop** connections provides the network with dynamical properties, letting a memory of the past computations in the model.
- This allows us to extend the representation capability of the model to the processing of sequences (and structured data).

### [[Why NN is a flexible model?]]
### [[Is the flexibility theoretically grounded?]]
### [[How to learn the weights of a NN?]]

# References
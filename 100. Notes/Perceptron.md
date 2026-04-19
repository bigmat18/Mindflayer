---
Data: 2026-04-18T22:29:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Neural Networks (NN)]]"
Area: "[[Master's degree]]"
---
# Perceptron
This concept was studied by Frank Rosenblatt (1957-1958,1960, …)

![[Pasted image 20260418222327.png]]

The idea of perceptron bring with it many concepts:
- Nice: single neuron is a very simple computational unit
- Minsky: “for the kind of recognition it can do, it is such a simple machine that it would astonishing if nature did not make use of it somewhere”
- A biologically inspired model with a simple computational unit (and its [[Learning Algorithms]])
- that is a model of historical importance for the ML, with different approaches since the early 60s

### Models with Perceptron
Perceptrons can be composed and connected to build a networks: from LTU (Linear translation unit) to NN. (MLP NN = Multi Layer Perceptron).
- Paradigmatic for the transition from linear to non-linear models towards the flexible models at the state-of-the-art in ML

**Note**: actually NN are realized by software (simulators) or hardware on chips

#### McCulloch & Pitts Networks
>  Logical calculus of the ideas immanent in nervous activity. 
- What does a given net compute?
- Can a given net compute a given logical sentence?

Neurons are in two possible states: **firing** (1) and **not firing** (0). All synapses (connections) are equivalent and characterized by a real number (their strength $w$), which is:
- **positive** for excitatory connections
- **negative** for inhibitory connections

A neuron i becomes **active** when the sum of those connections $w_{ij}$ coming from neurons $j$ connected to it which are active, plus a **bias**, is larger than zero.
- Binary inputs
- Again binary output (aka binary classification task)

###### AND, OR Boolean functions

![[Pasted image 20260418224450.png]]

###### Exclusive OR (XOR)
![[Pasted image 20260418224600.png]]

###### XOR by a Two Layers Network

![[Pasted image 20260418224707.png]]

We use the **DE MORGAN rule**: not(a and b) = not (a) or not (b)

### Hidden Layer and Representation
Useful concept of internal **(re)representation** of input variables via internal (hidden) units. Developing high level (hidden) features is a key factor in NN. The representation in the hidden layer makes easier the task to the output layer (last unit)
- E.g. See the figure above on the right: point (1,0) (in the sud-est corner) becames h1=0 h2=1 (north-west corner) → now it is a linear separable problem

A look ahead:
- Such **composition** of sub-/intermediate operations to perform a complex task can be extended through many layers of abstraction
- In NN such internal representation/intermediate features can be **learned**
- Learning internal distributed representation → representation learning and deep learning concepts


### Perceptron Learning Algoritm
Two kinds of method for learning (also historical view):
1. **Adaline** = Adaptive Linear Neuron (Widrow, Hoff): inear unit during training: LMS direct solution and gradient descent solution
	- Regression tasks: See the LMS algorithm
	- For classification: See the LTU and LMS algorithm of a previous lectures (on linear model)
	- An approach that we will generalize to MLP
2. **Perceptron** (Rosenblatt): non-linear unit during training: with hard limiter or Threshold activation function
	- Only classification: Capabilities studies, convergence theorem

The idea behind the perceptron learning algorithm is to minimize the number of misclassified patterns:
- find $w$ s.t. $sign (w^T x) = d$
- On-line algorithm: a step can be made for each input pattern
- Note that to build linear classifier we will have from now 3 learning alg:
	1. Adaline LMS direct solution
	2. Adaline LMS gradient based alg
	3. **Perceptron learning algorithm**

The step of the algorithm are:
1. Initialize the weights (either to zero or to a small random value)
2. pick a learning rate $\eta$ (this is a number between 0 and 1)
3. until stopping condition is satisfied (e.g. weights don't change):
	- For each training pattern $(x, d)$ $(d = +1, -1)$ let $out$ be the output
	- Compute output activation $out= sign(w^T x)$
	- If $out = d$ dont change weights (i.e. minimize only misclassifications)
	- If $out \neq d$ update the wights:
	![[Pasted image 20260418231830.png]]

Or (in a different form):

![[Pasted image 20260418231856.png]]

### Geometrica View
**An example:** Before updating the weight $w$ (pointing the positive region), we note that both $p_1$ and $p_2$ are incorrectly classified (the red dashed line is decision boundary). 

Suppose we choose $p_1$ to update the weights as in picture below. $p_1$ has target value $d=1$, so that $w$ is moved a small amount in the direction of $p_1$. The new boundary (blue dashed line) is better than before (and $w_new$ closer to $p_1$). 

![[Pasted image 20260418233057.png]]

### Delta Rule
- The form $w_{new} = w + \eta dx$ is seen as **Hebbian Learning**
- The other form $w_{new} = w + \eta (d-out) x \Leftarrow x = w + \eta \delta x$ 
	- is in the form of **error-cerrection learning**

Recall from [[Least Squares|LMS]]: this is an error corrections rule (**delta**/ Windrow-Hoff/Adaline/LMS rule) that changes the $w$ proportionally to the error (target d-output)
- E.g. (target d-output) = err = 0 -> no correction
- (input > 0) if err + (output is too low), increase $w$ -> increse $w^T x$ -> reduce erro
- ...

In terms of “neurons”: the adjustment made to a synaptic weight is proportional to the product of error signal and the input signal that excite the synapse. **Easy to compute** when **errors signal $\delta$** is directly measurable (we know the desired response for each unit)

### Perceptron Covergence Theorem
The perceptron is guaranteed to converge (classifying correctly all the input patterns) in a finite number of steps if the problem is linearly separable.
- (independently of the starting point, although the final solution is not unique and it depends on the starting point)
- May be “unstable” if the problem is not separable
	- In particular it develop cycles (with a set of weights that are not necessarily optimal)

**Note**: I’ll simply the notations in the proof (e.g. assume the not reported scalar product with T as usual among vectors)

We can focus on **all positive patterns** task (as shown in the following:
- Assume $(x_i, d_i)$ in the TR set, with $d_i = +1$ or $-1$ and $i=1\dots l$
- Linearly separable -> $\exists w^*$ solution s.t.
$$
d_i (w^*x_i) \geq \alpha_i \text{ with } \alpha = \min_i d_i (w^* x_i) > 0
$$
- Hence $w^*(d_ix_i) \geq \alpha$ 
- Defining $x_i' = (d_ix_i)$ then $w^*$ is a solution $\Leftrightarrow$ $w^*$ is a solutions of $(x_i', +1)$
- Assuming $w(0)=0$
	- $\eta =1$
	- $\beta = \max_i ||x_i||^2$ ($i=1, \dots, l$ and $||\dots||$  denotes the [[Orthogonality#Vector Norms and Distance|Euclidean norm]])

After $q$ errors (all false negative)
$$
w(q) = \sum_{j=1 \to q}x_{(i_j)}
$$Because $w(j) = w(j-1) + w_{ij}$

where $i$ is used to denote the patters belonging to the subset of misclassified pattres, e.g. the indices 2, 8, 9, 14 in the TR set

**Rule**: $w_{new} = w + \eta dx$ (but all $d$ are +1 here, only positive patters, and $n=1$)

> **Theorem**: For linearly separable tasks, the perceptron algorithm converges after a finite number of steps

The **basic idea** is that we can find lower and upper bound to $||w(q)||^2$ as a function of $q^2$ steps (lower bound) a $q$ steps (upper bound) -> we can find $q$ (numer of steps) s.t. the algorithm converges:

![[Pasted image 20260419000444.png]]

#### Proof
##### Lower bound on $||w(q)||^2$
![[Pasted image 20260419000643.png]]

![[Pasted image 20260419000701.png]]
##### Upper bound on $||w(q)||^2$
![[Pasted image 20260419000742.png]]

##### Bring them together
![[Pasted image 20260419000855.png | 500]]

### Perceptron Learning Alg vs LMS Alg
Apparently similar but note that:
- LMS rule was derived (by the gradient) without threshold activation functions: minimization of the error of the linear unit (using directly $w^Tx$)
- The perceptron use $out = sign(w^Tx)$

Hence, for training:
- $\delta = (d -w^Tx)$ for the LMS approach
- $\delta = (d-sign (w^T x))$ for the perceptron learning algorithm

Of course the model trained with LMS can still be used for classification
applying the threshold function 

LMS not necessarily minimize the number of TR examples misclassified by the LTU

![[Pasted image 20260419001435.png]]



# References
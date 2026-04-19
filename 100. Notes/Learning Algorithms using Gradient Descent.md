---
Data: 2026-04-06T17:59:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Linear Models]]"
Area: "[[Master's degree]]"
---
# Learning Algorithms using Gradient Descent

The first importat stuff is learning what is the [[Gradient]]. In a nutshell the The gradient at a point is a vector pointing in the direction of the steepest slope at that point.

The steepness of the slope at that point is given by the magnitude of the gradient vector.
- The **gradient** shows the direction where the function grows
- The **negative of the gradient** shows the direction where the function decreases

![[Pasted image 20260406180937.png | 400]]

Previous derivation suggest the line to construct an iterative algorithm based on:

![[Pasted image 20260406181037.png]]

**Gradient** = **ascent direction**: we can move toward the minimum with a gradient **descent**. This means changin $w$ with $\Delta w =-$ gradient of $E(w)$

Local search: it begins with an initial weight vector. We modify it iteratively to decrease up to minimize the error function (steepest descent).

![[Pasted image 20260406181429.png | 500]]

#### Error surface for linear model with 2 weights (w)
![[Pasted image 20260406181526.png | 500]]

#### The gradient vector
We can work in a multi-dim space without the need to visualize it.
![[Pasted image 20260406181604.png| 500]]

#### Using the Delta Rule
Hence, as iterative approach we will move using a learning rule based on a «delta» (changing) of $w$ proportional to the (opposite) of the local gradient.

The movements will be made iteratively according to:
$$
w_{new} = w + \eta \cdot \Delta w \:\:\:\: \text{or component-wise, ie, for each w}
$$
that is the **learing rule**. And $\eta$ is the step size (learning rate) parameter (ruling the speed of out gradient descending)

### [[Gradient Method|Gradiant descent algorithm]]
A simple view of gradient descent is the following:
1. Start with weight vector $w_{initial}$ (small), fix $\eta$ ($0 < \eta < 1$)
2. Compute $\Delta w =$ - "Gradient of $E(w)$" = $-\frac{\partial E(w)}{\partial w}$ (of for each $w_j$)
3. Compute $w_{new} = w + \eta \cdot \Delta w$ (of for each $w_j$)
4. Repeat (w) until converge of $E(w)$ si sufficiently small

$\eta$ is the step size or **learning rate**: that is the speed/stability trade-off: can be (gradually) decreased to zero (guarantee convergence, avoiding oscillation around the min).

##### Batch Version
For **bach version** the gradient is the sum over all the $l$ patters:
$$
\frac{\partial E(w)}{\partial w_j}= -2 \sum_{p=1}^l (y_p - x_p^T w)x_{p,j}
$$
provide a more precision evalutation of the gradient over a set of $l$ data. And we upgrade the weight after this sun.
##### On-line Version
For the **on-line/stochastic version** we upgrade the weights with error that is computed for each pattern.
- hence, the second pattern output is based on weights already updated from first, and so ahead.
- It makes progress with each examples it looks at: ita can be the faster, but need smaller $\eta$
$$
\frac{\partial E_p(w)}{\partial w_j}= -2 (y_p - x_p^T w)x_{p,j} = -\Delta_p w_j
$$
###### Example
- We update w after (repeating) an “epoch” of l training data (blue)
- On-line algorithm (stochastic gradient descent - SGD) We update w after each pattern p ($\Delta_p$ for each pattern→ (purple and green)

![[Pasted image 20260406183735.png | 300]]

###### Example 
A learniong curve examples. These are **learning curves**: They show how the error decreases through gradient descent iterations.

![[Pasted image 20260406183931.png | 550]]

### Gradient descent as Error correction delta rule
![[Pasted image 20260406184108.png | 400]]

Where $x_{p,j}$ is the component $j$ of the input pattern $p, y_p$ the output for $p, w$ free par, $l$ num of examples (The constant 2 can be omitted)

This is an **error correction** rule (**Widrow-Hoff** or delta rule) that change each $w_j$ proportionally to the error (**target y - output**):
- E.g. (target y-output) = $err=0$ -> no correction
- ($input_j$>0) if err + (output is too low), positive delta → increase $w_j$ → increment the output → less err
- ($input_j$>0) if err - (output is too high), negative delta → decrease $w_j$ → reduce output → less err

We improve by learning from previous errors “seeking and blundering we learn (Goethe)

### Delta-W as Error Correction Learning rule
![[Pasted image 20260406184526.png]]


# References
---
Data: 2026-04-06T10:31:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Linear Models]]"
Area: "[[Master's degree]]"
---
# Regression Models

Process of estimating of a real-value function on the basis of finite set of noisy samples (supervised task) known pairs ($x$, $f(x) + \text{ random noise}$) the task is find $f$ for the data.

![[Pasted image 20251111131942.png | 600]]

### Univariante Linear Regression
Univariante case is a simple linear regression: we start with 1 input variable $x$ =, 1 output varialbe $y$. We assume a model $h_w(x)$ expressed as $out = w_1 x + w$, where $w$ are real-valued coefficients/free parameters (weights)

Infinite **hp space** (continuous w values) but we have nice solution from classical math
- Surprisingly we can “learn” by this basic tool
- Although simple it includes many relevant concept of modern ML and it is a basis of evolved methods in the field

### Learning via LMS
- **Training**: find w such that minimize error/empirical loss (best data fitting on the training set with l examples): i.e. we are now focusing on the $R_{emp}$
	- **Given**: a set $l$ training examples $(x_p, y_p), p=1\dots, l$
	- **Find**: $h_w (x)$ in the form $w_1 x + w_0$ (hence the values of $w$) that minimizes the expected loss on the training data.

For the loss we use the square fo errors: [[Least Squares]] that means find $w$ to minimize the residual sum of squares
$$
Loss(h_w) = E(w) = \sum_{p=1}^l (y_p - h_w(x_p))^2 = \sum_{p=1}^l (y_p -(w_1 x_p + w_0))^2 
$$
where $x_p$ is p-the input/pattern/example $y_p$ the output for $p$, $w$ free par, $l$ num of examples

**Note**: to have the mean divide by $l$. Indeed for the univariante case, with variable $x_p = x_{p,l} = (x_p)_1$
##### Why LMS to fix the data with $h$?
[[Least Squares]]: Find $w$ to minimize the residual sum of squares 

![[Pasted image 20260406111307.png | 350]]

$$
y = w_1x + w_0 + noise \:\:\:\text{ with } \:\:\: h_w(x) = w_1 x + w_0
$$
Different blue lines will have different green bars. Minimizing the green bars (residuals /errors) is a way to find the best approximation/fitting of the data. i.e. our $h_w(x)$ or blue line).
The squares of errors $E(w)$ quantify such green bars:
$$
E(w) = \sum_{p=1}^l (y_p - h_w(x_p))^2
$$
$E(w)$ = green, $y$ = red, $h_w$ = blue

The method of **least squares** is a standard approach to the approximate solution of over-determined systems, i.e., sets of equations in which there are more equations than unknowns.

##### How to Solve?
Remember: local minimum as stationary point: the gradient is zero.
$$
\frac{\partial E(w)}{\partial w_1} \:\:\:\: i=1, \dots, \text{dim\_input} + 1 = 1, \dots, n+1
$$

![[Pasted image 20260406114853.png | 350]]

For the simple Linear Regression (2 free parameter)
$$
\frac{\partial E(w)}{\partial w_0} = 0 \:\:\:\: \frac{\partial E(w)}{\partial w_1} = 0
$$
Convex loss function -> we have the following solution (no local minima)

![[Pasted image 20260406112658.png]]


# References
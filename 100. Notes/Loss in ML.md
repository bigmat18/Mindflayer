---
Data: 2025-11-14T15:02:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Introduction to Machine Learning]]"
Area: "[[Master's degree]]"
---
# Loss in ML
We speak about a "good" approximation to a function $f$ from examples. But how we can measure the quality of the approximation?
- Recall that we produce the "distance" between $h(x)$ and $d$
- We want to measure the "distance" between $h(x)$ and $d$ (objective function for minimisation of errors)

We use a ("inner") loss function/measure: $L(h_w(x), d)$ (for a pattern $x$). 
E.g. hight value $\to$ poor approximation

The error (or Risk of Loss) is an expected value of this $L$, for example a "sum" or mean of the inner loss L over the set of samples
$$
Loss(h_w) = E(w) = \frac{1}{l}\sum_{p=1}^{l} L(h_w(x_p), d_p)
$$
We will change L for different tasks. **Note**: index p is used for the samples $p=1\dots l$ 

### Regression Loss
The task is predicting a numerical value
- **Output**: $d_p = f(x_p) + e$ (real value function + random error)
- **H**: a set of real-valued functions
- **Loss function** $L$: measures the approximation accuracy/error. A common loss function for regression is the squared error:
$$
L(h_w(x_p), d_p) = (d_p - h_w(x_p))^2
$$
The mean over the data set proved the **Mean Square Error (MSE)**
###### Example
In the example we have $h(x) = w_1x + w_0$ as the blue line and in green the errors at the **data points $(x_i, y_i)$** in red, where the target $d_i$ for $x_i$ is denoted $y_i$ in the example

![[Screenshot 2025-11-14 at 15.14.04.png | 400]]
Note: this plot is taken elsewhere, I used different colors before: here the line is in blue. Also, the y are therein the desidered (target d) values.

The **Mean Square Error (MSE)** is the mean of the square of the error green errors:
![[Screenshot 2025-11-14 at 15.16.06.png| 300]]
where $w$ is the free parameters of the linear model
### Classification Loss
Classification of data into discrete classes.
- **Output**: for example $\{0,1\}$
- **H**: a set of indicator functions
- **Loss function** $L$: measures the classification error

![[Screenshot 2025-11-14 at 15.29.06.png | 400]]

**Definition**: The mean over the data set provide the number/percentage of misclassified patterns. For example 20 out of 100 are misclassified, 20% errors, 80% of accuracy

### Clustering and Vector Quantisation Loss
The goal is to found the optimal partitioning of unknown distribution in $x$-space into regions (clusters) approximated by a cluster center of prototype.
- **H**: a set of vector quantizers $x \to c(x)$ that means continues $\to$ discrete space
- **Loss function** $L$: measures the vector quantizer optimality. A common loss function would be the squared error distortion:

![[Screenshot 2025-11-14 at 15.33.30.png | 400]]

Proximity of the pattern to the centroid of its cluster
# References
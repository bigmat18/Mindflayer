---
Data: 2026-05-21T00:50:00
Tags:
  - note
  - youngling
  - paper
Connection:
  - "[[Training of ELM - IRLS vs Level Bundle Methods]]"
Area: "[[Computational mathematics for learning and data analysis]]"
---
# Extreme Learning Machine

In gereral the speed of feedforward [[Introduction to Neural Networks|Neural Networks]] is far slower that required it has been major bottleneck in their applications for past decades. Two reason:
1. The slow [[Gradient Method]] algortihms extensively used to train neural networks
2. All the parameters of the networks are tuned iteratively by using such learning algorithms

##### Motivations
The Neural Network con feedforward have been extensively used in many fields due to their ability:
- to approximate complex nonlinear mappings directly from the input sample
- to provide models for a large class of natural and artificial phenomena that are difficult to handle using classical parametric techniques

As said above, there is bottleneck in these types of NN that is they are too slow in real context to traine real networks. This is because the standard methods tries to approximate all input sets iteratively using methods absed su **[[Learning Algorithms using Gradient Descent|gradient discen]]t**.

Unlike the popular thinking and most feedforward networks need to be tuned one may not necessarily adjust the input weights and first hidden layer biases  in applications.

### Single Hidden Layer Feedforward Networks (SLFNs)
For $N$ arbitrary distinct samples $(x_i, t_i)$, standard SLFNs with $\tilde{N}$ hidden nodes and activation function $g(x)$ are mathematically modeled as:
$$\sum_{i=1}^{\tilde{N}}\beta_{i}g_{i}(x_{j})=\sum_{i=1}^{\tilde{N}}\beta_{i}g(w_{i}\cdot x_{j}+b_{i})=o_{j}$$
for $j=1,...,N$.

where we have the following variables:
- **$w_i$**: The weight vector connecting the $i$-th hidden node and the input nodes.
- **$\beta_i$**: The weight vector connecting the $i$-th hidden node and the output nodes.
- **$b_i$**: The threshold (or bias) of the $i$-th hidden node.
- **$w_i \cdot x_j$**: The inner product of the vectors $w_i$ and $x_j$.

##### Zero-Error Approximation
If these standard SLFNs can approximate the $N$ samples with zero error, it means that $\sum_{j=1}^{N}||o_{j}-t_{j}||=0$. In this scenario, there exist specific values for $\beta_i$, $w_i$, and $b_i$ such that:
$$\sum_{i=1}^{\tilde{N}}\beta_{i}g(w_{i}\cdot x_{j}+b_{i})=t_{j}$$
for $j=1,...,N$. The $N$ equations above can be also expressed much more compactly as a linear system:

$$H\beta=T$$
Where:
- **$H$**: This is called the **hidden layer output matrix** of the neural network. The $i$-th column of this matrix represents the output of the $i$-th hidden node with respect to all the inputs $x_1, x_2, ..., x_N$.
- **$\beta$**: The output weight matrix.
- **$T$**: The target (expected output) matrix.

> **Theorem 1** : Given a standard SLFN with $N$ hidden nodes and an activation function $g:R\rightarrow R$ which is infinitely differentiable in any interval, for $N$ arbitrary distinct samples $(x,t)$, where $x_{j}\in R^{n}$ and $t_{i}\in R^{m}$, for any $w_{i}$ and $b_{i}$ randomly chosen from any intervals of $R^{n}$ and $R$, respectively, according to any continuous probability distribution, then with probability one, the hidden layer output matrix $H$ of the SLFN is invertible and $||H\beta-T||=0$.

> **Theorem 2**: Given any small positive value $\epsilon>0$ and activation function $g:R\rightarrow R$ which is infinitely differentiable in any interval, there exists $\tilde{N}\le N$ such that for $N$ arbitrary distinct samples $(x_{i},t_{i})$, where $x_{i}\in R^{n}$ and $t_{i}\in R^{m}$, for any $w_{i}$ and $b_{i}$ randomly chosen from any intervals of $R^{n}$ and $R$, respectively, according to any continuous probability distribution, then with probability one, $||H_{N\times\tilde{N}}\beta_{\tilde{N}\times m}-T_{N\times m}||<\epsilon$.

### Proposed Extreme Learning Machine (ELM)
##### Conventional Gradient-Based solutions
Traditionally, in order to train an SLFN, one may with to find specific $w, b, \beta$, which is equivalent to minimizing the cost function
$$
E = \sum_{j=1}^N \bigg( \sum_{i=1}^{\tilde{N}} \beta_i g(w_i \cdot x_j + b_i) - t_j \bigg)^2
$$
When $H$ is uknown gradient-based learning algortihms are generally used to search the minimum of $||H\beta - T||$ . Using the gradient method we need to iteratively adjusted as follows:
$$
W_k = W_{k-1} - \eta \frac{\partial E(W)}{\partial W}
$$
The popular learning algorithms used in feedforward neural networks is the BP learning where gradietns can be computed efficiently by propagation from the output to the input.

There are several issues BP learning algortihms:
1. $\eta$ to slow bring to a very slowly converges. $\eta$ too large bring unstable and diverges
2. Another problem it is that the algorithm stop into the local minimum, that is undesirable if it is loclated far above global minimum.
3. NN could be over-trained using BP algortihms
4. These methods in most application are very time-consuming

##### Proposed Minimum norm least-squares solution
As we saw in **Theorem 1** and **Theorem 2** the input weights $w_i$ and bias $b_i$ can be randomly assigned and than $H$ can actualy remain unchanged. 

So for fixed input weights $w_i$ and hidden layers biases $b_i$, to train SLFN is siimply equivalent to finding a least-sqaures solution for $\beta$ of the linear system $H\beta = T$

- If the number $\tilde{N}$ of the hidden nodes is equal to the number $N$ of distinct training samples $\tilde{N} = N$ the $H$ matrix is square and invertible, the SLDN can approximate these training samples with zero error
- In most cases the number of hidden nodes is much less than the number of distinct trining samples $\tilde{N}<<N$ so $H$ is a nonsquare matrix and there may not exist $w_i, b_i, \beta_i$ such that $H\beta =T$

In the **second case $\tilde{N}<<N$**  the smallest norm least square solution of the above linear system is
$$
\hat{\beta} = H^+T
$$
where $H^T$ is the [[Least Squares#Moore-Penrose Pseudoinverse|Moore-Penrose]] generalized inverse of matrix $H$

##### Proposed Learning Algorithm for SLFN
Given a training set made by $(x_i, t_i)$ activation function $g(x)$ and hidden node number $\tilde{N}$ 
1. Randomly assign input weight $w_i$ and bias $b_i$
2. Calculate the hidden layer output matrix $H$
3. Calculate the output weight $\beta$
$$
\beta = H^+T
$$
where $T = [t_1, \dots, t_N]^T$
###### Remark 1
We have with this tecnquies the following important properties:
1. **Minimum training error**
2. **Smallest norm of weights**
3. The minimum norm leas-squares solution $H\beta = T$ is unique, which is $\hat{\beta}=H^+T$
###### Remark 2
As Shown in **Theorem 2** in theory this algorithm works for any infinitely differential activation function $g(x)$. The only **upper bound** of the required number of hidden nodes is the number of distinct trining samples, that is $\tilde{N}<<N$
###### Remark 3
In the past there has been different proof for SLFN
- **Tamura and Tareishi and Huang**: prove that SLFN with randomly chosen sigmodal hidden nodes can exactly learn N distict observation
- **Huang**: prove that it the input weights and hidden biases are allowd to be tuned, SLFN with most N hidden nodes and with almost any nonlinear activation function (diff and non diff) can exactly learn N distinct observations
- **Ferrati and Stengel**: with N sigmoidal hidden nodes and with input weights randomly generated but **hidden bias appropriately tuned** can exactly learn N distinct observation

This paper add both in the proof, weights and hidden bias randomly generated bring to N distinct observation learned.
###### Remark 4
**Modular Networks** have also been suggested in several works,, which partition the trining samples into $L$ subsets each lerned by an SLFN separately. 

These SLFNs can actually share common hidden nodes. That means, the ith hidden node of the first SLFN can also work as the ith hidden node of the rest SLFNs and the total number of hidden nodes requires in these L SLFNs is still $max_i(s_i)$
###### Remark 5
Several method can be used to calculate the Moore-Penrose generalized inveerse of $H$. These methods may include, but are not limited to:
- orthogonal projection
- orthogonalization method
- iterative method
- singular value decomposition (SVD)

The **ortogonal projection** can be used when $H^TH$ is non singular but not always it is non singular or may tend to be singular in some applications and thus orthogonal projection method may not perform well in all applicaiton, so the paper suggest to use SVD.
# References
- [[Extreme_Learning_Machine_Theory_Applications.pdf]]
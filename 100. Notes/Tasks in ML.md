---
Data: 2025-11-14T14:42:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Introduction to Machine Learning]]"
Area: "[[Master's degree]]"
---
# Tasks in ML

The task defines the purpose of the application:
- Knowledge that we want to achieve? (e.g. pattern in DM or model in ML)
- Which is the helpful nature of the result?
- What information are available?
###### Predictive Tasks (Mainly in ML)
Like Classification and Regression: function approximation

![[Pasted image 20251111124957.png | 550]]
Example recall the **"pilot" example on handwritten digits**: build a function from examples
###### Descriptive
Like Cluster Analysis, Association Rules: find subsets or groups of unclassified data

#### Supervised Learning
**Given**: Training examples as `<input, output> = <x,d>` (**labeled examples**) for an unknown function $f$ (know only at the given points of example)
- Target value: desiderate value `d` or `t` or `y` ... is given by the teacher according to $f(x)$ to lavel the data.

**Find**: A good approximation to $f$ (a hypothesis h that can used for prediction
on unseen data $x’$, i.e. that is able to generalize)

![[Pasted image 20251111125400.png]]

Target $d$ (or $t$ or $y$): a categorical or number label
- **Classification**: discrete value outputs:
$$
f(x) \in \{1, 2, 3, \dots, k\} \text{ classes (discrete-valyed function)}
$$
- **Regression**: real continuous output values (approximate a real-valued target function, in $R$ or $R^k$)

#### Unsupervised Learning
There is not teacher, and the TR (training set) is a set of unlabeled data `<x>`. Example is to find natural groupings in a set of data
- Clustering
- Dimensionality reduction/ Visualization / Pre-processing 
- Modeling the data density

![[Pasted image 20251111130659.png | 550]]

### Classification 
Classification (supervised): patterns (features vectors) are seen as members of a class and the goal is to assign the patterns observed classes (label)
- Classification: $f(x)$ return the correct class for $x$
- Number of classes:
	- **= 2**: $f(x)$ is a Boolean function: binary classification, **concept learning** (T/F or 0/1 or -1/+1 or negative/positive)
	- **> 2**: multi-class problem ($C_1, C_2, C_3, \dots, C_k$)

###### Example
From DATA to TASK (example classification). Terminology:
- Inputs are the "independent variables"
- Outputs are the "dependent variables" or "responses"

![[Pasted image 20251111131303.png]]

The classification may be viewed as the allocation of the input space in decision regions (e.g. 0/1).

**Example**: graphical illustration of a linear separator on a instance space $x^T = (x_1, x_2) in \mathbb{R}², f(x) = 0/1 (or -1/+1)$

![[Pasted image 20251111131624.png | 700]]

**Geometrical 3D view: Classifier**
![[Pasted image 20251111131741.png | 500]]
### [[Regression Models]]
Process of estimating of a real-value function on the basis of finite set of noisy samples (supervised task) known pairs ($x$, $f(x) + \text{ random noise}$) the task is find $f$ for the data.

![[Pasted image 20251111131942.png | 600]]

**Regression**: $x = variables$ (eg real values), $f(x)$ real values: curve fitting ($x$ is 1-dim in the example but it becomes k-dim in general)

Process of estimating of a real-value function on the basis of finite set of noisy samples.  Known pairs ($x$, $f(x) + \text{ random noise}$)

![[Pasted image 20251111132255.png | 550]]

# References
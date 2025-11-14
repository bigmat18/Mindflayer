---
Data: 2025-11-11T12:02:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
Area: "[[Master's degree]]"
---
# Introduction to Machine Learning

## Data
The data represent the available fact (experience) and are used to capture the structure of the analyzed objects.
#### Flat
Are data with attribue-value language, they are fixed-size vectors of properties (features), with a single table of tuple (measurements of the objects).

![[Pasted image 20251111121128.png]]

For flat data can be done **numerical encoding** for same categories, example:
- 0/1 (or -1 / +1) for 2 classes
- For more classes:
	- 1, 2, 3, ... (**Warning**: grade of similarity (1 vs 2 or 3):  useful for "order categorical" variables as small, medium, large)
	- **1-of-k** (or 1-hot) encoding: useful for symbols

![[Pasted image 20251111121450.png | 600]]

- **Dimension of data**: number of examples $l$
- **Dimension** (of input $x$): number of features $n$
- If we will index the features/inputs/variables by $i$ or $j$: variable $x_i$ is (typically) the $i-th$ feature/property/attribute/element/component of $x$
- $x_p$ (or $x_i$) is (typically) the $p-th$ (or $i-th$) pattern/example/row
- $x_{p,i}$ can be attribute $i$ of the pattern $p$.
#### Structures
**Structured** data are for examples lists, trees, graphs, multi-relational data (table). 

For example: images, microarray, temporal data, strings of a language, DNA e proteins, hierarchical relationships, molecules, hyperlink connectivity in web pages, ...

![[Pasted image 20251111121607.png]]
#### Noise
Addition of external factors to the stream of (target) information (signal); due to randomness in the measurements, not due to the underlying law, example the Gaussian noise.

![[Pasted image 20251111124538.png]]
#### Outliers
Are unusual values that re not consistent with most observations (es due to abnormal measurements errors). To avoid it:
- outlier detection in preprocessing operations: remove
- robust modeling methods
#### Features Selection
Selection of a small number of informative features: it can provide an optimal input representation for a learning problem.

## Tasks
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
#### Classification 
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
#### Regression
Process of estimating of a real-value function on the basis of finite set of noisy samples (supervised task) known pairs ($x$, $f(x) + \text{ random noise}$) the task is find $f$ for the data.

![[Pasted image 20251111131942.png | 600]]

**Regression**: $x = variables$ (eg real values), $f(x)$ real values: curve fitting ($x$ is 1-dim in the example but it becomes k-dim in general)

Process of estimating of a real-value function on the basis of finite set of noisy samples.  Known pairs ($x$, $f(x) + \text{ random noise}$)

![[Pasted image 20251111132255.png | 550]]

## Models
Aim: to capture/describes the relationships among the data (on the basis of the task) by a “language” (numerical, symbolic, …). The “language” is related to the representation used to get knowledge. The model defines the class of functions that the learning machine can implement (hypotheses space)

**Example**: set of functions $h(x, w)$ where $w$ is the (abstract) parameter.

- **Training example** (superv.): An example of the form $(x, f(x)+noise)$, $x$ is usually an input vector of features, ($d$ or $t$ or) $y=f(x)+noise$ is called the target value
- **Target function**: the true function $f$
- **Hypothesis**: A proposed function $h$ believed to be similar to $f$. An expression in a given language that describes the relationships among data.
- **Hypothesis space** H: The space of all hypotheses (specific models) that can, in principle be output by the learning algorithm
# References
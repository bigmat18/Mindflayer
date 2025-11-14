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
# Data in ML

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
# References
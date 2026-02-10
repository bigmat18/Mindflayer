---
Data: 2026-02-10T17:59:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Multidimensional Data Visualization

### Multi-set Bar Charts
- Also called grouped bar charts.
- Like bar chart for more datasets.
- Allow accurate numerical comparisons.
- It can be used to show mini-histograms.

![[Pasted image 20260210165922.png | 400]]

###### Multiple bars
Distributions of each variable among the different categories/data points.
![[Pasted image 20260210165946.png]]

###### Multiple Views
Distributions of each variable among the different categories/data points (each variable has its own display).
![[Pasted image 20260210170026.png]]

###### Stacked Bar charts
Each dataset is drawn on top of each other. More segments each bar more difficult to read.
![[Pasted image 20260210170120.png | 300]]

- **Simple Stacked Bar Charts:** Useful if the visualization of the absolute values (and their sum) is meaningful.
- **Percentage Bar Charts:** Better to show the relative differences between quantities in the different groups.

### Spineplots
- Generalization of stacked bar charts.
- Special case of mosaic plot.
- Permit to show both percentages and proportions between variables.

###### Example (car data)
![[Pasted image 20260210170252.png | 350]]

![[Pasted image 20260210170318.png | 350]]

Convey both percentages and proportions.
![[Pasted image 20260210170416.png | 400]]

### Mosaic Plots
They give an overview of the data by visualizing the relative proportions. Also known as Mekko charts.
![[Pasted image 20260210180847.png | 500]]
###### Example: Titanic data (from Wikipedia)
![[Pasted image 20260210180918.png | 500]]

- Variable gender -> vertical axis.
![[Pasted image 20260210181004.png |300]]

- Add variable Class -> Horizontal axis.
![[Pasted image 20260210181024.png | 300]]

- Add variable survived -> vertical axis.
![[Pasted image 20260210181043.png | 300]]

**Advantages**:
- Maximal use of the available space
- Good overview of the proportions between data
- Good overview of the variable dependency

**Disadvantages**
- Extension to many variables is difficult
### [[Treemaps]]

### [[Scatterplots]]

### [[Chernoff faces]]

### [[Multidimensional icons]]
### Parallel Coordinates 
Originally attributed to Philbert Maurice d'Ocagne (1885). Extends classical Cartesian Coordinates
System to visualize multivariate data. Re-discovered and popularized by Alfred Isenberg in 1970s.

###### Example
- Two variables
![[Pasted image 20260210181420.png | 400]]

- Three variables
![[Pasted image 20260210181440.png | 400]]

- Four Variables
![[Pasted image 20260210181505.png | 400]]

- N variables
![[Pasted image 20260210181526.png | 500]]

Order of the axes play a fundamental role in readability. But we have n! combinations. Many strategies have been investigating for axes re-ordering.

###### Example Re-ordering and edge-bundling
![[Pasted image 20260210181616.png | 500]]

### Star Plot
Known with many names: radar chart, spider chart, web chart, etc. Analogous to parallel coordinates, but the axes are positioned in polar coordinates (equi-angular). Position of the first axis is uninformative.

![[Pasted image 20260210181659.png]]

- Easy to compare properties of a class of objects or a category.
- Not easy to understand trade-off between different variables.
- Not suitable for many variables or many data.
# References
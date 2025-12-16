---
Data: 2025-12-16T16:47:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Symbol maps

A symbol (a disk, a square, a shape) is placed in a point and scaled such that its area is proportional to the quantity associated to the point. We can use area, shape, and color as channels:
- area and shape again at our disposal. we could not use them with choropleths
- symbols avoid confusing geographic area with data values

![[Pasted image 20251216164830.png | 450]]

One can even use graphs/charts/complex glyphs as symbols, even encoding multiple attributes

![[Pasted image 20251216164910.png | 500]]

###### Pros
- intuitive and easy to understand 
- they solve the issue of region dimension vs the value of the attribute visualized 
	- the symbol dimension is proportional to data values and not to the region dimension
	- (complex) symbols can have uniform dimension and even consist of other charts/diagrams
###### Cons
- complex symbols can be harder to understand
- occlusions, overlaps
	- some symbols can hide other symbols 
	- some symbols can hide region boundaries

![[Pasted image 20251216165136.png | 500]]

### Dot density maps
Marks, typically points, are positioned over a map
- each dots represents a given constant quantity of items (from 1 to N)
- all dots are equal: it is their quantity and distribution that gives information

![[Pasted image 20251216165232.png | 500]]

Normalizations may be needed: (e.g., 1 dot = # cases / 100 people)
###### Pros
- easy and intuitive
- mitigate the confusion between region areas and displayed values
###### Cons
- accurate estimates are difficult
- occlusions, overlaps
- performance if high numbers of dots must be visualized

# References
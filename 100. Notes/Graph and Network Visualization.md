---
Data: 2025-12-16T17:17:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Graph and Network Visualization

[[Graph Algorithms|Graph]] are represented by a set of notes V end a set of edges E

![[Pasted image 20251216171824.png | 400]]

###### Drawing Conventions
![[Pasted image 20251216171920.png | 400]]
###### Drawings on the grid
![[Pasted image 20251216171937.png | 400]]

### Aesthetics
###### Few Crossings
The drawing is clearer if the number of intersections is reduced. Below a view of a non-planar and a planar drawing of the same graph.
![[Pasted image 20251216172021.png | 300]] ![[Pasted image 20251216172040.png | 300]]

###### Small size
A small size allows to see a larger picture of the drawing. You may minimize the area, the maximum
edge length, the average edge length, etc
![[Pasted image 20251216172156.png | 400]]

###### Number of bends
Edges with many bends are difficult to follow
![[Pasted image 20251216172220.png | 400]]

###### Contrast of aesthetics
Two drawings of the same graph within the “orthogonal on the grid” drawing convention
![[Pasted image 20251216172326.png | 450]]

### Non-planar graphs
A drawing is planar if it does not have intersections (edge crossings). A graph is planar if it admits a planar
drawing. Not all graphs are planar

Two examples of non-planar graphs are:
- complete biparte graph with six vertices ($K_{3,3}$)
![[Pasted image 20251216172926.png | 200]]

- complete graph with five vertices ($K_5$)
![[Pasted image 20251216172938.png | 200]]

**Definition**: a graph is non-planar if and only if it contains a subdivision of $K_5$ or $K_{3,3}$

## Overview and details
Many data sets are too large to visualize on one screen (**scale problem**)
- too many items (data cases)
- too many variables (attributes)

An excess of graphic objects produces an information overflow and also there is a computational and memory scalability problem.

One of the fundamental challenges in information visualization is How to allow end-user to work with, navigate through, and generally analyze a set of data that is too large to fit in the display.

Overview and details: 
- An overview of the data set can be extremely valuable
	- helps present overall patterns
	- assists user with navigation and search
	- orients activities
- Details are also important
	- viewers also will want to examine individual cases and variables
	- not all details are of interest, but only a few at a time
	- generally provide details on demand

Overviews of the data set can follow two paradigms:
###### Perspective view
- offers a panoramic view of the data
- allows to get the big picture
- high level categories and trends can be inferred

###### Compendium view
- offers a sketchy view of the data
- allows to get a compendium
- high level categories and trends are directly shown

Overview and details can be combined in two ways:

###### Space
- use different portions of screen to show overview and details
- focus + context techniques
###### Time 
- alternate between overview and details sequentially in same place
- zooming, panning, scrolling


### Focus + context
**Target**: show focus (details) and context (overview) at the same time
**Classification**:
- single view (no focus + context)
- coordinated pair
- view and thumbnails
- fisheye view

##### Single View (details-only)
Interaction by zooming and panning. Works better when:
- zoom factor is relatively small
- far away context may be neglected

![[Pasted image 20251216175153.png | 250]]

##### Coordinated pair
Combined display of the overview and local magnified view
- relative importance of overview and detail?
- two separate views?
- overview as an inset?

![[Pasted image 20251216175238.png]]

##### Thumbnail documents overview
Example is power point side overview.

![[Pasted image 20251216175301.png | 450]]

### Interactive approaches
Interactive approaches for visualizing and exploring large graphs
- graph visualized incrementally or at different levels of details
- strong interaction between the user and the drawing
##### Top-down exploration
Based on Overview first, zoom and filter, then details on demand

 An overview or sketch of the information is provided
- The user interactively explores the data asking for more details
- Typical sketches are obtained by
	- **[[Filtering]]** (deleting) nodes and edges
	- **sampling** (retaining only a fraction)
	- **[[Clustering]]** (merging) nodes
###### Limitations
- preservation of the user mental map is difficult
- the definition of overview is dependent from the problem at hand

##### Bottom-up exploration
Based on Search, show context, expand on demand. The graph is visualized a piece at a time
- topological window moving through canvas
- incremental enhancement of the drawing
###### Limitations
- no overview

# References
---
Data: 2025-12-15T13:06:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Data and Task Abstractions

###### Example 1
- Dataset: Table taken from Il Sole 24 Ore
- Data visualized: laurea degrees in different fields, % of men and women
- Graphical representation: Bar chart

![[Pasted image 20251215134928.png | 400]]

###### Example 2
- Dataset: Table taken from Eurostat
- Data visualized: difference in salaries according to gender in EU
- Graphical representation: Area chart

![[Pasted image 20251215135007.png | 350]]

###### Example 3
- Dataset: Table about Pride and Prejudice
- Data visualized: frequency of word usage
- Graphical representation: Word cloud

![[Pasted image 20251215135047.png | 400]]

## Data semantics and data type
### Semantics
The **semantics** of the data is its real-world meaning
- Apple: is it a first name, the name of a company, a fruit, a city?
- 12: day of a month, month in a year, age, height, coordinate?

The data semantics should be provided by the dataset creator (as metadata)

![[Pasted image 20251215135158.png | 400]]

### Types
The **type** of the data is its mathematical or structural interpretation. Is it an item, an attribute, a link, ...?
The five basic data types in visualization Munzner, 2014

![[Pasted image 20251215135256.png | 450]]

#### Items
**Items**: individual entities that are discrete in the dataset
- e.g., people, market stocks, cities...
- in practice, the objects/entities you want to visualize

![[Pasted image 20251215140206.png | 500]]
#### [[Attributes]]
#### Links
A **link** is a relationship between items, typically within a network.

![[Pasted image 20251215142531.png | 500]]
#### Positions
A **position** is a spatial data, providing a location in either 2D or 3D space
- e.g., (latitude,longitude) pair describing a location on the Earth’s surface;
- e.g., (x,y,z) specifying a location within the region of space measured by a medical scanner

![[Pasted image 20251215142726.png | 550]]
#### Grids
**Grids** specify the strategy for sampling continuous data in terms of both
geometric and topological relationships between its cells

![[Pasted image 20251215142748.png |550]]

### Dataset Types
A **dataset** is a collection of information that is the target of analysis. The four basic dataset types are tables, networks, fields, and geometry.

![[Pasted image 20251215142917.png | 600]]

The four basic dataset types - tables, networks, fields, and geometry - arise from combinations of the data types of items, attributes, links, positions, and grids.

![[Pasted image 20251215142952.png | 550]]
#### [[Tables]]
#### [[Networks]]
#### [[Fields (Continues)]]
#### [[Geometry (Spatial)]]

### Datasets availability
- **Static (or offline) dataset**: the dataset is available all at once
- **Dynamic (or online) dataset**: the dataset is a dynamic stream, with information trickling in over the duration of the visualization session
	- e.g., adding/removing items, or changing attribute values

![[Pasted image 20251215152843.png | 400]]
# References
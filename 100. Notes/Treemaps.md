---
Data: 2025-12-16T16:17:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Treemaps

Alternative way to visualize hierarchies, by displaying quantities via area size:
- Each rectangle represents a category, with nested subcategory rectangles
- Child nodes are contained in the area region representing the parent node («Contained in» means «child of»)
- When a quantity is assigned to a category, its area size is displayed in proportion to that quantity and to the other quantities within the same parent category in a part-to-whole relationship
	- the area size of the parent category is the total of its subcategories
- Different alternative tiling strategies

![[Pasted image 20251216161907.png | 250]]

- Compact and space-efficient
- Less good at showing the levels in the hierarchy than e.g. sunburst charts
	- only leaf attributes are displayed
- Good for an overview of structures
	- useful for comparing proportions via area size, but not always accurate

![[Pasted image 20251216162011.png]]

![[Pasted image 20251216162033.png]]

# References
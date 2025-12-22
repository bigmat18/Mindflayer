---
Data: 2025-12-15T19:19:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Charts

![[Pasted image 20251215192010.png]]

## Charts for arranging [[Tables|tables]]
A **key** is an independent attribute that can be used as a unique index to look up items in a table, while a **value** is a dependent attribute (the value of a cell)
- key attributes can be categorical or ordinal, whereas values can also be quantitative

To decide on the visualization, ask yourself how many keys and how many values are there in the table.

For example:
- Two values, no key: Possible visualization: Scatterplot
- One key, one value: Possible visualization: Bar chart
- Two keys, one value: Possible visualization: Heatmap

Using space to express quantitative attributes is a straightforward use of the spatial position channel to visually encode data. The attributes are mapped to spatial position along axes
#### [[Scatterplots]]
#### [[Bar Charts]]
#### [[Histograms]]
#### [[Line Charts]]


## Chart multiple attributes
#### [[Bubble Charts]]
#### [[Stacked Bar Charts]]
#### [[Stacked line charts]]
#### [[Flow maps]]
#### [[Heatmaps]]
#### [[Wordclouds]]
#### [[Multidimensional icons]]
#### [[Petals as a glyph]]
#### [[Chernoff faces]]


## Radial and Maps
#### Axis Layout
Spatial axes are an important ingredient of visualizations.
- labelling
- starting value: values different from 0 can yield to misleading visualizations but can also augment perception if necessary
- orientation: rectilinear, parallel, or radial layouts:

![[Pasted image 20251216153519.png | 450]]

 **Rectilinear**: items distributed along two perpendicular axes, with values ranging from a minimum value on one side to a maximum value on the other side

![[Pasted image 20251216153350.png]]

**Parallel**: axes placed parallel to each other
![[Pasted image 20251216153547.png | 450]]

**Radial**: items distributed around a circle, using the angle channel
![[Pasted image 20251216153603.png]]

The natural coordinate system in radial layouts is polar coordinates
- one dimension is measured as an angle from a starting line
- the other dimension is measured as a distance from a center point

![[Pasted image 20251216154530.png | 450]]

Rectilinear and radial layouts are not equivalent from a perceptual point of view
- the angle channel is less accurately perceived than a rectilinear spatial position channel
	- recall the ranking of channels and the effectiveness principle
- the angle channel is cyclic, because the starting and ending point are the same, as opposed to the linear nature of position
	- recall the semantics of attributes and the expressiveness principle
#### [[Radial bar Charts]]
#### [[Pie Charts]]


## Visualizing a [[Networks|Tree]] dataset
#### [[Sunburst Charts]]
#### [[Treemaps]]
#### [[Bubble hierarchies]]


## Visualizing [[Geometry (Spatial)|spatial/geometric data]]
The given spatial position is an attribute of primary importance, the central tasks often revolve around understanding spatial relationships, useful for:
- Geographic data
- Computer Graphics and Geometry Processing

![[Pasted image 20251216163538.png]]

A sensible visual encoding choice is to use the provided spatial position as the substrate for the visual layout
- rather than to visually encode other attributes using the spatial position channel
#### [[Choropleth maps]]
#### [[Symbol maps]]
#### [[Contiguous Cartograms]]

# References
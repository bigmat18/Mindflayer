---
Data: 2025-12-15T20:17:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Heatmaps

A 2-dim matrix alignment to arrange two keys and a value
- one key is distributed along the rows and the other along the columns
- each rectangular cell in the matrix is occupied by an area mark encoding a single quantitative value attribute with color

![[Pasted image 20251215201841.png | 500]]

Benefits: visually encoding quantitative data with color using small area marks is very compact
- good for overviews with high information density
- good scalability
- e.g., a matrix of 200 x 200 with 40K items is easily handled (limit: up to a pixel per cell...)

Drawbacks: [[Discriminability]]
- only a small number of different levels of the quantitative attribute can be distinguishable, because of the limits of colour perception in small, non-contiguous regions
- up to 11 bins on a display size of 1000 x 1000 Munzner 2014
- also, not suited to color-blind people

Keys can be ordered (even if they are categorical) to identify patterns

![[Pasted image 20251215202005.png | 600]]
# References
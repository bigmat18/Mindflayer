---
Data: 2025-12-15T19:26:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Scatterplots

Scatterplots encodes two quantitative attributes (values) using the vertical and horizontal spatial position channels; the mark type is a point. Scatterplots are effective for:
- providing overviews and characterizing distributions
- and specifically for finding outliers and extreme values
- judging the correlation between two attributes
	-  with this visual encoding, that task corresponds the easy perceptual judgement of noticing whether the points form a line along the diagonal. The stronger the correlation, the closer the points fall along a perfect diagonal line; positive correlation is an upward slope, and negative is downward.

![[Pasted image 20251215193931.png | 350]]

When judging correlation is the primary intended task, a regression line is often superimposed on the raw scatterplot. recall: contextual information

![[Pasted image 20251215194036.png | 300]]

To shed more light on the correlation, one can use transformations
- e.g., logarithmically scaling

![[Pasted image 20251215194105.png | 400]]

The scalability of a scatterplot is limited by the need to distinguish points from each other, so it is well suited for dozens or hundreds of items. The scale is in the order of hundred

#### Slope charts
Alternative to scatter plots: parallel axes, each item is a line connecting two quantities

![[Pasted image 20251215194147.png | 500]]

### Multidimensiona Scatter Plot
Each possible pair of variables is represented in a standard 2D scatter plot.
![[Pasted image 20260210181217.png]]
# References
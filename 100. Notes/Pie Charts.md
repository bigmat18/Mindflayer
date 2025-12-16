---
Data: 2025-12-16T15:52:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Pie Charts

This is the most commonly used radial graphic, pie charts use area marks (2D polygons) and the angle (arc length) channel. Used to show percentages and proportions among categories:
- by dividing a circle into proportional segments, with each angle/arc length representing the value/proportion for each category
- the full circle represents the sum of all data (i.e., 100%)

![[Pasted image 20251216155319.png | 300]]

Despite their popularity, pie charts are clearly problematic when considered according to the visual channel properties we have discussed
- useful for giving a quick idea and to compare one slice vs total, not good for accurate comparisons
	- angle judgements on area marks are less accurate than length judgements on line marks
	- the wedges vary in width along the radial axis, from narrow near the center to wide near the outside, making the area judgement particularly difficult

![[Pasted image 20251216155458.png]]

- space occupancy
- scalability: up to about one dozen categories make sense

![[Pasted image 20251216160022.png | 600]]

The most useful property of pie charts is that they show the relative contribution of parts to a whole
- the sum of the wedge angles must add up to the 360 degrees of a full circle, as with normalized data (such as percentages) where the parts must add up to 100%

However, this property is not unique to pie charts: a single bar in a normalized stacked bar chart would also do the job with the more accurate channel of length.

![[Pasted image 20251216160114.png]]

### Incomplete pie charts
Showing a single slice to underline a value in the dataset.

![[Pasted image 20251216160215.png | 450]]

### Donuts charts
- Pie charts with the center area cut out
- More focus on arc length than on area
- More space efficient: also more space for labels

![[Pasted image 20251216160416.png | 400]]
### Polar area charts
Polar area charts vary the length of the wedge, rather than varying the angle as in a classic pie chart
- just as a bar chart varies the length of the bar

![[Pasted image 20251216160357.png | 400]]

### Radar charts (or spider charts)
- Items displayed along circles
- Useful for periodic data

![[Pasted image 20251216160512.png | 400]]

# References
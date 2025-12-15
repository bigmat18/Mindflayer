---
Data: 2025-12-15T20:12:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Stacked line charts

Values across categories are stacked one on top of the other. Popular in recent years, yet they have noticeable limitations
- not good for reading and comparing individual values over time
- since the values are stacked, the shape of lines (and therefore the pattern you see) is affected by the shape of the lines below the one you are observing
- interaction/filtering to compensate: drill down into a subset of individual series
- they are meaningless for data that should not be summed (e.g., temperature)

![[Pasted image 20251215201348.png]]

### Line chart series
- Collection of lines in the same chart
- Easy to compare values
- Though, too many overlapping curves can make the graph hard to read
	- possible solutions: either deciding and filtering what to show, or using grey

![[Pasted image 20251215201500.png | 450]]


### Small multiples
- An alternative way to show multiple attributes that holds for all types of charts
- Create multiple replicas of the same graph, with each graph in its own chart
- Simple but often a good idea
	- more effective visualization than a single plot including too many data

![[Pasted image 20251215201553.png| 450]]
# References
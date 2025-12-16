---
Data: 2025-12-16T16:40:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Choropleth maps

A choropleth map shows regions as area marks, and a quantitative attribute encoded as color over regions
- also texture can be used as a channel

![[Pasted image 20251216163723.png | 500]]

Usually employed for showing how a quantity distributes across geographical areas/regions
- showing where people lives, rather than the actual data...
- consider normalization, e.g. by the number of people living in the region

The major design choices for choropleths are:
- how to construct the colormap
- what region boundaries to use (spatial aggregation)

###### Example
- US unemployment rates from 2008
- a sequential, white-to-blue colormap
	- nine levels with monotonically decreasing luminance
- region granularity is counties within states

###### Example
Bush vs Kerry US elections in 2004

![[Pasted image 20251216163857.png | 500]]

- Useful to display how data distributes over geographical regions
- The ideal situation would be when regions have approximately the same area
- Always ask yourself whether data should be normalized
- Pay attention to the choice of the colormap according to the semantics of data
	- pay attention to the number of bins (discriminability)

###### Pros
- easy to understand
- familiar, natural representation (differently from conventional representations)

###### Cons
- passing on the wrong message
- communicating area rather than data
- strongly depending on colors
# References
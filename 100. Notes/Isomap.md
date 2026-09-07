---
Data: 2026-07-19T14:53:00
Tags:
  - note
  - padawan
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Isomap

The core idea is to preserve the [[PDS on Surface#PDS on Surface|Geodesic Distance]] between data points. Geodesic is the shortest path between two points on a curved space.

![[Pasted image 20260210230742.png | 250]]

![[Pasted image 20260210230750.png]]

1. **Construct neighborhood graph:** Define graph G over all data points by connecting points $(i,j)$ if and only if the point i is a K neareast neighbor of point j
2. **Compute the shortest path.** Using the Floyd’s algorithm. It is an algorithm to find the shorted paths between all pairs of vertices in a weighted graph
3. **Construct the d-dimensional embedding**

![[Pasted image 20260210231011.png | 550]]

![[Pasted image 20260210231031.png | 550]]

# References
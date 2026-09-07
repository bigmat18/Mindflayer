---
Data: 2026-08-16T14:43:00
Tags:
  - note
  - master
  - article
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Matrix Traversal

Most problem that ask to traverse a **Matrix** can be seen as graph problems. The key insight of this intuation is that a grid is really just a **graph in disguise**. Each cell is a node, and adjacent cells (up, down, left, right) are connected by edges. This mental model unlocks all the graph traversal algorithms.
- **Nodes**: Each cell (i, j) is a node
- **Edges**: Connect to adjacent cells (typically 4 or 8 neighbors)
- **No explicit edge list**: Neighbors are computed on demand using direction arrays

![[Pasted image 20260816144953.png | 500]]

For this reason we can use for the most of the problems the classic algortihms that we use to solve graph problems: [[Ricerca in ampiezza (BFS)]], [[Ricerca in profondità (DFS)]]


# References
- https://algomaster.io/learn/dsa/matrix-traversal-introduction
- https://www.youtube.com/watch?v=DjYZk8nrXVY&t=88s
# Leetcode
- [ ] [733. Flood Fill](https://leetcode.com/problems/flood-fill/)
- [ ] [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [ ] [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)
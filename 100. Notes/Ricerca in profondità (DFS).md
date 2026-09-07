**Data time:** 01:34 - 11-05-2025

**Tags:** [[Introduction to Artificial Intelligence]] [[Agenti Risolutori di problemi (Finging Algorithm)]] [[Graph Algorithms]]

**Area**: [[Bachelor's Degree]]
# Ricerca in profondità (DFS)

This is useful to explore all path or branch in a graphs or trees to solve problems like:
- finding a path between two nodes (not shortest, for that [[Ricerca in ampiezza (BFS)|BFS]] is better)
- checking if a graphs contains a cycle
	- **Undirected**: you can use both **DFS** or [[Ricerca in ampiezza (BFS)|BFS]]
	- **Directed**: if a node is visited yet doesn't mean that exist a cycle, so you have to use **DFS**
- finding a topological over in a directed acyclic graph
- counting number of connected components in a graph
- Counting all possible path ([[Backtracking]])

```python
def recursive_depth_first_search(problem, node):
	#controlla se lo stato fel nodo è uno stato obbiettivo
	if problem.goal_test(node.state):
		return node.solution()
	#in caso contrario continua
	for action in problem.actions(node.state):
		child_node = node.child_node(proble, action)
		result = recursive_depth_first_search(problem, chidl_node)
		if result is not None:
			return result
	return None
```

The idea is to traverse all adjacent vertices one by one. When we traverse an adjacent vertex we completrly visti all vertices reachable thought that adjacent vertex. This is similar to [[Binary Tree Traversal#Pre-Order BT Traversal|Preorder Traversal in Binary Tree]], the key difference is that graphs may contain cycles, to avoid this we use a boolean visited array to flag the already visited nodes
#### Analisi
- Utilizza una coda di tipo **LIFO**
- **Strategia completa**: Si
- **Strategie ottimale**: Si
- **Complessità in tempo**: $O(V + E)$
- **Complessità in spazio**: $O(V + E)$

### Ricerca in profondità limitata (DL)
This version use the same idea but, instead of continuity go deeply until we don't find any other unvisited node, we se a max level $l$ of depth, and we will go down until $l$
#### Analisi
- **Strategia completa**: Si per problemi in cui si conosce un limite superiore per la profondità della soluzione. Completa se D < l
- **Strategie ottimale**: No
- **Complessità in tempo**: $O(b^l)$
- **Complessità in spazio**: $O(b \cdot l)$


# References
- https://www.geeksforgeeks.org/dsa/depth-first-search-or-dfs-for-a-graph/
# Leetcode
- [ ] [133. Clone Graph](https://leetcode.com/problems/clone-graph/)
- [ ] [113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)
- [ ] [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)

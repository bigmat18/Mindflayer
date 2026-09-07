**Data time:** 01:34 - 11-05-2025

**Tags:** [[Introduction to Artificial Intelligence]] [[Agenti Risolutori di problemi (Finging Algorithm)]] [[Graph Algorithms]]

**Area**: [[Bachelor's Degree]]
# Ricerca in ampiezza (BFS)

This is another technique used with graph, similar to [[Ricerca in profondità (DFS)]]. This is a traversal technique that explore nodes level by level, is the same approach also used in [[Binary Tree Traversal#Level-Order BT Traversal|Binary tree Level order traversal]].

It is very useful for problem like:
- Finding the shorter path between two nodes (NOT [[Ricerca in profondità (DFS)|DFS]])
- Printing all nodes of a tree level by level
- Finding all connected components in a graph (both BFS or [[Ricerca in profondità (DFS)|DFS]])
- Finding the shortest transformation sequence from one word to other

```python
def breadth_first_search(problem): # """Ricerca-grafo in ampiezza"""
	# insieme degli stati gia visitati (implementato come una lista)
	explored =[] 
	# il costo del cammino e inizializzato nel costruttore del nodo
	node = Node(problem.initial_state) 
	
	if problem.goal_test(node.state):
		return node.solution(explored_set =explored)
		
	frontier =FIFOQueue() # la frontiera e una coda FIFO
	frontier.insert(node)
	
	while not frontier.isempty(): # seleziona il nodo per l espansione
		node = frontier.pop()
		# inserisce il nodo nell insieme dei nodi esplorati
		explored.append(node.state) 
		
		for action in problem.actions(node.state):
			child_node =node.child_node(problem,action)
			if (child_node.state not in explored) and 
			   (not frontier.contains_state(child_node.state)):
				if problem.goal_test(child_node.state):
				return child_node.solution(explored_set =explored)
		# se lo stato non e uno stato obiettivo 
		# allora inserisci il nodo nella frontiera
		frontier.insert(child_node)
		
	# in questo caso ritorna con fallimento
	return None 
```

First, it visits all nodes directly adjacent to the source. Then, it moves on to visit the adjacent nodes of those nodes, and this process continues until all reachable nodes are visited
- BFS is different from [DFS](https://www.geeksforgeeks.org/dsa/depth-first-search-or-dfs-for-a-graph/) in a way that closest vertices are visited before others. We mainly traverse vertices level by level
- There are a popular graph algorithm like [Dijkstra's shortest path](https://www.geeksforgeeks.org/dsa/dijkstras-shortest-path-algorithm-greedy-algo-7/), [Kahn's Algorithm](https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/), and [Prim's algorithm](https://www.geeksforgeeks.org/dsa/prims-minimum-spanning-tree-mst-greedy-algo-5/)  based on BFS.
- BFS itself can be used to detect cycle in a directed and undirected graph, find shortest path in an unweighted graph and many more problem

Utilizza una coda di tipo **FIFO**. Definiamo:
- **B** = fattore di ramificazione
- **D** = profondità del noto obbiettivo
- **M** = lunghezza massima dei cammini nello spazio degli stati
#### Analisi
- **Strategia completa**: Si
- **Strategie ottimale**: Si
- **Complessità in tempo**: $O(b^2)$
- **Complessità in spazio**: $O(b^d)$

# References
- https://www.geeksforgeeks.org/dsa/breadth-first-search-or-bfs-for-a-graph/
- [[IIA_notes.pdf#page=19]]
# Leetcode
- [ ] [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [ ] [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)
- [ ] [127. Word Ladder](https://leetcode.com/problems/word-ladder/)
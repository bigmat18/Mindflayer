**Data time:** 01:34 - 11-05-2025

**Tags:** [[Introduction to Artificial Intelligence]] [[Agenti Risolutori di problemi (Finging Algorithm)]] [[Graph Algorithms]]

**Area**: [[Bachelor's Degree]]
# Ricerca in ampiezza (BF)


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

![[IIA-Appunti.pdf#page=19]]
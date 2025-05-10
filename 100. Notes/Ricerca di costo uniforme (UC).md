**Data time:** 01:39 - 11-05-2025

**Tags:** [[Introduction to Artificial Intelligence]] [[Agenti Risolutori di problemi (Finging Algorithm)]] [[Graph Algorithms]]

**Area**: [[Bachelor's Degree]]
# Ricerca di costo uniforme (UC)

Generalizzazione della ricerca in ampiezza ([[Ricerca in ampiezza (BF)|BF]]). Si sceglie il nodo di costo minore sulla frontiera (si intende il costo $g(n)$ de cammino), si espande sui contorni di uguale (o meglio uniforme) costo invece che sui contorni di uguale profondità.

```python
def unform_cost_search(problem):
	# insieme (implementato come una lista) degli stati gia’ visitati
	explored =[] 
	# il costo del cammino e’ inizializzato nel costruttore del nodo
	node = Node(problem.initial_state) 
	# la frontiera e’ una coda coda con priorita’
	frontier =PriorityQueue(f =lambda x:x.path_cost) 
	#lambda serve a definire una funzione anonima a runtime
	frontier.insert(node)
	while not frontier.isempty():
		# seleziona il nodo e strae il nodo 
		# con costo minore, per l’espansione
		node = frontier.pop() 
	
		if problem.goal_test(node.state):
			return node.solution(explored_set =explored)
		else:
			# se non lo e’ inserisci lo stato nell’insieme degli esplorati
			explored.append(node.state)
			
			for action in problem.actions(node.state):
				child_node =node.child_node(problem, action)
			if (child_node.state not in explored) and 
				(not frontier.contains_state(child_node.state)):
				frontier.insert(child_node)
			elif frontier.contains_state(child_node.state) and
	(frontier.get_node(frontier.index_state(child_node.state)).path_cost >child_node.path_cost):
				frontier.remove(frontier.index_state(child_node.state))
				frontier.insert(child_node)
	return None # in questo caso ritorna con fallimento
```
Si utilizza una **Coda di priorità**
#### Analisi
Garantite purché il costo degli archi sia maggiore di $\epsilon > 0$
- **Strategia completa**: Si 
- **Strategie ottimale**: Si

Assunto che $C^*$ come il costo della soluzione ottima e $\lfloor C^* / \epsilon \rfloor$ numero di mosse nel caso peggiore arrotondare per difetto.
- **Complessità in tempo**: $O(b^{1 + \lfloor C^* / \epsilon \rfloor})$
- **Complessità in spazio**: $O(b^{1 + \lfloor C^* / \epsilon \rfloor})$

# References
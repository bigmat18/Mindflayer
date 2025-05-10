**Data time:** 01:34 - 11-05-2025

**Tags:** [[Introduction to Artificial Intelligence]] [[Agenti Risolutori di problemi (Finging Algorithm)]] [[Graph Algorithms]]

**Area**: [[Bachelor's Degree]]
# Ricerca in profondità (DF)

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

Utilizza una coda di tipo **LIFO**
#### Analisi
- **Strategia completa**: Si
- **Strategie ottimale**: Si
- **Complessità in tempo**: $O(b^2)$
- **Complessità in spazio**: $O(b \cdot m)$
## Ricerca in profondità limitata (DL)
SI va in profondità fino ad un certo livello predefinito l.
#### Analisi
- **Strategia completa**: Si per problemi in cui si conosce un limite superiore per la profondità della soluzione. Completa se D < l
- **Strategie ottimale**: No
- **Complessità in tempo**: $O(b^l)$
- **Complessità in spazio**: $O(b \cdot l)$
# References
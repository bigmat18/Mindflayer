**Data time:** 20:40 - 19-09-2024

**Status**: #master 

**Tags:** [[ISO-OSI Network layer]] - [[Graph Algorithms]]

**Area**: [[Bachelor's Degree]]
# Distance vector algorithm

- **Distribuito**: ogni nodo aggiorna i suoi vicini solo quando il proprio DV cambia, i vicini avvisano i rispettivi vicini dei cambiamenti
- **Iterativo**: si ripetono operazioni
- **Asincrono**: non richiede che tutti i nodi operino al passo con gli altri
###### Equazione Bellman-Ford
Formula per trovare il percorso a costo minimo tra un nodo sorgente ed uno destinatario
$$d_{xy} = min_v \{c(x,v) + d_{vy}\}$$
Si prende uno dei vicini v di x t.c. il costo da x a v + la distanza da v a y sia il più piccolo

![[Screenshot 2024-09-19 at 21.04.35.png | 400]]
###### Informazione mantenute da un nodo
- Conoscere il costo del collegamento verso ciascun vicino $c(x,v)$
- Vettore distanza $D_x = [D_{xy}\::\: \forall y \in N]$ 
- Vettore distanza dei vicini $D_x = [D_{xy}\::\: \forall y \in N] \:\:\forall$ vicino 

###### Algoritmo
1. Ciascuno nodo x inizia con una stima delle distanze verso tutti i nodi in N
2. Ogni nodo invia una copia del proprio vettore distanza a ciascuno dei suoi vicini
3. Quando un nodo x riceve un nuovo DV (vettore distanza) lo salva e usa l'equazione di Bellman-form per aggiornare il proprio vettore distanza.
4. Si ripete finché tutti i nodi continuano a cambiare i propri DV in maniera asincrona ciascuna stima dei costi $D_{xy}$ converge all'effettivo costo del percorso minimo.
5. Se cambia il costo del collegamento il nodo lo rileva e ricalcola il vettore distanza

##### Count-to-infinity problem

![[Screenshot 2024-09-19 at 21.13.24.png | 200]]

1. Nodo y si accorge che il collegamento verso x ha un nuovo costo di 60, ma z ha detto che la sua distanza da x è 50
2. Nodo y calcolo la nuova distanza verso x, 6, e invia notifica a z
3. Nodo z apprende che per arrivare a x tramite y c'è un costo di 6, così aggionra la sua distanza x
4. Il ciclo procederà per 44 iterazioni fino a quando z considera il costo del proprio percorso attraverso y maggiore di 50
5. Nodo z determina che il percorso a costo minimo verso x passa attraverso la connessione a x, e y instraderà verso x passando da z.
##### Split-horizon con Poisoned reverse
Se nodo X inoltra a V i pacchetti destinati a Z allora X invia a V $D_X[Z] = \infty$. Questo è detto **Count to infinity problem**

# References
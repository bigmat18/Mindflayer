---
Data: 2025-10-23T15:03:00
Tags:
  - note
  - youngling
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Graph Algorithms]]"
Area: "[[Bachelor's Degree]]"
---
# Dijkstra

L'algoritmo di Dijkstra, concepito dall'informatico olandese Edsger W. Dijkstra nel 1956, è un algoritmo greedy utilizzato per risolvere il problema dei **cammini minimi da una sorgente singola** (Single-Source Shortest Path, SSSP) in un grafo pesato e orientato.

**Dati di input:**
1. Un grafo `G = (V, E)`, dove `V` è l'insieme dei vertici (nodi) ed `E` è l'insieme degli archi (collegamenti).
2. Una funzione peso `w(u, v)` che associa un peso non-negativo a ogni arco `(u,v)`
3. Un vertice sorgente `s` da cui calcolare i percorsi.

**Obiettivo:**
Trovare la distanza minima (il percorso di peso totale minimo) dal vertice sorgente `s` a ogni altro vertice `v` nel grafo.

**Vincolo Fondamentale:** L'algoritmo di Dijkstra funziona correttamente solo se **tutti i pesi degli archi sono non-negativi** (`w(u, v) >= 0`). In presenza di pesi negativi, l'approccio greedy dell'algoritmo potrebbe fallire; in tal caso, si devono usare algoritmi come Bellman-Ford.


## 1. Logica dell'Algoritmo

L'idea centrale di Dijkstra è di costruire iterativamente un insieme `S` di vertici per i quali il cammino minimo dalla sorgente è stato trovato e finalizzato.

L'algoritmo mantiene per ogni vertice `v` una stima della distanza `d[v]` dalla sorgente `s`. Inizialmente, `d[s] = 0` e `d[v] = infinity` per tutti gli altri vertici.

Il processo si svolge come segue:
1.  **Inizializzazione**:
    * Crea un insieme `S` di vertici finalizzati, inizialmente vuoto.
    * Inizializza le distanze: `d[s] = 0` e `d[v] = ∞` per ogni `v ≠ s`.
    * Crea una coda di priorità `Q` contenente tutti i vertici di `V`, con priorità data dalla loro distanza `d`.

2.  **Ciclo Principale**: Finché la coda `Q` non è vuota:
    * **Selezione Greedy**: Estrai dalla coda `Q` il vertice `u` con la stima di distanza minima `d[u]`. Questo è il passo "greedy": si assume ottimisticamente che la via più breve per il nodo non ancora esplorato sia quella trovata finora.
    * **Finalizzazione**: Aggiungi `u` all'insieme `S`. La distanza `d[u]` è ora considerata definitiva.
    * **Rilassamento (Relaxation)**: Per ogni vertice `v` adiacente a `u` (cioè, per ogni arco `(u, v)`):
        * Se si trova un percorso più breve per `v` passando attraverso `u` (ovvero se `d[u] + w(u, v) < d[v]`), allora aggiorna la distanza di `v`:
            `d[v] = d[u] + w(u, v)`
        * Aggiorna la priorità di `v` nella coda `Q`.

L'algoritmo termina quando tutti i vertici raggiungibili dalla sorgente sono stati aggiunti a `S` (o quando la coda `Q` è vuota).


## 2. Implementazioni e Analisi della Complessità

La performance dell'algoritmo di Dijkstra dipende criticamente dalla struttura dati usata per implementare la coda di priorità `Q`, in particolare per le operazioni di `extract-min` (estrarre il vertice con distanza minima) e `decrease-key` (aggiornare la distanza di un vertice).

### Implementazione 1: Array Semplice

* **Struttura Dati**: Si usa un semplice array per memorizzare le distanze. La coda `Q` è gestita implicitamente.
* **Operazioni**:
    *  `extract-min`: Richiede una scansione lineare di tutti i vertici per trovare quello non ancora in `S` con la distanza minima. **Costo: $O(|V|)$**.
    *  `decrease-key`: Consiste semplicemente nell'aggiornare un valore nell'array. **Costo: $O(1)$**.
* **Complessità Totale**: Il ciclo principale viene eseguito $|V|$ volte. Ad ogni iterazione, l'operazione dominante è `extract-min`.
    * **$O(|V| \cdot |V| + |E|) = O(|V|^2)$**.
* **Quando usarla**: Questa implementazione è efficiente per **grafi densi**, dove il numero di archi $|E|$ è vicino a $|V|^2$. In questo scenario, il termine $|V|^2$ domina comunque, e la semplicità di implementazione è un vantaggio.

### Implementazione 2: Heap Binario (la più comune)

* **Struttura Dati**: Una coda a min-priorità implementata con un heap binario.
* **Operazioni**:
    * `extract-min`: L'estrazione della radice dell'heap. **Costo: $O(\log |V|)$**.
    * `decrease-key`: L'aggiornamento della priorità di un nodo. **Costo: $O(\log |V|)$**.
* **Complessità Totale**: L'algoritmo esegue $|V|$ operazioni di `extract-min` e al più $|E|$ operazioni di `decrease-key` (una per ogni arco, nel caso peggiore).
    * **$O(|V| \log |V| + |E| \log |V|) = O((|V| + |E|) \log |V|)$**. Per grafi connessi, $|E| \ge |V|-1$, quindi la complessità si semplifica in **$O(|E| \log |V|)$**.
* **Quando usarla**: È la scelta standard per **grafi sparsi** (dove $|E|$ è molto più piccolo di $|V|^2$), poiché la sua performance è significativamente migliore di quella quadratica.

### Implementazione 3: Heap di Fibonacci

* **Struttura Dati**: Una struttura dati più avanzata che ottimizza l'operazione di `decrease-key`.
* **Operazioni** (costo ammortizzato):
    * `extract-min`: **$O(\log |V|)$**.
    * `decrease-key`: **$O(1)$**.
* **Complessità Totale**: Con $|V|$ estrazioni e $|E|$ aggiornamenti, la complessità ammortizzata è:  
$$
O(|V| \log |V| + |E|)
$$
* **Quando usarla**: Offre la migliore performance asintotica, specialmente per grafi densi dove batte l'heap binario. Tuttavia, la sua implementazione è molto complessa e le costanti nascoste nella notazione O-grande la rendono spesso più lenta in pratica rispetto a un heap binario, a meno che i grafi non siano estremamente grandi.

### Tabella Riassuntiva

| Struttura Dati | Time 'extract-min'       | Time 'decrease-key' | Time Totale                   |
| -------------- | ------------------------ | ------------------- | ----------------------------- |
| Array Based    | $O( \|V\| )$             | $O(1)$              | $O(\|V\|^2)$                  |
| Heap Binario   | $O(\log\|V\|)$           | $O(\log\|V\|)$      | $O(\|E\| \log \|V\|)$         |
| Heap Fibonacci | $O(\log\|V\|)$ (ammort.) | $O(1)$ (ammort.)    | $O(\|E\| + \|V\| \log \|V\|)$ |


## 3. Codice di Esempio (C++ con Heap Binario)

Questa implementazione usa una lista di adiacenza per rappresentare il grafo e una `std::priority_queue` di C++ (che è un max-heap, adattato per funzionare come min-heap) per la coda di priorità.

```cpp
// Definiamo un alias per le coppie (peso, vertice) per chiarezza
using iPair = pair<int, int>;

void dijkstra(const vector<vector<iPair>>& adj, int start_node, int num_vertices) {
    // Coda a min-priorità per memorizzare i vertici da visitare.
    // La coppia è (distanza, vertice) per ordinare in base alla distanza.
    priority_queue<iPair, vector<iPair>, greater<iPair>> pq;

    // Vettore per memorizzare le distanze minime dalla sorgente
    vector<int> dist(num_vertices, numeric_limits<int>::max());

    // Inizializzazione
    pq.push({0, start_node});
    dist[start_node] = 0;

    while (!pq.empty()) {
        // Estrai il vertice con la distanza minima
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();

        // Ottimizzazione: un percorso più breve per 'u'
        // in una precedente iterazione, ignora questa voce.
        if (d > dist[u]) {
            continue;
        }

        // Itera su tutti i vicini di 'u'
        for (const auto& edge : adj[u]) {
            int v = edge.first;
            int weight = edge.second;

            // Rilassamento dell'arco (u, v)
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
}
```

# References

*   **Dijkstra, E. W. (1959).** *A note on two problems in connexion with graphs*. Numerische Mathematik, 1(1), 269–271. (Il paper originale)
*   **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).** *Introduction to Algorithms* (4th ed.). MIT Press. (Capitolo 22: "Elementary Graph Algorithms", Sezione 22.3: "Dijkstra's algorithm")
*   **GeeksforGeeks.** *Dijkstra's Shortest Path Algorithm*. Un'ottima risorsa online con diverse implementazioni e spiegazioni dettagliate.
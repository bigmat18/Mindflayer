---
Data: 2025-10-23T15:23:00
Tags:
  - note
  - youngling
Connection:
  - "[[Dynamic Programming]]"
Area: "[[Bachelor's Degree]]"
---
# Bellman-Ford per i Cammini Minimi

L'algoritmo di Bellman-Ford, sviluppato da Richard Bellman e Lester Ford Jr., risolve il problema dei **cammini minimi da una sorgente singola** (Single-Source Shortest Path, SSSP) in un grafo pesato e orientato, proprio come l'algoritmo di Dijkstra.

La sua caratteristica fondamentale, che lo distingue da Dijkstra, è la capacità di funzionare correttamente anche in presenza di **archi con peso negativo**. Tuttavia, questa flessibilità ha un costo in termini di performance, rendendolo più lento di Dijkstra.

**Dati di input:**
1. Un grafo `G = (V, E)`.
2. Una funzione peso `w(u, v)` che associa un peso (positivo, nullo o negativo) a ogni arco.
3. Un vertice sorgente `s`.

**Obiettivo:**
Trovare la distanza minima da `s` a ogni altro vertice `v`. Se il grafo contiene un ciclo a peso negativo raggiungibile dalla sorgente, l'algoritmo è in grado di rilevarlo e segnalarlo.

## 1. Logica dell'Algoritmo

L'algoritmo si basa su un principio semplice e potente: il **rilassamento iterativo**. L'idea è che il cammino minimo da una sorgente `s` a un qualsiasi altro vertice `v` non può contenere più di `|V| - 1` archi (supponendo che non ci siano cicli a peso negativo).

Basandosi su questa osservazione, Bellman-Ford "rilassa" ripetutamente tutti gli archi del grafo. Dopo la `i`-esima iterazione, l'algoritmo garantisce di aver trovato la distanza minima per tutti i vertici raggiungibili con un percorso di al più `i` archi. Di conseguenza, dopo `|V| - 1` iterazioni complete, avrà trovato tutti i cammini minimi.

Il processo si svolge come segue:
1. **Inizializzazione**:
    * Per ogni vertice `v` nel grafo, inizializza la stima della distanza: `d[s] = 0` per la sorgente e `d[v] = ∞` per tutti gli altri vertici.
    * Inizializza un array di predecessori `p[v]` a `null`.

2. **Ciclo Principale (Rilassamento)**:
    * Ripeti il seguente processo per `|V| - 1` volte:
        * Per **ogni arco `(u, v)`** presente nel grafo `E`:
            * Esegui l'operazione di **rilassamento**: se si trova un percorso più breve per `v` passando da `u` (cioè, se `d[u] + w(u, v) < d[v]`), aggiorna la distanza e il predecessore di `v`:
                `d[v] = d[u] + w(u, v)`
                `p[v] = u`

Alla fine di questo ciclo, se non ci sono cicli negativi, l'array `d` conterrà le distanze minime dalla sorgente `s`.


## 2. Rilevamento di Cicli a Peso Negativo

Una delle caratteristiche più potenti di Bellman-Ford è la sua capacità di rilevare la presenza di cicli a peso negativo raggiungibili dalla sorgente. Un ciclo di questo tipo renderebbe il concetto di "cammino minimo" indefinito, poiché si potrebbe percorrere il ciclo all'infinito per diminuire il peso del percorso a `-∞`.

Il rilevamento avviene con un semplice passo aggiuntivo:
* **Ciclo di Controllo**: Dopo aver completato le `|V| - 1` iterazioni, si esegue un'**ulteriore iterazione** (la `|V|`-esima) su tutti gli archi `(u, v)`.
    *   Se in questa iterazione è ancora possibile "rilassare" un qualsiasi arco (cioè, se si trova un valore `d[v]` che può essere ulteriormente diminuito), allora significa che esiste un ciclo a peso negativo nel grafo.

Il motivo è che un cammino minimo semplice non può avere più di `|V|-1` archi. Se la distanza si riduce ancora alla `|V|`-esima iterazione, significa che il percorso "ottimo" trovato ha `|V|` archi, il che implica necessariamente la presenza di un ciclo.

---

## 3. Analisi della Complessità

La struttura dell'algoritmo è molto diretta, il che rende l'analisi semplice.

* **Complessità Temporale: $O(|V| \cdot |E|)$**
    * L'algoritmo consiste in un ciclo principale che viene eseguito `|V| - 1` volte.
    * All'interno di questo ciclo, si itera su tutti gli `|E|` archi del grafo.
    * Questo porta a una complessità totale proporzionale a `(|V| - 1) * |E|`, che si semplifica in $O(|V| \cdot |E|)$. Il ciclo di controllo aggiuntivo ha un costo di $O(|E|)$ e non cambia la complessità asintotica.

* **Complessità Spaziale: $O(|V|)$**
    * L'algoritmo richiede spazio per memorizzare l'array delle distanze `d` e l'array dei predecessori `p`, entrambi di dimensione `|V|`.

### Confronto con [[Dijkstra]]

| Caratteristiche   | Bellman-Ford                                                     | Dijkstra                                                    |
| ----------------- | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| **Pesi Negativi** | Si (permessi)                                                    | No (non permessi)                                           |
| **Complessità**   | $O(\|V\| \cdot \|E\|)$                                           | $O(\|E\| \log \|V\|)$                                       |
| **Uso Ideale**    | Grafi con possibili pesi negativi, rilevamento di cicli negativi | Grafi con pesi non-negativi, dove la velocità è prioritaria |


## 4. Codice di Esempio (C++)

Per Bellman-Ford, è comodo rappresentare il grafo come una semplice lista di archi, poiché l'algoritmo itera su tutti gli archi a ogni passo.

```cpp
// Struttura per rappresentare un arco pesato nel grafo
struct Edge {
    int source;
    int destination;
    int weight;
};

void bellman_ford(const vector<Edge>& edges, 
				  int num_vertices, int start_node) 
{
    // Vettore per memorizzare le distanze minime dalla sorgente
    vector<int> dist(num_vertices, numeric_limits<int>::max());
    dist[start_node] = 0;

    // 1. Rilassa tutti gli archi |V| - 1 volte
    for (int i = 0; i < num_vertices - 1; ++i) {
        for (const auto& edge : edges) {
            if (dist[edge.source] != numeric_limits<int>::max() &&
                dist[edge.source] + edge.weight < dist[edge.destination]) {
                dist[edge.destination] = dist[edge.source] +
                                         edge.weight;
            }
        }
    }

    // 2. Controlla la presenza di cicli a peso negativo
    for (const auto& edge : edges) {
        if (dist[edge.source] != numeric_limits<int>::max() &&
            dist[edge.source] + edge.weight < dist[edge.destination]) {
            return;
        }
    }
}
```

# References

*   **Bellman, R. (1958).** *On a routing problem*. Quarterly of Applied Mathematics, 16, 87-90.
*   **Ford Jr., L. R. (1956).** *Network Flow Theory*. Rand Corporation.
*   **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).** *Introduction to Algorithms* (4th ed.). MIT Press. (Capitolo 22: "Elementary Graph Algorithms", Sezione 22.4: "The Bellman-Ford algorithm")

---
Data: 2025-10-23T15:00:00
Tags:
  - note
  - youngling
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Dynamic Programming]]"
Area: "[[Bachelor's Degree]]"
---
# Zaino 0-1 (0-1 Knapsack Problem)

Il problema dello Zaino 0-1 è un classico problema di ottimizzazione combinatoria. L'obiettivo è selezionare un sottoinsieme di oggetti, ognuno con un proprio peso e valore, da inserire in uno zaino con capacità di peso limitata, in modo da massimizzare il valore totale degli oggetti scelti.

Il nome "0-1" deriva dal fatto che per ogni oggetto abbiamo una decisione binaria: o lo si prende interamente (1) o non lo si prende affatto (0). Non è possibile prendere frazioni di un oggetto.

**Dati di input:**
1.  Un insieme di `n` oggetti.
2.  Per ogni oggetto `i`: un valore `v_i` e un peso `w_i`.
3.  Una capacità massima `W` per lo zaino.

**Obiettivo:** Trovare un sottoinsieme di oggetti la cui somma dei pesi non superi `W` e la cui somma dei valori sia la massima possibile.

---

## Formulazione del Problema

### Approccio Brute-Force
La soluzione più semplice consiste nell'esplorare tutte le possibili combinazioni di oggetti. Per `n` oggetti, esistono $2^n$ possibili sottinsiemi. Per ciascuno di essi, si calcola il peso e il valore totale, scartando quelli che superano la capacità `W` e tenendo traccia del massimo valore trovato. Questo approccio ha una complessità temporale di $O(n \cdot 2^n)$ ed è impraticabile per un numero di oggetti anche modesto.

### Approccio con Programmazione Dinamica

Un approccio molto più efficiente si basa sulla programmazione dinamica, sfruttando la sottostruttura ottimale del problema.

#### La Relazione di Ricorrenza
Definiamo `B(k, w)` come il valore massimo ottenibile utilizzando un sottoinsieme dei primi `k` oggetti (`0, 1, ..., k-1`) con un peso massimo consentito di `w`.

Quando consideriamo il `k`-esimo oggetto (con valore `v_k` e peso `w_k`), abbiamo due possibilità:

1.  **Non prendere l'oggetto `k`**: Questa scelta è sempre possibile. Il valore massimo sarà lo stesso che si poteva ottenere con `k-1` oggetti e la stessa capacità `w`. Cioè, `B(k-1, w)`.
2.  **Prendere l'oggetto `k`**: Questa scelta è possibile solo se il suo peso `w_k` non supera la capacità residua `w` (`w_k <= w`). In questo caso, il valore ottenuto sarà `v_k` più il valore massimo che si poteva ottenere con i restanti `k-1` oggetti e una capacità ridotta di `w_k`. Cioè, `v_k + B(k-1, w - w_k)`.

La soluzione ottima `B(k, w)` sarà quindi il massimo tra queste due opzioni. Questo ci porta alla seguente formula:

$$
B(k, w) =
\begin{cases}
  B(k-1, w) & \text{se } w_k > w \\
  \max(B(k-1, w), \quad v_k + B(k-1, w - w_k)) & \text{se } w_k \le w
\end{cases}
$$

I casi base sono `B(0, w) = 0` (nessun valore se non ci sono oggetti) e `B(k, 0) = 0` (nessun valore se la capacità dello zaino è zero).

#### Soluzione Bottom-Up con Tabella
Per evitare i calcoli ridondanti di un approccio ricorsivo, costruiamo una tabella (matrice) `M` di dimensioni `(n+1) x (W+1)`. La cella `M[i][w]` memorizzerà il valore massimo ottenibile usando i primi `i` oggetti con una capacità di zaino pari a `w`.

La tabella viene riempita in modo **bottom-up**:
1.  **Inizializzazione**: La prima riga (`i=0`) e la prima colonna (`w=0`) vengono riempite con zeri, secondo i casi base.
2.  **Riempimento**: Si procede riga per riga (per ogni oggetto `i` da 1 a `n`) e, per ogni riga, colonna per colonna (per ogni capacità `w` da 1 a `W`), applicando la formula ricorsiva basandosi sui valori già calcolati nella tabella.

Alla fine, la cella `M[n][W]` conterrà la soluzione al problema originale.

### Codice di Esempio

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int zaino_01(const int value[], const int weight[], int maxW, int n)
{
    // Matrice M[i][w] per memorizzare il valore massimo per i primi 'i' oggetti
    // con una capacità di zaino 'w'.
    std::vector<std::vector<int>> M(n + 1, std::vector<int>(maxW + 1, 0));

    // Scorre tutti gli oggetti (righe)
    for (int i = 1; i <= n; i++)
    {
        // Scorre tutte le possibili capacità dello zaino (colonne)
        for (int w = 1; w <= maxW; w++)
        {
            // Valore e peso dell'oggetto corrente (l'indice è i-1 perché l'array è 0-based)
            int current_value = value[i - 1];
            int current_weight = weight[i - 1];

            // Se l'oggetto corrente non entra nello zaino di capacità 'w'
            if (current_weight > w)
            {
                // La soluzione è la stessa di quella senza questo oggetto
                M[i][w] = M[i - 1][w];
            }
            else
            {
                // Altrimenti, scegliamo l'opzione migliore:
                // 1. Non prendere l'oggetto: M[i - 1][w]
                // 2. Prendere l'oggetto: current_value + M[i - 1][w - current_weight]
                M[i][w] = std::max(M[i - 1][w], current_value + M[i - 1][w - current_weight]);
            }
        }
    }

    // Il risultato finale si trova nell'angolo in basso a destra
    return M[n][maxW];
}
```

### Analisi della Complessità: Algoritmo Pseudo-Polinomiale
*   **Complessità Temporale e Spaziale: $O(n \cdot W)$**.
    L'algoritmo riempie una matrice `n x W`, eseguendo un'operazione a tempo costante per ogni cella.

A prima vista, $O(nW)$ sembra polinomiale. Tuttavia, è classificato come **pseudo-polinomiale**. Perché? Un algoritmo è veramente polinomiale se la sua complessità è un polinomio nella *dimensione dell'input in bit*. La dimensione del numero `W` non è `W`, ma il numero di bit necessari per rappresentarlo, cioè circa $log_2 W$. Poiché la complessità dipende dal *valore numerico* di `W` e non dalla sua dimensione in bit, la complessità effettiva in termini di dimensione dell'input è $O(n \cdot 2^{\log W})$, che è esponenziale rispetto alla lunghezza di `W`.

### Ricostruire la Soluzione
Per sapere quali oggetti sono stati scelti, si può fare un **backtracking** sulla matrice `M` partendo da `M[n][W]`:
1.  Inizia da `i = n`, `w = W`.
2.  Se `M[i][w] != M[i-1][w]`, significa che l'oggetto `i` è stato incluso nella soluzione. Aggiungi l'oggetto `i` alla lista e aggiorna la capacità `w = w - weight[i-1]`.
3.  Spostati alla riga precedente: `i = i - 1`.
4.  Ripeti finché `i > 0`.
# References
---
Data: 2025-10-23T14:20:00
Tags:
  - note
  - master
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Dynamic Programming]]"
Area: "[[Bachelor's Degree]]"
---
# Binomial Coefficient

Il coefficiente binomiale, indicato come $\binom{n}{k}$ (si legge "n su k"), rappresenta il numero di modi in cui è possibile scegliere `k` elementi da un insieme di `n` elementi, senza tener conto dell'ordine.

La sua formula matematica è:
$$ \binom{n}{k} = \frac{n!}{k!(n-k)!} $$
Tuttavia, il calcolo diretto tramite i fattoriali è computazionalmente costoso e può facilmente portare a overflow con numeri relativamente piccoli. Per questo motivo, si preferisce utilizzare la sua definizione ricorsiva, nota come **Identità di Pascal**:

$$
\binom{n}{k} =
\begin{cases}
  1 & \text{se } k = 0 \text{ o } k = n \\
  \binom{n-1}{k-1} + \binom{n-1}{k} & \text{altrimenti}
\end{cases}
$$

Questa relazione è alla base del famoso **Triangolo di Tartaglia (o di Pascal)**, dove ogni elemento è la somma dei due elementi sopra di esso.

---

## 1. Approccio Ricorsivo (Top-Down)

Questo metodo implementa direttamente la definizione ricorsiva. È concettualmente semplice ma, come per Fibonacci, soffre di una grave inefficienza dovuta al ricalcolo ripetuto degli stessi valori (sottoproblemi sovrapposti).

### Codice di Esempio

```cpp
int rec_coef_bin(int n, int k)
{
    // Caso base: i bordi del Triangolo di Pascal sono sempre 1.
    if ((k == 0) || (k == n))
    {
        return 1;
    }
    // Passo ricorsivo: ogni elemento è la somma dei due sopra di esso.
    else
    {
        return rec_coef_bin(n - 1, k - 1) + rec_coef_bin(n - 1, k);
    }
}
```

### Analisi della Complessità
*   **Complessità Temporale: $O(2^n)$**. L'albero delle chiamate ricorsive cresce esponenzialmente, portando a un numero di operazioni molto elevato.
*   **Complessità Spaziale: $O(n)$**. Lo spazio è determinato dalla massima profondità dello stack di ricorsione.

---

## 2. Approccio con Programmazione Dinamica (Bottom-Up)

Per superare l'inefficienza della ricorsione, utilizziamo la programmazione dinamica per calcolare i valori una sola volta e memorizzarli. L'idea è di costruire virtualmente il Triangolo di Pascal riga per riga, fino ad arrivare alla riga `n`.

### Ottimizzazione dello Spazio

Una soluzione standard utilizzerebbe una matrice `(n+1) x (k+1)` per memorizzare tutti i coefficienti binomiali, con un costo spaziale di $O(n \cdot k)$.

```
    _ _ _ _ _ k colonne
  | 1 0 0 0 0
  | 1 1
  | 1   1
  | 1 Y Z 1
  |     X   1
n righe
```

Tuttavia, possiamo notare che per calcolare una riga `i` del triangolo, abbiamo bisogno soltanto dei valori della riga precedente, `i-1`. Questo ci permette di ottimizzare drasticamente lo spazio, utilizzando solo due array (o addirittura uno solo con un po' più di attenzione) per memorizzare la riga precedente e quella corrente.

L'algoritmo procede come segue:
1.  Inizializza un array `prev_row` che rappresenta una riga del triangolo.
2.  Calcola la `current_row` usando i valori di `prev_row`.
3.  Una volta calcolata, `current_row` diventa la `prev_row` per l'iterazione successiva.
4.  Ripeti fino a raggiungere la riga `n`.

### Codice di Esempio (con Spazio Ottimizzato)

```cpp
#include <vector>

int it_coef_bin(int n, int k)
{
    // Se k è fuori dal range [0, n], il risultato è 0 (non valido)
    if (k < 0 || k > n) {
        return 0;
    }

    // Usiamo dei vector per una gestione più sicura della memoria.
    // prev_row conterrà i coefficienti della riga i-1.
    // curr_row conterrà i coefficienti della riga i in costruzione.
    std::vector<int> prev_row(n + 1, 0);
    std::vector<int> curr_row(n + 1, 0);

    // Inizializzazione della riga 0 del triangolo: C(0, 0) = 1
    prev_row[0] = 1;

    // Costruiamo il triangolo riga per riga, da i=1 fino a n
    for (int i = 1; i <= n; i++)
    {
        // Il primo elemento di ogni riga è sempre 1 (C(i, 0))
        curr_row[0] = 1;

        // Calcoliamo gli elementi intermedi della riga i
        // j va da 1 fino a i
        for (int j = 1; j <= i; j++)
        {
            curr_row[j] = prev_row[j - 1] + prev_row[j];
        }

        // Dopo aver calcolato la riga corrente, questa diventa
        // la riga precedente per la prossima iterazione.
        prev_row = curr_row;
    }

    // Il risultato è il k-esimo elemento dell'ultima riga calcolata.
    return prev_row[k];
}
```

### Analisi della Complessità
*   **Complessità Temporale: $O(n \cdot k)$**. L'algoritmo consiste di due cicli annidati. Il ciclo esterno itera `n` volte, e quello interno itera al più `k` (o `n`) volte. Il numero totale di somme eseguite è proporzionale all'area del triangolo fino alla riga `n`, quindi $O(n^2)$ nel caso peggiore in cui $k \approx n/2$ o più precisamente $O(n \cdot k)$.
*   **Complessità Spaziale: $O(n)$**. Grazie all'ottimizzazione, manteniamo in memoria solo due array di dimensione `n+1`, riducendo lo spazio da quadratico a lineare.
# References
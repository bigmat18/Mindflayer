---
Data: 2025-10-23T14:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Dynamic Programming]]"
Area: "[[Bachelor's Degree]]"
---
# Rod Cutting Problem

Questo problema classico della programmazione dinamica consiste nel trovare il modo migliore per tagliare una corda (o un'asta) di lunghezza `n` per massimizzare il ricavo. Vengono forniti i prezzi `p_i` per ogni pezzo di lunghezza `i`.

È importante notare che un pezzo di lunghezza `n` può essere venduto intero oppure tagliato in vari pezzi più piccoli.

**Esempio:**
Data una corda di lunghezza `n=4` e la seguente tabella di prezzi:
| Lunghezza (i) | 1  | 2  | 3   | 4    |
|---------------|---|---|---  |----|
| Prezzo (p_i)    | 1  | 5  | 8   | 9    |

Una possibile soluzione è non tagliare la corda, ottenendo un ricavo di 9. Un'altra è tagliarla in due pezzi da 2, con ricavo $5 + 5 = 10$. La soluzione ottimale, però, è tagliare la corda in due pezzi di lunghezza 1 e 3 (o 3 e 1), ottenendo un ricavo di $p_1 + p_3 = 1 + 8 = 9$ (errore nell'esempio originale, la migliore combinazione è 2+2=10). Se i prezzi fossero `(1, 3), (2, 4), (3, 10), (4, 12)`, la soluzione ottimale sarebbe 1+3, con ricavo $3+10=13$.

---

## 1. Approccio Ricorsivo Bruto (Top-Down)

La prima soluzione che viene in mente è esplorare tutte le possibilità in modo ricorsivo. L'idea è di fare un primo taglio di lunghezza `i` (con `i` che va da 1 a `n`) e poi trovare la soluzione ottimale per il pezzo di corda rimanente di lunghezza `n-i`.

La formula di ricorrenza che descrive questo approccio è:
$$ r_n = \max_{1 \le i \le n} (p_i + r_{n-i}) $$
dove $r_n$ è il ricavo massimo per una corda di lunghezza `n` e la base della ricorsione è $r_0 = 0$.

### Codice di Esempio

```cpp
#include <algorithm> // Per std::max

/**
 * @brief Calcola il massimo ricavo per il taglio di una corda con un approccio top-down.
 *
 * @param p Array dei prezzi, dove p[i-1] è il prezzo per un pezzo di lunghezza i.
 * @param n Lunghezza della corda.
 * @return Il massimo ricavo ottenibile.
 */
int taglio_top_down(int p[], int n)
{
    // Caso Base: una corda di lunghezza 0 non ha valore.
    if (n == 0)
        return 0;

    int max_revenue = -1; // Inizializzato a un valore molto piccolo.

    // Prova a fare il primo taglio di tutte le lunghezze possibili, da 1 a n.
    for (int i = 1; i <= n; i++)
    {
        // La soluzione è il massimo tra le opzioni:
        // prezzo del primo pezzo (p[i-1]) + ricavo ottimo per il resto (n-i).
        max_revenue = std::max(max_revenue, p[i - 1] + taglio_top_down(p, n - i));
    }

    return max_revenue;
}
```

### Inefficienza e Sottoproblemi Sovrapposti
Questo approccio è estremamente inefficiente perché ricalcola molte volte la soluzione ottima per la stessa lunghezza. Ad esempio, per calcolare $r_4$, l'algoritmo calcolerà $r_2$ sia quando il primo taglio è 2, sia quando il primo taglio è 1 e il taglio successivo è 1.

L'albero delle chiamate ricorsive mostra chiaramente le ripetizioni:
```text
                              r_4
                    /      /      \      \
                  1+r_3   2+r_2   3+r_1   4+r_0
                    /|\      / \      |
                1+r_2 ...  1+r_1 ... 1+r_0
                  / \
               1+r_1 ...
```

Come si vede, `r_2` e `r_1` vengono invocati più volte.

### Analisi della Complessità
*   **Complessità Temporale: $O(2^n)$**. Il numero di chiamate cresce esponenzialmente con `n`, rendendo l'algoritmo impraticabile per lunghezze non banali.
*   **Complessità Spaziale: $O(n)$** per lo stack di ricorsione.

---

## 2. Approccio con Programmazione Dinamica (Bottom-Up)

Per risolvere il problema dei sottoproblemi sovrapposti, usiamo la programmazione dinamica. L'approccio **Bottom-Up** consiste nel risolvere i problemi più piccoli per primi e usare le loro soluzioni per costruire la soluzione per problemi via via più grandi.

Calcoliamo il ricavo ottimo per una corda di lunghezza 1, poi 2, poi 3, e così via fino a `n`, memorizzando ogni risultato in un array.

### Codice di Esempio

```cpp
#include <vector>
#include <algorithm>

/**
 * @brief Calcola il massimo ricavo per il taglio di una corda con un approccio bottom-up.
 *
 * @param p Array dei prezzi, dove p[i-1] è il prezzo per un pezzo di lunghezza i.
 * @param n Lunghezza della corda.
 * @return Il massimo ricavo ottenibile.
 */
int taglio_bottom_up(int p[], int n)
{
    // r[j] conterrà il ricavo massimo per una corda di lunghezza j.
    std::vector<int> r(n + 1);
    // s[j] memorizza la dimensione del primo pezzo nel taglio ottimale per una corda di lunghezza j.
    // Questo è utile per ricostruire la soluzione.
    std::vector<int> s(n + 1);

    r[0] = 0; // Caso base

    // Calcola il ricavo ottimo per ogni lunghezza j da 1 a n.
    for (int j = 1; j <= n; j++)
    {
        int max_revenue = -1;
        // Per trovare r[j], prova tutti i possibili primi tagli 'i' (da 1 a j).
        for (int i = 1; i <= j; i++)
        {
            // Il ricavo per questa opzione è p[i-1] + r[j-i]
            // Nota: r[j-i] è già stato calcolato e memorizzato!
            if (max_revenue < p[i - 1] + r[j - i])
            {
                max_revenue = p[i - 1] + r[j - i];
                // Memorizza la dimensione di questo primo taglio ottimo.
                s[j] = i;
            }
        }
        r[j] = max_revenue;
    }

    // Per stampare i tagli:
    // int temp_n = n;
    // while (temp_n > 0) {
    //     std::cout << "Taglio: " << s[temp_n] << std::endl;
    //     temp_n = temp_n - s[temp_n];
    // }

    return r[n];
}
```

### Analisi della Complessità
*   **Complessità Temporale: $O(n^2)$**. L'algoritmo è dominato da due cicli `for` annidati. Il numero totale di operazioni è la somma dei primi `n` interi, che è $\frac{n(n+1)}{2}$, appartenente alla classe $O(n^2)$.
*   **Complessità Spaziale: $O(n)$**. Utilizziamo due array di dimensione `n+1` per memorizzare i risultati intermedi (`r` e `s`).
# References
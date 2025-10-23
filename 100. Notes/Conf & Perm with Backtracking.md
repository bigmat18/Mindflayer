---
Data: 2025-10-23T14:15:00
Tags:
  - note
  - youngling
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Dynamic Programming]]"
Area: "[[Bachelor's Degree]]"
---
# Conf e Perm with Backtracking

In questo documento esploriamo due classici problemi combinatori: la generazione di tutte le possibili **configurazioni** (o sottinsiemi) di un insieme di elementi e la generazione di tutte le **permutazioni** di una sequenza. Entrambi i problemi vengono risolti elegantemente utilizzando un approccio ricorsivo basato sulla tecnica del **backtracking**.

---

## 1. Generazione di Tutte le Configurazioni (Sottinsiemi)

L'obiettivo è generare tutte le possibili combinazioni in cui ogni elemento di un insieme può essere presente o assente. Un modo efficace per rappresentare questo concetto è usare un array di supporto `A` della stessa dimensione dell'insieme, dove:
*   `A[i] = 1` significa che l'i-esimo elemento è **incluso** nella configurazione.
*   `A[i] = 0` significa che l'i-esimo elemento è **escluso**.

Generare tutte le configurazioni equivale a generare tutte le possibili stringhe binarie di lunghezza `n`, che sono $2^n$.

### Approccio Ricorsivo (Backtracking)

L'algoritmo esplora ricorsivamente tutte le scelte possibili. Per ogni posizione `k` dell'array, si provano entrambe le opzioni (0 e 1), e per ciascuna scelta si procede alla posizione successiva `k+1`.

1.  **Scelta**: Alla posizione `k`, assegna `0`.
2.  **Esplora**: Richiama la funzione per la posizione `k+1`.
3.  **Scelta**: Alla posizione `k`, assegna `1`.
4.  **Esplora**: Richiama la funzione per la posizione `k+1`.

Quando si raggiunge la fine dell'array (`k == n-1`), una configurazione completa è stata costruita e può essere processata (ad esempio, stampata).

### Codice di Esempio

```cpp
#include <iostream>

// Procedura ausiliaria per stampare una configurazione/permutazione
void controllo(int A[], int dim)
{
    for (int i = 0; i < dim; i++)
    {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Genera tutte le configurazioni binarie di un array di n elementi.
 *
 * @param A Array di supporto per memorizzare la configurazione corrente.
 * @param k Indice dell'elemento corrente da decidere (0 o 1).
 * @param n Dimensione totale dell'array.
 */
void configurazioni(int A[], int k, int n)
{
    // Ciclo per esplorare le due scelte possibili per l'elemento k: 0 (escluso) e 1 (incluso).
    for (int i = 0; i <= 1; i++)
    {
        A[k] = i;

        // Caso Base: se abbiamo deciso per tutti gli elementi (da 0 a n-1),
        // abbiamo una configurazione completa e la processiamo.
        if (k == n - 1)
        {
            controllo(A, n);
        }
        // Passo Ricorsivo: se non siamo alla fine, facciamo la chiamata
        // ricorsiva per decidere l'elemento successivo k+1.
        else
        {
            configurazioni(A, k + 1, n);
        }
    }
}
```

### Complessità
L'albero di ricorsione ha $2^n$ foglie (le configurazioni finali). Per arrivare a ogni foglia si compiono $n$ passi.
*   **Complessità Temporale**: $O(n \cdot 2^n)$, poiché esistono $2^n$ configurazioni e la stampa di ciascuna richiede $O(n)$ tempo.

---

## 2. Generazione di Tutte le Permutazioni

Una permutazione è una disposizione ordinata degli elementi di un insieme. Per un insieme di `n` elementi distinti, esistono $n!$ (n-fattoriale) permutazioni.

### Approccio Ricorsivo (Backtracking)

Anche in questo caso, la strategia è quella di costruire la soluzione un pezzo alla volta. L'idea è di "fissare" un elemento in ogni posizione `k` e generare ricorsivamente tutte le permutazioni per la parte restante dell'array (da `k+1` a `n-1`).

1.  **Itera**: Per la posizione `k`, scorri tutti gli elementi da `k` fino alla fine dell'array.
2.  **Scelta**: Scambia l'elemento in posizione `k` con l'elemento corrente `i` del ciclo. In questo modo, ogni elemento ha la possibilità di occupare la posizione `k`.
3.  **Esplora**: Richiama la funzione ricorsivamente per la posizione successiva, `k+1`.
4.  **Backtrack**: **Annulla la scelta**. Scambia di nuovo gli elementi per ripristinare l'array allo stato precedente. Questo passo è fondamentale per garantire che il ciclo successivo parta dalla configurazione corretta.

Quando si arriva alla fine dell'array (`k == n-1`), una permutazione completa è stata generata.

### Codice di Esempio

```cpp
/**
 * @brief Genera tutte le permutazioni di un array P.
 *
 * @param P Array di cui generare le permutazioni.
 * @param k Indice della posizione da cui iniziare a permutare.
 * @param n Dimensione totale dell'array.
 */
void permutazioni(int P[], int k, int n)
{
    // Caso Base: se k arriva all'ultimo elemento, significa che abbiamo
    // fissato le posizioni da 0 a n-1. La permutazione è completa.
    if (k == n - 1)
    {
        controllo(P, n);
    }
    else
    {
        // Per ogni elemento 'i' dalla posizione corrente 'k' fino alla fine...
        for (int i = k; i < n; i++)
        {
            // 1. Scelta: Scambia l'elemento in posizione 'k' con quello in posizione 'i'.
            // Questo fissa un nuovo elemento in posizione 'k'.
            std::swap(P[k], P[i]);

            // 2. Esplora: Genera tutte le permutazioni per la parte restante dell'array (da k+1 in poi).
            permutazioni(P, k + 1, n);

            // 3. Backtrack: Annulla lo scambio per ripristinare l'array e
            // permettere al ciclo di funzionare correttamente alla prossima iterazione.
            std::swap(P[k], P[i]);
        }
    }
}
```

### Complessità
*   **Complessità Temporale**: $O(n \cdot n!)$, poiché esistono $n!$ permutazioni e la stampa di ciascuna richiede $O(n)$ tempo.
# References
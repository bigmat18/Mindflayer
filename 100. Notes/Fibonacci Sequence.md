---
Data: 2025-10-23T14:04:00
Tags:
  - note
  - youngling
Connection:
  - "[[Programming & Algorithms]]"
Area: "[[Bachelor's Degree]]"
---
# Fibonacci Sequence 

La sequenza di Fibonacci è una delle successioni di numeri interi più famose in matematica. La sua definizione formale è la seguente:

$$
F_n =
\begin{cases}
  0 & \text{se } n = 0 \\
  1 & \text{se } n = 1 \\
  F_{n-1} + F_{n-2} & \text{se } n \ge 2
\end{cases}
$$

In questo documento analizziamo due metodi per calcolarla: l'approccio ricorsivo classico (Top-Down) e un approccio iterativo basato sulla programmazione dinamica (Bottom-Up).

---

## 1. Approccio Ricorsivo Classico (Top-Down)

Questo approccio traduce direttamente la definizione matematica in una funzione ricorsiva. Si parte dal numero `n` desiderato e si "scende" ricorsivamente fino a raggiungere i casi base (0 e 1).

### Codice di Esempio

```cpp
int classic_fib(int n)
{
    // Casi base della ricorsione
    if (n <= 1)
    {
        return n;
    }
    // Passo ricorsivo
    else
    {
        return classic_fib(n - 1) + classic_fib(n - 2);
    }
}
```

### Funzionamento e Inefficienze

Questo metodo è definito **Top-Down** perché per risolvere il problema principale (`F_n`), lo scompone in sottoproblemi sempre più piccoli (`F_{n-1}`, `F_{n-2}`, etc.), creando un albero di chiamate che viene percorso dall'alto verso il basso.

Il problema principale di questo approccio è la **sovrapposizione dei sottoproblemi**. Molti valori vengono ricalcolati più e più volte, portando a un'inefficienza esponenziale.

Visualizziamo l'albero delle chiamate per `F_5`:

```text
                  F_5
                /     \
            F_4         F_3
           /   \       /   \
         F_3   F_2     F_2   F_1
        /   \ /   \   /   \
      F_2 F_1 F_1 F_0 F_1 F_0
     /   \
   F_1 F_0
```

Come si può notare, `F_3` viene calcolato 2 volte, `F_2` 3 volte, e così via. All'aumentare di `n`, il numero di calcoli ridondanti cresce in modo esponenziale.

### Analisi della Complessità

*   **Complessità Temporale: $O(2^n)$**
    L'albero delle chiamate ha una profondità `n` e si ramifica quasi completamente, portando a un numero di operazioni che cresce esponenzialmente con `n`.
*   **Complessità Spaziale: $O(n)$**
    Lo spazio è determinato dalla massima profondità dello stack delle chiamate ricorsive, che è pari a `n`.

---

## 2. Approccio con Programmazione Dinamica (Bottom-Up)

La programmazione dinamica risolve il problema dell'inefficienza memorizzando i risultati dei sottoproblemi per evitare di ricalcolarli. L'approccio **Bottom-Up** (dal basso verso l'alto) consiste nel risolvere prima i problemi più piccoli e usare le loro soluzioni per costruire, passo dopo passo, la soluzione al problema più grande.

Si parte dai valori noti `F_0 = 0` e `F_1 = 1` e si calcolano in modo iterativo tutti i valori successivi fino a `F_n`.

### Codice di Esempio (Versione Ottimizzata)

La versione standard di questo approccio utilizzerebbe un array di `n+1` elementi per memorizzare tutti i valori di Fibonacci calcolati. Tuttavia, possiamo ottimizzare lo spazio notando che per calcolare `F_k` sono necessari solo i due valori precedenti, `F_{k-1}` e `F_{k-2}`.

```cpp
int dynamic_fib(int n)
{
    if (n <= 1) {
        return n;
    }

    // Manteniamo solo gli ultimi due valori calcolati
    // Inizializzati come F_0 e F_1
    int val1 = 0; // Corrisponde a F_{k-2}
    int val2 = 1; // Corrisponde a F_{k-1}
    int current_fib; // Corrisponderà a F_k

    // Partiamo da k=2 perché F_0 e F_1 sono già noti
    for (int k = 2; k <= n; k++)
    {
        current_fib = val1 + val2; // Calcola F_k
        val1 = val2;               // F_{k-2} diventa il vecchio F_{k-1}
        val2 = current_fib;        // F_{k-1} diventa il nuovo F_k
    }
    return val2; // Alla fine del ciclo, val2 contiene F_n
}
```

### Analisi della Complessità

*   **Complessità Temporale: $O(n)$**
    L'algoritmo esegue un singolo ciclo `for` che va da 2 a `n`. Il numero di operazioni è quindi direttamente proporzionale a `n`.
*   **Complessità Spaziale: $O(1)$**
    Grazie all'ottimizzazione, utilizziamo un numero costante di variabili (`val1`, `val2`, `current_fib`) indipendentemente dal valore di `n`. Se avessimo usato un array, la complessità spaziale sarebbe stata \( O(n) \).

---

## Tabella di Riepilogo

| Caratteristica            | Approccio Ricorsivo Classico (Top-Down) | Approccio Dinamico Ottimizzato (Bottom-Up) |
| ------------------------- | --------------------------------------- | ------------------------------------------ |
| **Metodologia**           | Ricorsiva, scompone il problema         | Iterativa, costruisce la soluzione         |
| **Complessità Temporale** | $O(2^n)$ (Esponenziale)                 | $O(n)$ (Lineare)                           |
| **Complessità Spaziale**  | $O(n)$ (Stack di ricorsione)            | $O(1)$ (Spazio costante)                   |
| **Efficienza**            | Molto bassa per `n` grandi              | Molto alta e scalabile                     |
# References
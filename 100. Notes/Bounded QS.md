---
Data: 2026-10-25T22:10:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area:
  - "[[Master's Degree.base]]"
---
# Bounded QS

QUICKSORT is frequently referred to as an in-place sorter, because it does not use extra space for ordering the array S. This is true if we limit ourself to the pseudo- code of [[Quick-Sort]], but it is no longer true if we consider the cost of managing the recursive calls.

In fact, at each recursive call, the operating system **must allocate space to save the local variables of the caller**, in order to retrieve them whenever the recursive call ends.

Each recursive call has a space cost of $\Theta(1)$, which has to be mul- tiplied by the number of nested calls QUICKSORT can issue on an array `S[1, n]`. This number can be $\Theta(n)$ in the worst case.

![[Pasted image 20261025223002.png]]

Per garantire un limite superiore rigoroso allo spazio ausiliario, l'algoritmo **BOUNDEDQS** ristruttura il flusso di controllo combinando tre tecniche fondamentali:
1. **Soglia di arresto** $n_0$: La ricorsione non scende fino a sequenze di dimensione 1 o 0. Quando la dimensione dell'intervallo corrente $j - i$ diventa minore o uguale a una costante $n_0$ (tipicamente impostata a qualche decina di elementi), si interrompe la ricorsione e si esegue l'**InsertionSort**
2. La chiamata ricorsiva viene forzata ad avvenire **sempre e soltanto sulla partizione di dimensione minore o uguale alla metà dell'array corrente** ($\le \frac{j-i}{2}$)
3. **Tail-Recursion Elimination**: La chiamata ricorsiva sulla partizione più grande viene eliminata e trasformata in un **ciclo iterativo** **while**, semplicemente riassegnando gli estremi dell'intervallo ($i$ o $j$)

This bring to the following theorem:

> [!success] Theorem
>  **BOUNDEDQS** sorts n atomic items in the RAM model taking $O(n log n)$ expected time, and using O(log n) additional working space.

Poiché la chiamata ricorsiva viene effettuata solo sul sotto-array di dimensione al più pari a $\frac{j-i}{2}$, **ogni chiamata ricorsiva effettiva dimezza la dimensione del sotto-problema da gestire nello stack**. Di conseguenza, la profondità massima dello stack di ricorsione non può mai superare $\lceil \log_2 n \rceil$ record di attivazione nel caso peggiore

# References'
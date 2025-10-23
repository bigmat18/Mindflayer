---
Data: 2025-10-23T14:11:00
Tags:
  - note
  - master
Connection:
  - "[[Programming & Algorithms]]"
Area: "[[Bachelor's Degree]]"
---
# Longest Common Subsequence (LCS)

In questo documento, analizziamo il problema della ricerca della sottosequenza comune più lunga (LCS) tra due sequenze date, utilizzando un approccio basato sulla programmazione dinamica.

---

## 1. Definizioni Preliminari

### Sottosequenza
Data una sequenza `A` di lunghezza `n`, una sequenza `S` di lunghezza `k` è una **sottosequenza** di `A` se `S` può essere ottenuta da `A` eliminando zero o più elementi. Formalmente, devono esistere indici $0 \le i_0 < i_1 < \dots < i_{k-1} \le n-1$ tali che $S[j] = A[i_j]$ per ogni `j` da 0 a `k-1`.

**Esempio:** `(A, D, A)` è una sottosequenza di `(C, A, B, D, A)`.

### Sottosequenza Comune
Una sequenza `S` è una **sottosequenza comune** di `A` e `B` se è una sottosequenza di entrambe.

**Esempio:**
Date le sequenze `A = (A,B,C,B,D,A,B)` e `B = (B,D,C,A,B,A)`, una sottosequenza comune è `(B,C,B,A)`.

Il problema LCS consiste nel trovare, tra tutte le possibili sottosequenze comuni, quella di **lunghezza massima**.

---

## 2. Formulazione del Problema

### Approccio Brute-Force (Inefficiente)
Una prima soluzione potrebbe essere quella di generare tutte le possibili sottosequenze di una delle due sequenze (ad esempio `A`) e, per ciascuna di esse, verificare se è anche una sottosequenza dell'altra (`B`).

Se `A` ha lunghezza `m`, esistono $2^m$ possibili sottosequenze. Verificare ciascuna di esse su `B` (di lunghezza `n`) richiederebbe tempo $O(n)$. Il costo totale sarebbe quindi $O(n \cdot 2^m)$, che è esponenziale e impraticabile per sequenze di medie dimensioni.

### Sottostruttura Ottimale e Ricorrenza
Un approccio più efficiente si basa sulla **programmazione dinamica**. L'idea è di scomporre il problema in sottoproblemi più piccoli e risolverli una sola volta, memorizzandone il risultato.

Consideriamo i prefissi delle sequenze:
*   `A_i`: il prefisso di `A` di lunghezza `i` (i primi `i` caratteri).
*   `B_j`: il prefisso di `B` di lunghezza `j` (i primi `j` caratteri).

Definiamo `LCS(i, j)` come la lunghezza della sottosequenza comune più lunga tra `A_i` e `B_j`. Possiamo definire la seguente relazione di ricorrenza:

1.  **Caso Base**: Se una delle due sequenze è vuota (`i=0` o `j=0`), la LCS ha lunghezza 0.
2.  **Caratteri Finali Uguali**: Se gli ultimi caratteri dei prefissi combaciano (`A[i-1] == B[j-1]`), allora questo carattere fa parte della LCS. La lunghezza totale sarà 1 più la LCS dei prefissi rimanenti (`A_{i-1}` e `B_{j-1}`).
3.  **Caratteri Finali Diversi**: Se `A[i-1] != B[j-1]`, il carattere comune non si trova alla fine. La LCS sarà quindi la più lunga tra le due possibili opzioni:
    *   LCS tra `A_i` e `B_{j-1}` (ignorando l'ultimo carattere di `B`).
    *   LCS tra `A_{i-1}` e `B_j` (ignorando l'ultimo carattere di `A`).

Questo ci porta alla formula:
$$
LCS(i, j) =
\begin{cases}
  0 & \text{se } i = 0 \text{ o } j = 0 \\
  LCS(i-1, j-1) + 1 & \text{se } i, j > 0 \text{ e } A[i-1] = B[j-1] \\
  \max(LCS(i, j-1), LCS(i-1, j)) & \text{se } i, j > 0 \text{ e } A[i-1] \ne B[j-1]
\end{cases}
$$

---

## 3. Soluzione Bottom-Up con Tabella

Implementare la formula precedente con una semplice ricorsione (approccio Top-Down) sarebbe inefficiente a causa della sovrapposizione dei sottoproblemi. Adottiamo quindi un approccio **Bottom-Up**, costruendo una tabella (matrice) `L` di dimensioni `(m+1) x (n+1)` dove `L[i][j]` conterrà il valore di `LCS(i, j)`.

La tabella viene compilata partendo dall'angolo in alto a sinistra (`L[0][0]`) e procedendo riga per riga, colonna per colonna.

*  **Inizializzazione**: La prima riga e la prima colonna della matrice vengono riempite con zeri, corrispondenti al caso base (confronto con una stringa vuota).
*   **Riempimento**: Ogni cella `L[i][j]` viene calcolata in base ai valori delle celle già calcolate (`L[i-1][j-1]`, `L[i-1][j]`, `L[i][j-1]`), seguendo esattamente la logica della formula ricorsiva.

Alla fine del processo, la cella `L[m][n]` conterrà la lunghezza della LCS delle intere sequenze `A` e `B` che corrisponderà al risultato.

```
   0  1  2     n
  _______________
0| 0  0  0  0  0
1| 0  Z  Y1
2| 0 Y2  X
 | 0
m| 0
```

Prendiamo il caso sopra rappresentato, se colonna 2 e riga 2 hanno lo stesso valore siamo nel secondo caso della funzione quindi X = Z + 1, in caso controario si prende il massimo fra Y1 e Y2. Tutta la prima riga e la prima colonna sono a 0 perché se si cerca LCS fra una stringa normale ed una vuota si ha sempre 0.
### Codice di Esempio

```cpp
#include <iostream>
#include <algorithm> // Per std::max

// a: prima sequenza, m: lunghezza di a
// b: seconda sequenza, n: lunghezza di b
int LCS(char a[], char b[], int m, int n)
{
    // Matrice per memorizzare i risultati dei sottoproblemi.
    // L[i][j] conterrà la lunghezza della LCS di a[0..i-1] e b[0..j-1]
    int L[m + 1][n + 1];

    // Riempimento della tabella in modo bottom-up
    for (int i = 0; i <= m; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            // Caso base: una delle due sequenze è vuota
            if (i == 0 || j == 0)
            {
                L[i][j] = 0;
            }
            // Caso 2: Gli ultimi caratteri sono uguali
            else if (a[i - 1] == b[j - 1])
            {
                L[i][j] = L[i - 1][j - 1] + 1;
            }
            // Caso 3: Gli ultimi caratteri sono diversi
            else
            {
                L[i][j] = std::max(L[i - 1][j], L[i][j - 1]);
            }
        }
    }

    // L[m][n] contiene la lunghezza della LCS per le intere sequenze
    return L[m][n];
}
```

### Analisi della Complessità
*   **Complessità Temporale: $O(m \cdot n)$**
    Il costo è dominato dal riempimento della matrice `(m+1) x (n+1)`. Poiché ogni cella viene calcolata in tempo costante $O(1)$, il tempo totale è proporzionale al numero di celle.

*   **Complessità Spaziale: $O(m \cdot n)$**
    L'algoritmo richiede una matrice per memorizzare i risultati intermedi, occupando uno spazio proporzionale alle dimensioni delle due sequenze.

---

## 4. Ricostruire la Sottosequenza

L'algoritmo visto calcola solo la *lunghezza* della LCS. Per ricostruire la sequenza effettiva, si può effettuare un **backtracking** sulla matrice `L` partendo dalla cella `L[m][n]`.

1.  Parti da `(i, j) = (m, n)`.
2.  Se `a[i-1] == b[j-1]`, questo carattere fa parte della LCS. Aggiungilo alla tua soluzione e spostati in diagonale a `(i-1, j-1)`.
3.  Se `a[i-1] != b[j-1]`, confronta `L[i-1][j]` e `L[i][j-1]`. Spostati nella cella con il valore maggiore (se sono uguali, la scelta è indifferente).
4.  Ripeti il processo finché non raggiungi la prima riga o la prima colonna.

Poiché i caratteri vengono trovati in ordine inverso, la sequenza finale dovrà essere invertita.
# References
---
Data: 2025-10-23T14:57:00
Tags:
  - note
  - master
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Dynamic Programming]]"
Area: "[[Bachelor's Degree]]"
---
# Edit Distance (Distanza di Levenshtein)

La **Edit Distance**, o **Distanza di Levenshtein**, è una metrica utilizzata per misurare la "dissimilarità" tra due sequenze di caratteri (stringhe). Essa corrisponde al numero minimo di operazioni di modifica elementari necessarie per trasformare una stringa nell'altra.

Le operazioni consentite sono:
1.  **Inserimento** di un carattere.
2.  **Cancellazione** di un carattere.
3.  **Sostituzione** di un carattere con un altro.

L'obiettivo è trovare un **allineamento ottimo** tra le due sequenze, ovvero quello che minimizza il costo totale. Assegniamo un costo a ogni operazione:
*   **Costo 0**: se due caratteri allineati sono uguali (match).
*   **Costo 1**: se due caratteri allineati sono diversi (sostituzione) o se un carattere è allineato con uno spazio vuoto (inserimento/cancellazione).

**Esempio:**
Per trasformare `S1 = "APE"` in `S2 = "ARPA"`, un allineamento ottimo è:

```
A _ P E  
A R P A
```

Il costo si calcola sommando i costi di ogni colonna:
*   `A` vs `A`: costo 0 (match)
*   `_` vs `R`: costo 1 (inserimento di 'R')
*   `P` vs `P`: costo 0 (match)
*   `E` vs `A`: costo 1 (sostituzione di 'E' con 'A')

Distanza totale = $0 + 1 + 0 + 1 = 2$.

---

## Formulazione Ricorsiva

Per risolvere il problema, possiamo definirlo in termini di sottoproblemi più piccoli. Sia `ED(i, j)` la distanza di edit tra i primi `i` caratteri di `S1` e i primi `j` caratteri di `S2`. Per calcolare `ED(i, j)`, consideriamo le possibili operazioni sull'ultimo carattere di ciascun prefisso:

1.  **Cancellazione**: 
	- Allineiamo `S1[i-1]` con uno spazio. 
	- Trasformiamo `S1[0...i-2]` in `S2[0...j-1]` 
	- e poi cancelliamo `S1[i-1]`. 
	- Costo totale: $1 + ED(i-1, j)$.
2.  **Inserimento**: 
	- Allineiamo `S2[j-1]` con uno spazio. 
	- Trasformiamo `S1[0...i-1]` in `S2[0...j-2]` 
	- e poi inseriamo `S2[j-1]`. 
	- Costo totale: $1 + ED(i, j-1)$.
3.  **Sostituzione/Match**: 
	- Allineiamo `S1[i-1]` con `S2[j-1]`. 
	- Trasformiamo `S1[0...i-2]` in `S2[0...j-2]` e 
	- poi gestiamo l'ultima coppia. 
	- Il costo di quest'ultima operazione è 0 se i caratteri sono uguali, 1 altrimenti. 
	- Costo totale: $costo(S1[i-1], S2[j-1]) + ED(i-1, j-1)$.

La soluzione ottima sarà il minimo tra queste tre opzioni. La relazione di ricorrenza è quindi:

$$
ED(i, j) = \min
\begin{cases}
  ED(i-1, j) + 1 \\
  ED(i, j-1) + 1 \\
  ED(i-1, j-1) + \text{costo}(S1_{i-1}, S2_{j-1})
\end{cases}
$$

I casi base si hanno quando una delle due stringhe è vuota. La distanza tra una stringa di lunghezza `i` e una stringa vuota è `i` (richiede `i` cancellazioni). Quindi, $ED(i, 0) = i$ e $ED(0, j) = j$.

---

## Soluzione con Programmazione Dinamica (Bottom-Up)

Un'implementazione ricorsiva diretta sarebbe molto inefficiente a causa della massiccia sovrapposizione dei sottoproblemi. Utilizziamo quindi la programmazione dinamica, costruendo una matrice `M` di dimensioni `(dim1+1) x (dim2+1)` dove `M[i][j]` conterrà il valore di `ED(i, j)`.

La matrice viene riempita seguendo questi passaggi:

1. **Inizializzazione**:
    - La prima riga viene riempita con i valori da 0 a `dim2`. `M[0][j] = j` rappresenta il costo per trasformare una stringa vuota in un prefisso di `S2` di lunghezza `j` (richiede `j` inserimenti).
    * La prima colonna viene riempita con i valori da 0 a `dim1`. `M[i][0] = i` rappresenta il costo per trasformare un prefisso di `S1` di lunghezza `i` in una stringa vuota (richiede `i` cancellazioni).
2. **Riempimento**: La matrice viene compilata dall'angolo in alto a sinistra verso quello in basso a destra. Ogni cella `M[i][j]` viene calcolata usando la formula ricorsiva, basandosi sui valori delle celle già calcolate: `M[i-1][j]`, `M[i][j-1]` e `M[i-1][j-1]`.

Alla fine del processo, il valore `M[dim1][dim2]` conterrà la distanza di edit tra le due stringhe complete.

### Codice di Esempio

```cpp
#include <iostream>
#include <algorithm>
#include <vector>

// Funzione ausiliaria per calcolare il costo di sostituzione.
// Restituisce 0 se i caratteri sono uguali, 1 altrimenti.
int substitution_cost(char c1, char c2)
{
    return (c1 == c2) ? 0 : 1;
}

int edit_distance(const char s1[], const char s2[], int dim1, int dim2)
{
    // Matrice per memorizzare i risultati dei sottoproblemi.
    // M[i][j] conterrà la edit distance tra i primi 'i' char di s1
    // e i primi 'j' char di s2.
    std::vector<std::vector<int>> M(dim1 + 1, std::vector<int>(dim2 + 1));

    // Inizializzazione della prima colonna (costo di cancellazioni)
    for (int i = 0; i <= dim1; i++)
        M[i][0] = i;

    // Inizializzazione della prima riga (costo di inserimenti)
    for (int j = 0; j <= dim2; j++)
        M[0][j] = j;

    // Riempimento della matrice
    for (int i = 1; i <= dim1; i++)
    {
        for (int j = 1; j <= dim2; j++)
        {
            int cost_cancellazione = M[i - 1][j] + 1;
            int cost_inserimento = M[i][j - 1] + 1;
            int cost_sostituzione = M[i - 1][j - 1] + substitution_cost(s1[i - 1], s2[j - 1]);

            M[i][j] = std::min({cost_cancellazione, cost_inserimento, cost_sostituzione});
        }
    }

    // Il risultato finale si trova nell'angolo in basso a destra.
    return M[dim1][dim2];
}
```

### Analisi della Complessità
- **Complessità Temporale: $O(m \cdot n)$**. Il costo è determinato dal doppio ciclo necessario per riempire la matrice di dimensioni (m+1) x (n+1). Ogni cella viene calcolata in tempo costante.
- **Complessità Spaziale: $O(m \cdot n)$**. È necessario memorizzare l'intera matrice per i calcoli. È possibile ottimizzare lo spazio a $O(\min(m, n))$ mantenendo in memoria solo le ultime due righe (o colonne) necessarie per il calcolo.

# References
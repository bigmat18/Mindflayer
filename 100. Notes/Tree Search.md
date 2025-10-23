---
Data: 2025-10-23T15:52:00
Tags:
  - note
  - master
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Tree Algorithms]]"
Area: "[[Bachelor's Degree]]"
---
# Ricerca di un Valore in un Albero Binario di Ricerca (BST)

La ricerca è l'operazione più fondamentale in un Albero Binario di Ricerca e quella che ne mette in luce la principale efficienza. L'intera struttura dell'albero è progettata per rendere questa operazione veloce.

**Proprietà del BST:**
Per qualsiasi nodo `x` nell'albero, tutti i valori nel suo sottoalbero sinistro sono minori di `x.value`, e tutti i valori nel suo sottoalbero destro sono maggiori. Questa proprietà è la chiave che guida l'algoritmo di ricerca.

---

## 1. Logica dell'Algoritmo

L'algoritmo di ricerca è semplice e diretto. Dato un valore `v` da cercare:
1.  Si inizia dalla radice (`root`) dell'albero.
2.  Si confronta `v` con il valore del nodo corrente, `current.value`:
    *   Se `v == current.value`, il valore è stato trovato. **Successo**.
    *   Se `v < current.value`, si sa per certo che, se il valore esiste, deve trovarsi nel sottoalbero sinistro. Si ripete quindi la ricerca partendo dal figlio sinistro.
    *   Se `v > current.value`, il valore, se esiste, deve trovarsi nel sottoalbero destro. Si ripete la ricerca partendo dal figlio destro.
3.  Se durante questo processo si arriva a un puntatore `nullptr`, significa che il percorso in cui il valore dovrebbe trovarsi è terminato, e quindi il valore **non è presente** nell'albero. **Fallimento**.

---

## 2. Implementazione Ricorsiva

La versione ricorsiva è una traduzione diretta della logica descritta sopra.

*   **Casi Base**:
    1.  Se il nodo corrente è `nullptr`, il valore non è stato trovato.
    2.  Se il valore del nodo corrente corrisponde a quello cercato, il valore è stato trovato.
*   **Passo Ricorsivo**: Se nessuno dei casi base è verificato, si richiama la funzione sul figlio sinistro o destro, a seconda del confronto.

```cpp
/**
 * @brief Cerca un valore in un BST (versione ricorsiva).
 * @param root Il nodo da cui iniziare la ricerca.
 * @param value Il valore da cercare.
 * @return true se il valore è presente, false altrimenti.
 */
bool ricercaABR_R(node* root, int value) {
    // Caso base 1: siamo arrivati in fondo senza trovare il valore.
    if (root == nullptr) {
        return false;
    }

    // Caso base 2: il valore è stato trovato.
    if (value == root->value) {
        return true;
    }

    // Passo ricorsivo: decidiamo in quale sottoalbero continuare la ricerca.
    if (value < root->value) {
        return ricercaABR_R(root->left, value);
    } else {
        return ricercaABR_R(root->right, value);
    }
}
```

---

## 3. Implementazione Iterativa

La versione iterativa esegue la stessa logica utilizzando un ciclo `while` invece della ricorsione. Questa versione è generalmente preferibile in produzione perché evita il consumo di memoria sullo stack delle chiamate e previene il rischio di stack overflow per alberi molto alti e sbilanciati.

```cpp
/**
 * @brief Cerca un valore in un BST (versione iterativa).
 * @param root Il nodo da cui iniziare la ricerca.
 * @param value Il valore da cercare.
 * @return true se il valore è presente, false altrimenti.
 */
bool ricercaABR_I(node* root, int value) {
    node* current = root;

    // Continua a cercare finché non troviamo il valore o non raggiungiamo un nullptr.
    while (current != nullptr && value != current->value) {
        if (value < current->value) {
            current = current->left;
        } else {
            current = current->right;
        }
    }

    // Se il ciclo è terminato, il risultato dipende dal fatto che 'current' sia
    // diventato nullptr (valore non trovato) o meno (valore trovato).
    return current != nullptr;
}
```

*Nota: A volte, le funzioni di ricerca restituiscono un puntatore al nodo (`node*`) invece di un booleano. Questo è utile se, una volta trovato il nodo, si desidera eseguire altre operazioni su di esso. In tal caso, si restituirebbe `current` alla fine, che sarebbe `nullptr` in caso di fallimento.*

---

## 4. Analisi della Complessità

Per entrambe le versioni (ricorsiva e iterativa), l'algoritmo percorre un singolo cammino dalla radice verso il basso. Il numero di nodi visitati è quindi pari alla profondità del nodo cercato, e nel caso peggiore è pari all'altezza dell'albero.

*   **Complessità Temporale: $O(h)$**, dove `h` è l'altezza dell'albero.
    *   **Caso Migliore/Medio (Albero Bilanciato):** Se l'albero è bilanciato, l'altezza `h` è circa $log_2(n)$. La complessità è **$O(\log n)$**, che è molto efficiente.
    *   **Caso Peggiore (Albero Degenerato):** Se l'albero è completamente sbilanciato (simile a una lista concatenata), l'altezza `h` è `n`. La complessità peggiora a **$O(n)$**.

---

## 5. Codice Completo di Esempio

```cpp
#include <iostream>

struct node {
    int value;
    node* left;
    node* right;
};

// Funzione di inserimento corretta per i test
void insert(node** root, int value);
// Dichiarazioni delle funzioni di ricerca
bool ricercaABR_R(node* root, int value);
bool ricercaABR_I(node* root, int value);

int main() {
    node* root = nullptr;

    insert(&root, 15);
    insert(&root, 6);
    insert(&root, 18);
    insert(&root, 7);
    insert(&root, 3);
    insert(&root, 20);

    std::cout << "Ricerca del valore 7 (iterativo): "
              << (ricercaABR_I(root, 7) ? "Trovato" : "Non trovato") << std::endl;

    std::cout << "Ricerca del valore 99 (ricorsivo): "
              << (ricercaABR_R(root, 99) ? "Trovato" : "Non trovato") << std::endl;

    // ... deallocazione dell'albero ...
    return 0;
}

// ... implementazioni complete ...
```

# References
---
Data: 2025-10-23T15:37:00
Tags:
  - note
  - youngling
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Tree Algorithms]]"
Area: "[[Bachelor's Degree]]"
---
# Inserimento di un Nodo in un Albero Binario di Ricerca (BST)

L'inserimento è una delle operazioni fondamentali in un [[Balanced Binary Search Tree (BST)]]. L'obiettivo è aggiungere un nuovo valore all'albero mantenendo la sua proprietà fondamentale.

**Proprietà del BST:**
Per qualsiasi nodo `x` nell'albero:
*   Tutti i valori nel sottoalbero sinistro di `x` sono minori o uguali a `x.value`.
*   Tutti i valori nel sottoalbero destro di `x` sono maggiori di `x.value`.

Questa proprietà garantisce che operazioni come la ricerca, l'inserimento e la cancellazione possano essere eseguite in modo efficiente.

---

## 1. Logica dell'Algoritmo

L'algoritmo per l'inserimento è intuitivo e si basa sulla ricerca della posizione corretta per il nuovo nodo.

1.  **Ricerca della Posizione**:
    *   Si parte dalla radice (`root`) dell'albero.
    *   Si confronta il nuovo valore da inserire con il valore del nodo corrente.
    *   Se il nuovo valore è minore, ci si sposta nel sottoalbero sinistro.
    *   Se il nuovo valore è maggiore o uguale, ci si sposta nel sottoalbero destro.
    *   Questo processo viene ripetuto finché non si raggiunge un puntatore `NULL`. Questo punto `NULL` è la posizione esatta in cui il nuovo nodo deve essere inserito come figlio del nodo visitato per ultimo.

2.  **Inserimento**:
    *   Si crea il nuovo nodo.
    *   Lo si collega come figlio (sinistro o destro, a seconda dell'ultima comparazione) del nodo genitore trovato nel passo precedente.

3.  **Caso Speciale: Albero Vuoto**:
    *   Se l'albero è inizialmente vuoto (`root` è `NULL`), il nuovo nodo diventa semplicemente la radice dell'albero.

---
## 2. Implementazione Corretta e Idiomatica (C++)

Ecco una versione corretta dell'algoritmo che utilizza un puntatore a puntatore per modificare la radice, rendendo il codice robusto anche per alberi vuoti. Ho anche modernizzato leggermente il codice usando `nullptr` e `new` invece di `NULL` e `malloc`.

```cpp
#include <iostream>

struct node
{
    int value;
    node *parent;
    node *left;
    node *right;
};

/**
 * @brief Inserisce un nuovo nodo in un BST.
 *
 * @param root Puntatore al puntatore della radice dell'albero.
 *             Questo permette di modificare la radice se l'albero è vuoto.
 * @param value Il valore da inserire.
 * @return Il puntatore al nodo appena creato, o nullptr in caso di errore.
 */
node* insert(node** root, int value)
{
    // 1. Crea e inizializza il nuovo nodo
    node* new_node = new (std::nothrow) node; // nothrow non lancia eccezioni
    if (new_node == nullptr) {
        std::cout << "Memoria esaurita!" << std::endl;
        return nullptr;
    }
    new_node->value = value;
    new_node->left = nullptr;
    new_node->right = nullptr;
    new_node->parent = nullptr;

    // 2. Trova la posizione corretta per l'inserimento
    node* y = nullptr;      // Puntatore al genitore (trailing pointer)
    node* x = *root;        // Puntatore corrente, parte dalla radice

    while (x != nullptr) {
        y = x; // Aggiorna il genitore
        if (new_node->value < x->value) {
            x = x->left;
        } else {
            x = x->right;
        }
    }

    // 3. Collega il nuovo nodo
    new_node->parent = y;

    if (y == nullptr) {
        // L'albero era vuoto, il nuovo nodo è la radice.
        // Modifichiamo il puntatore originale tramite il doppio puntatore.
        *root = new_node;
    } else if (new_node->value < y->value) {
        // Il nuovo nodo è un figlio sinistro
        y->left = new_node;
    } else {
        // Il nuovo nodo è un figlio destro
        y->right = new_node;
    }

    return new_node;
}

void simmetrica(node* root) {
    if (root != nullptr) {
        simmetrica(root->left);
        std::cout << root->value << " ";
        simmetrica(root->right);
    }
}

int main() {
    // La radice deve essere un puntatore, inizializzato a nullptr.
    node* root = nullptr;

    insert(&root, 15);
    insert(&root, 6);
    insert(&root, 18);
    insert(&root, 7);
    insert(&root, 3);
    insert(&root, 2);
    insert(&root, 4);

    std::cout << "Visita simmetrica dell'albero: ";
    simmetrica(root);
    std::cout << std::endl;

    // ... deallocazione dell'albero ...
    return 0;
}
```

---

## 3. Analisi della Complessità

L'operazione di inserimento richiede una singola discesa dalla radice fino a una foglia (o quasi). Il numero di nodi visitati è quindi limitato dall'altezza dell'albero.

*   **Complessità Temporale: $O(h)$**, dove `h` è l'altezza dell'albero.
    *   **Caso Migliore/Medio (Albero Bilanciato):** Se l'albero è ragionevolmente bilanciato, l'altezza `h` è proporzionale a $\log_2(n)$, dove `n` è il numero di nodi. La complessità è **$O(\log n)$
    *   **Caso Peggiore (Albero Degenerato):** Se gli elementi vengono inseriti in ordine (crescente o decrescente), l'albero degenera in una lista concatenata. In questo caso, l'altezza `h` è uguale a `n`, e la complessità diventa **$O(n)$**.

# References
*   **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).** *Introduction to Algorithms* (4th ed.). MIT Press. (Capitolo 12: "Binary Search Trees", Sezione 12.2: "Querying a binary search tree")

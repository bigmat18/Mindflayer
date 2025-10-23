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
# Cancellazione di un Nodo in un Albero Binario di Ricerca (BST)

La cancellazione di un nodo da un [[Balanced Binary Search Tree (BST)]]  è un'operazione più complessa dell'inserimento, poiché è fondamentale preservare la proprietà del BST: per ogni nodo `x`, tutti i valori nel suo sottoalbero sinistro devono essere minori di `x.value` e tutti i valori nel suo sottoalbero destro devono essere maggiori di `x.value`.

L'algoritmo deve gestire tre scenari distinti, a seconda del numero di figli del nodo da eliminare.

---

## 1. Logica dell'Algoritmo: I Tre Casi

Sia `z` il nodo da eliminare.

### Caso 1: `z` non ha figli (è una foglia)
Questo è il caso più semplice. Per eliminare `z`, è sufficiente modificare il puntatore del suo genitore (`z.parent`) che puntava a `z`, impostandolo a `NULL`, e poi deallocare la memoria di `z`.

```text
      10                        10
     / \                       / \
    5   15   -> remove(7) ->  5   15
   / \                       /
  ... 7                     ...
```

### Caso 2: `z` ha un solo figlio
Anche questo caso è relativamente semplice. Si "scavalca" il nodo `z`, collegando direttamente il genitore di `z` con l'unico figlio di `z`. Il figlio di `z` prende il posto di `z` nel sottoalbero del genitore.

```text
      10                         10
     / \                        / \
    5   15   -> remove(15) ->  5   17
       / \                    / \
      ... 17                ...
```

### Caso 3: `z` ha due figli
Questo è il caso più complesso. Non possiamo semplicemente rimuovere `z`, perché lasceremmo due sottoalberi "orfani". La strategia consiste nel trovare un altro nodo nell'albero che possa prendere il posto di `z` senza violare la proprietà del BST.

Questo nodo sostituto deve essere:
*   Più grande di ogni elemento nel sottoalbero sinistro di `z`.
*   Più piccolo di ogni elemento nel sottoalbero destro di `z`.

Esistono due candidati perfetti:
1.  Il **predecessore** di `z`: il nodo con il valore più grande nel sottoalbero sinistro di `z`.
2.  Il **successore** di `z`: il nodo con il valore più piccolo nel sottoalbero destro di `z`.

La procedura standard è:
1.  Trova il successore (o predecessore) di `z`. Chiamiamolo `y`.
2.  Copia il valore di `y` nel nodo `z` (`z.value = y.value`).
3.  A questo punto, il problema si è ridotto a eliminare il nodo `y` dall'albero. Poiché `y` è il minimo del sottoalbero destro (o il massimo del sinistro), avrà **al massimo un figlio** (il figlio destro se è il minimo, o il sinistro se è il massimo).
4.  Si procede quindi a eliminare `y` usando la logica del Caso 1 o del Caso 2.

---
## 2. Implementazione C++

Un modo standard e robusto per implementare la cancellazione è usare una funzione ausiliaria `transplant` che si occupa di sostituire un sottoalbero con un altro. Questo semplifica notevolmente la logica dei casi 1 e 2.

### Codice di Esempio Corretto

```cpp
#include <iostream>

// La struct del nodo rimane invariata
struct node
{
    int value;
    node *parent;
    node *left;
    node *right;
};

// Funzione di utilità per trovare il minimo in un sottoalbero
node* tree_minimum(node* x) {
    while (x->left != nullptr) {
        x = x->left;
    }
    return x;
}

/**
 * @brief Sostituisce il sottoalbero radicato in 'u' con il sottoalbero radicato in 'v'.
 * Gestisce correttamente i puntatori del genitore di 'u'.
 * @param root Puntatore al puntatore della radice dell'albero.
 * @param u Nodo da sostituire.
 * @param v Nodo che sostituisce.
 */
void transplant(node** root, node* u, node* v) {
    if (u->parent == nullptr) {
        *root = v;
    } else if (u == u->parent->left) {
        u->parent->left = v;
    } else {
        u->parent->right = v;
    }
    if (v != nullptr) {
        v->parent = u->parent;
    }
}

/**
 * @brief Rimuove il nodo 'z' dall'albero radicato in 'root'.
 * @param root Puntatore al puntatore della radice.
 * @param z Nodo da eliminare.
 */
void tree_delete(node** root, node* z) {
    // Caso 1 e 2 (in parte): z ha al massimo un figlio destro
    if (z->left == nullptr) {
        transplant(root, z, z->right);
    }
    // Caso 2 (in parte): z ha solo un figlio sinistro
    else if (z->right == nullptr) {
        transplant(root, z, z->left);
    }
    // Caso 3: z ha due figli
    else {
        // Trova il successore di z (il minimo nel sottoalbero destro)
        node* y = tree_minimum(z->right);

        // Se il successore non è il figlio diretto di z
        if (y->parent != z) {
            // Sostituisci y con il suo (eventuale) figlio destro
            transplant(root, y, y->right);
            // Collega il sottoalbero destro di z a y
            y->right = z->right;
            y->right->parent = y;
        }

        // Sostituisci z con y
        transplant(root, z, y);
        // Collega il sottoalbero sinistro di z a y
        y->left = z->left;
        y->left->parent = y;
    }

    // Dealloca la memoria del nodo rimosso
    free(z);
}

// L'inserimento e la visita rimangono simili
node* insert(node** root, int value);
void simmetrica(node* root);

/* ... Implementazioni di insert e simmetrica ... */
```

### Spiegazione dell'Implementazione
1.  **`tree_minimum(node* x)`**: Funzione ausiliaria corretta per trovare il nodo con valore minimo in un sottoalbero (navigando sempre a sinistra).
2.  **`transplant(root, u, v)`**: È il cuore della logica di sostituzione. Prende un nodo `u` da rimpiazzare e un nodo `v` che lo rimpiazza. Aggiorna correttamente il figlio del genitore di `u` e il genitore di `v`, gestendo anche il caso in cui `u` sia la radice dell'albero.
3.  **`tree_delete(root, z)`**:
    *   **Caso 1/2**: Se `z` non ha un figlio sinistro, viene rimpiazzato dal suo figlio destro (che può essere `nullptr`, coprendo il caso foglia). Se non ha un figlio destro, viene rimpiazzato da quello sinistro. La funzione `transplant` gestisce tutto.
    *   **Caso 3**: Se `z` ha due figli, troviamo il suo successore `y`. La logica si assicura che `y` sia spostato correttamente al posto di `z`, adottando i figli di `z` e mantenendo la struttura dell'albero. Infine, il nodo `z` viene deallocato.

---

## 3. Analisi della Complessità

L'operazione di cancellazione, in tutte le sue varianti, richiede di scendere lungo l'albero per trovare il nodo da eliminare o il suo successore/predecessore.
*   **Complessità Temporale: $O(h)$**, dove `h` è l'altezza dell'albero.
    *   Nel caso di un albero bilanciato, `h` è circa $log_2(n)$, portando a una complessità di $O(\log n)$.
    *   Nel caso peggiore di un albero degenere (simile a una lista concatenata), `h` è `n`, portando a una complessità di $O(n)$.

# References

*   **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).** *Introduction to Algorithms* (4th ed.). MIT Press. (Capitolo 12: "Binary Search Trees", Sezione 12.3: "Deletion")

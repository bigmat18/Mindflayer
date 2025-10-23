---
Data: 2025-10-23T15:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Tree Algorithms]]"
Area: "[[Bachelor's Degree]]"
---
# Ricerca del Minimo e Massimo in un Albero Binario di Ricerca (BST)

La ricerca del valore minimo e massimo in un Albero Binario di Ricerca (BST) sono operazioni molto efficienti. Sfruttano direttamente la proprietà fondamentale dell'albero, che ne garantisce l'ordinamento.

**Proprietà del BST:**
Per qualsiasi nodo `x` nell'albero:
*   Tutti i valori nel suo sottoalbero sinistro sono minori di `x.value`.
*   Tutti i valori nel suo sottoalbero destro sono maggiori di `x.value`.

---

## 1. Ricerca del Valore Minimo

**Logica:**
In base alla proprietà del BST, per trovare l'elemento con il valore più piccolo, dobbiamo semplicemente navigare l'albero il più a sinistra possibile. Partendo dalla radice, continuiamo a spostarci sul figlio sinistro finché non raggiungiamo un nodo che non ha più un figlio sinistro. Quel nodo è il minimo dell'albero.

### Implementazione Ricorsiva
La versione ricorsiva traduce direttamente questa logica:
*   **Caso Base**: Se il nodo corrente non ha un figlio sinistro (`root->left == nullptr`), allora abbiamo trovato il minimo e restituiamo il suo valore.
*   **Passo Ricorsivo**: Altrimenti, richiamiamo la funzione sul figlio sinistro (`root->left`).

```cpp
/**
 * @brief Trova il valore minimo in un BST (versione ricorsiva).
 * @param root Il nodo da cui iniziare la ricerca.
 * @return Il valore minimo nell'albero.
 */
int ricercaMIN_R(node* root) {
    // Gestisce il caso di un albero vuoto per robustezza
    if (root == nullptr) {
        // Potrebbe restituire un valore sentinella o lanciare un'eccezione
        return -1; // o INT_MIN
    }

    if (root->left == nullptr) {
        return root->value;
    } else {
        return ricercaMIN_R(root->left);
    }
}
```

### Implementazione Iterativa
La versione iterativa è spesso preferita perché è leggermente più efficiente (evita l'overhead delle chiamate a funzione) e non rischia di causare uno stack overflow su alberi molto sbilanciati.

```cpp
/**
 * @brief Trova il valore minimo in un BST (versione iterativa).
 * @param root Il nodo da cui iniziare la ricerca.
 * @return Il valore minimo nell'albero.
 */
int ricercaMIN_I(node* root) {
    if (root == nullptr) {
        return -1; // Gestione albero vuoto
    }

    // Continua a scendere a sinistra finché è possibile
    while (root->left != nullptr) {
        root = root->left;
    }
    return root->value;
}
```

---

## 2. Ricerca del Valore Massimo

**Logica:**
Simmetricamente alla ricerca del minimo, per trovare l'elemento con il valore più grande, dobbiamo navigare l'albero il più a **destra** possibile. Partendo dalla radice, continuiamo a spostarci sul figlio destro finché non raggiungiamo un nodo che non ha un figlio destro. Quel nodo è il massimo dell'albero.

### Implementazione Ricorsiva

```cpp
/**
 * @brief Trova il valore massimo in un BST (versione ricorsiva).
 * @param root Il nodo da cui iniziare la ricerca.
 * @return Il valore massimo nell'albero.
 */
int ricercaMAX_R(node* root) {
    if (root == nullptr) {
        return -1; // Gestione albero vuoto
    }

    if (root->right == nullptr) {
        return root->value;
    } else {
        return ricercaMAX_R(root->right);
    }
}
```

### Implementazione Iterativa

```cpp
/**
 * @brief Trova il valore massimo in un BST (versione iterativa).
 * @param root Il nodo da cui iniziare la ricerca.
 * @return Il valore massimo nell'albero.
 */
int ricercaMAX_I(node* root) {
    if (root == nullptr) {
        return -1; // Gestione albero vuoto
    }

    // Continua a scendere a destra finché è possibile
    while (root->right != nullptr) {
        root = root->right; // Correzione: deve andare a destra, non a sinistra
    }
    return root->value;
}
```
---

## 3. Analisi della Complessità

Entrambe le operazioni (minimo e massimo), sia in versione ricorsiva che iterativa, richiedono di percorrere un singolo cammino dalla radice fino a una foglia (o quasi).

*   **Complessità Temporale: $O(h)$**, dove `h` è l'altezza dell'albero.
    *   **Caso Migliore/Medio (Albero Bilanciato):** Se l'albero è bilanciato, l'altezza `h` è circa $log_2(n)$. La complessità è **$O(\log n)$**.
    *   **Caso Peggiore (Albero Degenerato):** Se l'albero è sbilanciato (ad esempio, gli elementi sono inseriti in ordine), l'altezza `h` è uguale a `n`. La complessità diventa **$O(n)$**.

---

## 4. Codice Completo di Esempio

Di seguito un esempio completo e funzionante, che include una funzione di inserimento per poter testare gli algoritmi.

```cpp
#include <iostream>

struct node {
    int value;
    node* left;
    node* right;
    // node* parent; // Opzionale ma utile per altre operazioni
};

// Funzione di inserimento corretta (usa puntatore a puntatore)
void insert(node** root, int value) {
    node* new_node = new node{value, nullptr, nullptr};
    if (new_node == nullptr) {
        std::cout << "Memoria esaurita!" << std::endl;
        return;
    }

    node* y = nullptr;
    node* x = *root;
    while (x != nullptr) {
        y = x;
        if (new_node->value < x->value)
            x = x->left;
        else
            x = x->right;
    }

    if (y == nullptr)
        *root = new_node; // Modifica la radice se l'albero era vuoto
    else if (new_node->value < y->value)
        y->left = new_node;
    else
        y->right = new_node;
}

// Implementazioni delle funzioni di ricerca (come sopra)
int ricercaMIN_R(node* root);
int ricercaMIN_I(node* root);
int ricercaMAX_R(node* root);
int ricercaMAX_I(node* root);

int main() {
    node* root = nullptr; // Inizializzare sempre a nullptr

    insert(&root, 15);
    insert(&root, 6);
    insert(&root, 18);
    insert(&root, 7);
    insert(&root, 3);
    insert(&root, 2);
    insert(&root, 4);
    insert(&root, 13);
    insert(&root, 9);
    insert(&root, 17);
    insert(&root, 20);

    std::cout << "Valore minimo (ricorsivo): " << ricercaMIN_R(root) << std::endl;
    std::cout << "Valore minimo (iterativo): " << ricercaMIN_I(root) << std::endl;
    std::cout << "Valore massimo (ricorsivo): " << ricercaMAX_R(root) << std::endl;
    std::cout << "Valore massimo (iterativo): " << ricercaMAX_I(root) << std::endl;

    // ... qui andrebbe inserita la logica per deallocare l'albero ...

    return 0;
}
```

# References
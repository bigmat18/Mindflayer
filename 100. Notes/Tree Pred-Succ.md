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
# Ricerca del Successore e Predecessore in un BST

Trovare il successore o il predecessore di un nodo è un'operazione comune negli Alberi Binari di Ricerca (BST). Queste operazioni sono fondamentali per algoritmi più complessi, come la cancellazione di un nodo con due figli.

**Definizioni:**
*   Il **Successore** di un nodo `x` è il nodo `y` con il valore più piccolo tra tutti quelli che sono maggiori di `x.value`. In pratica, è il nodo che verrebbe visitato immediatamente dopo `x` in una visita simmetrica (in-order) dell'albero.
*   Il **Predecessore** di un nodo `x` è il nodo `y` con il valore più grande tra tutti quelli che sono minori di `x.value`. È il nodo che verrebbe visitato immediatamente prima di `x` in una visita simmetrica.

---

## 1. Ricerca del Successore

Per trovare il successore di un nodo `x`, ci sono due scenari possibili.

**Caso 1: Il nodo `x` ha un sottoalbero destro**
Se il sottoalbero destro di `x` non è vuoto, allora il successore di `x` si trova per forza in quel sottoalbero. Per la proprietà del BST, tutti i nodi a destra sono maggiori di `x`. Per trovare quello con il valore *più piccolo* tra questi, dobbiamo semplicemente trovare il **minimo del sottoalbero destro**.

```text
      15
     /  \
    6    18  <-- Successore di 15 è il minimo del sottoalbero destro...
   / \   / \
  ... 7 17  20
         ^---- ...cioè 17.
```

**Caso 2: Il nodo `x` NON ha un sottoalbero destro**
Se `x` non ha un sottoalbero destro, il suo successore (se esiste) è un suo **antenato**. Dobbiamo risalire l'albero dal nodo `x` usando i puntatori al genitore. Il successore è il primo antenato `y` tale per cui `x` si trova nel sottoalbero **sinistro** di `y`.

Perché? Finché risaliamo da un figlio destro (`x` è il figlio destro di `y`), significa che `y` è più piccolo di `x`, quindi non può essere il successore. La prima volta che risaliamo da un figlio sinistro, troviamo il primo antenato che è più grande del nodo di partenza.

```text
      15
     /  \
    6    ...
   / \
  3   7
       \
        13 <-- Partiamo da qui (nodo 13)
       /
      9  <-- Per 9, il successore è 13 (Caso 2: 9 è figlio sinistro di 13)
```

### Implementazione Corretta

Le funzioni dovrebbero restituire un puntatore `node*` per poter gestire il caso in cui un successore non esista (ad esempio, per il nodo massimo dell'albero), restituendo `nullptr`.

```cpp
/**
 * @brief Trova il nodo con il valore minimo in un sottoalbero.
 * @param root Il nodo radice del sottoalbero.
 * @return Un puntatore al nodo minimo.
 */
node* tree_minimum(node* root) {
    if (root == nullptr) return nullptr;
    while (root->left != nullptr) {
        root = root->left;
    }
    return root;
}

/**
 * @brief Trova il successore di un dato nodo in un BST.
 * @param x Il nodo di cui trovare il successore.
 * @return Un puntatore al nodo successore, o nullptr se non esiste.
 */
node* successore(node* x) {
    // Caso 1: il nodo ha un sottoalbero destro
    if (x->right != nullptr) {
        return tree_minimum(x->right);
    }

    // Caso 2: non c'è un sottoalbero destro, bisogna risalire
    node* y = x->parent;
    // Risali finché non siamo più un figlio destro
    while (y != nullptr && x == y->right) {
        x = y;
        y = y->parent;
    }
    return y;
}
```

---

## 2. Ricerca del Predecessore

La logica è perfettamente simmetrica a quella del successore.

**Caso 1: Il nodo `x` ha un sottoalbero sinistro**
Il predecessore è il nodo con il valore **massimo nel sottoalbero sinistro** di `x`.

**Caso 2: Il nodo `x` NON ha un sottoalbero sinistro**
Il predecessore è il primo antenato `y` tale per cui `x` si trova nel sottoalbero **destro** di `y`.

### Implementazione Corretta

```cpp
/**
 * @brief Trova il nodo con il valore massimo in un sottoalbero.
 * @param root Il nodo radice del sottoalbero.
 * @return Un puntatore al nodo massimo.
 */
node* tree_maximum(node* root) {
    if (root == nullptr) return nullptr;
    // Correzione del bug: deve navigare a destra, non a sinistra.
    while (root->right != nullptr) {
        root = root->right;
    }
    return root;
}

/**
 * @brief Trova il predecessore di un dato nodo in un BST.
 * @param x Il nodo di cui trovare il predecessore.
 * @return Un puntatore al nodo predecessore, o nullptr se non esiste.
 */
node* predecessore(node* x) {
    // Caso 1: il nodo ha un sottoalbero sinistro
    if (x->left != nullptr) {
        return tree_maximum(x->left);
    }

    // Caso 2: non c'è un sottoalbero sinistro, bisogna risalire
    node* y = x->parent;
    // Risali finché non siamo più un figlio sinistro
    while (y != nullptr && x == y->left) {
        x = y;
        y = y->parent;
    }
    return y;
}
```

---

## 3. Analisi della Complessità

Sia per il successore che per il predecessore, l'algoritmo percorre un singolo cammino dalla radice verso il basso o dal nodo di partenza verso l'alto. La lunghezza di questo cammino è limitata dall'altezza dell'albero.

*   **Complessità Temporale: $O(h)$**, dove `h` è l'altezza dell'albero.
    *   **Caso Medio (Albero Bilanciato):** Se l'albero è bilanciato, $h \approx \log n$, e la complessità è **$O(\log n)$**.
    *   **Caso Peggiore (Albero Degenerato):** Se l'albero è sbilanciato, $h \approx n$, e la complessità diventa **$O(n)$**.

---

## 4. Codice Completo di Esempio

```cpp
#include <iostream>

// ... [Definizione della struct node con parent] ...
// ... [Implementazione di insert corretta con node** root e gestione di parent] ...
// ... [Implementazioni di successore, predecessore, tree_minimum, tree_maximum come sopra] ...

int main() {
    node* root = nullptr;
    insert(&root, 15);
    insert(&root, 6);
    // ... [altri inserimenti] ...
    node* node_13 = insert(&root, 13);
    node* node_17 = insert(&root, 17);

    // Esempio Successore
    node* succ = successore(node_13);
    if (succ != nullptr) {
        std::cout << "Il successore di 13 e': " << succ->value << std::endl; // Stampa 15
    }

    // Esempio Predecessore
    node* pred = predecessore(node_17);
    if (pred != nullptr) {
        std::cout << "Il predecessore di 17 e': " << pred->value << std::endl; // Stampa 15
    }
    
    return 0;
}
```


# References
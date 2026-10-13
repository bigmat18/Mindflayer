---
Data: 2026-09-29T19:37:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Dynamic Programming]]"
Area: "[[Master's Degree.base]]"
---
# Minimum Cost Path

**Problem**: We are given a matrix $M$ of $n \times m$ integers. The goal is to find the minimum cost path to move from the top-left corner to the bottom-right corner by moving only down or to right.

Consider the following matrix:
![[Pasted image 20261005193214.png]]

It is easy to see that the following recurrence solves the problem:
$$W[i][j] = M[i][j] + \begin{cases} 0 & \text{se } i = 0, j = 0 \\ W[i][j-1] & \text{se } i = 0, j > 0 \quad \text{(prima riga: si può solo andare a destra)} \\ W[i-1][j] & \text{se } i > 0, j = 0 \quad \text{(prima colonna: si può solo andare in basso)} \\ \min(W[i-1][j], W[i][j-1]) & \text{se } i > 0, j > 0 \end{cases}$$
Thus, the problem in **linear time** by filling a matrix $W$. Matrix $W$ for the example above is a follows:
![[Pasted image 20261005193728.png]]

The C++ Implementation:
```c++
// Calcola il costo del cammino minimo da (0, 0) a (n-1, m-1)
int minCostPath(const std::vector<std::vector<int>>& M) {
    if (M.empty() || M[0].empty()) return 0;

    int n = M.size();
    int m = M[0].size();

    // Tabella DP per memorizzare i costi minimi
    std::vector<std::vector<int>> W(n, std::vector<int>(m, 0));

    // 1. Cella iniziale (0, 0)
    W[0][0] = M[0][0];

    // 2. Inizializzazione prima riga (movimenti solo verso destra)
    for (int j = 1; j < m; ++j) {
        W[0][j] = W[0][j - 1] + M[0][j];
    }

    // 3. Inizializzazione prima colonna (movimenti solo verso il basso)
    for (int i = 1; i < n; ++i) {
        W[i][0] = W[i - 1][0] + M[i][0];
    }

    // 4. Riempimento delle celle rimanenti
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            W[i][j] = M[i][j] + std::min(W[i - 1][j], W[i][j - 1]);
        }
    }

    // Il costo minimo finale per la cella (n-1, m-1)
    return W[n - 1][m - 1];
}
```
# References
- [[DynamicProgramming.pdf]]

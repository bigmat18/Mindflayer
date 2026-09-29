---
Data: 2026-09-29T18:47:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Dynamic Programming]]"
Area: "[[Master's Degree.base]]"
---
# Subset Sum

**Problem**: given a set $S$ of $n$ non-negative integers, and a value $v$, determine if there is a subset of the given set with sum equal to given $v$.

The problem is a well-know **NP-Hard** problem, which admits a pseudo-polynomial time algorithm. The problem has a solution which is almost the same as 0/1 knapsack problem.

As in the 0/1 knapsack Problem, we construct a matrix `W` with `n+1` rows and `v+1` columns. Here the matrix contains booleans. This is the formulation for each entry:
$$
W[i][j] = \begin{cases}
True & \exists S \subseteq {0..i} \:\:t.c. \sum_{i\in S} i = j\\
False & otherwise
\end{cases}
$$
The entries of the first row and the first columns are set to true (`W[0][]` and `W[][0]`)

So, in the matrix form we can see that an entry `W[i][j]` is true either if `W[i-1][j]` is true or `W[i-1][j - S[i]]` is true. This can be written:
$$
W[i][j] = \begin{cases}
True & i = 0 \text{ or } j = 0\\
True & W[i-1][j] = True \text{ or } W[i-1][j-S[i]] = True \\
False & otherwise
\end{cases}
$$
As an **example**, consider the set $S=\{3, 2, 5, 1\}$ and value $v=6$. Below the matrix $W$ for this example:

![[Pasted image 20260929190303.png]]

This algorithm runs in a **time complexity** of $\Theta(vn)$ as well as a **space complexity** of $O(vn)$.

# References
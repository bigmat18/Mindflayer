---
Data: 2026-09-29T19:34:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Dynamic Programming]]"
Area: "[[Master's Degree.base]]"
---
# Largest Independent Set on Trees

**Problem**: given a binary tree $T$ with $n$ nodes, find one of its largest independent sets. An independent set is a set of nodes $I$ such that there is no edge connecting any pair of nodes in $I$.

![[Pasted image 20260929210310.png]]

In general, the largest independent set is not unique. For example, we can obtain different largest independent set by replacing nodes `a` and `b` in the example above.

Let's consider a **[[Binary Tree Traversal|bottom-up traversal]]** of the tree. For any nodes we have two possibilities: **add** or **not add** the node $u$ to the independent set. 

In the former case, $u$'s children cannot be part of the independent set but its grandchildren could. In the latter case, $u$'s children could be part of the independent set. 
- Let`LIST(u)` be the size of the independent set of the subtree rooted at $u$ 
- Let $C_u$ and $G_u$ be the set of children and the set of grandchildren of $u$

Thus, we have the following reccurrence:
$$
LIST(u) = \begin{cases}
1 & \text{if u is a leaf}\\
\max\{1 + \sum_{v\in G_u} LIST(v), \sum_{v\in C_u} LIST(v)\} & otherwise
\end{cases}
$$
The problem is, thus, solved with a [[Binary Tree Traversal#Post-Order BT Traversal|post-order visit]] of $T$ in a **linear time**.


> [!NOTE] Observation
> Observe that, the same problem applied on a graphs is an NP-hard problem.

Below an example of implementation in C++:
```c++
struct Node {
    int data;
    Node *left, *right;
    Node(int val) : data(val), left(nullptr), right(nullptr) {}
};

// first = incluso, second = escluso
std::pair<int, int> LISS_Helper(Node* root) {
    if (root == nullptr) {
        return {0, 0};
    }

    // Visita post-order: prima i sottoalberi sinistro e destro
    auto left_res = LISS_Helper(root->left);
    auto right_res = LISS_Helper(root->right);

    // Se includiamo root: sommiamo 1 + i valori ESCLUSI dei figli
    int incl = 1 + left_res.second + right_res.second;

    // Se escludiamo root: per ogni figlio prendiamo il max tra incluso ed escluso
    int excl = std::max(left_res.first, left_res.second) + 
               std::max(right_res.first, right_res.second);

    return {incl, excl};
}

int LISS(Node* root) {
    auto res = LISS_Helper(root);
    return std::max(res.first, res.second);
}
```
# References
- https://www.geeksforgeeks.org/dsa/largest-independent-set-problem-using-dynamic-programming/
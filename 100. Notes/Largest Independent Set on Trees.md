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
// A naive recursive implementation of
// Largest Independent Set problem 
#include <bits/stdc++.h>
using namespace std; 

// A utility function to find 
// max of two integers 
int max(int x, int y) 
{ 
    return (x > y) ? x : y; 
} 

/* A binary tree node has data, 
pointer to left child and a 
pointer to right child */
class node 
{ 
    public:
    int data; 
    node *left, *right; 
}; 

// The function returns size of the 
// largest independent set in a given 
// binary tree 
int LISS(node *root) 
{ 
    if (root == NULL) 
    return 0; 

    // Calculate size excluding the current node 
    int size_excl = LISS(root->left) + 
                    LISS(root->right); 

    // Calculate size including the current node 
    int size_incl = 1; 
    if (root->left) 
        size_incl += LISS(root->left->left) +
                     LISS(root->left->right); 
    if (root->right) 
        size_incl += LISS(root->right->left) + 
                     LISS(root->right->right); 

    // Return the maximum of two sizes 
    return max(size_incl, size_excl); 
} 
```
# References
- https://www.geeksforgeeks.org/dsa/largest-independent-set-problem-using-dynamic-programming/
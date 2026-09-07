---
Data: 2026-08-16T00:07:00
Tags:
  - note
  - master
  - article
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Binary Tree Traversal

**Binary trees** are fundamental data structure in many problems. Traversing a binary tree means visiting all the nodes in a specific order. There are several traversal methods, each with its unique applications and benefits.
### In-Order BT Traversal
In-order traversal, the child **left is visited first**, followed by the node itself, and then the **right** child. It follow the **Left - Root - Right** order.

```c++
class GFG {
public:
    static void inOrderTraversal(Node* root) {
        if (root == nullptr) return;

        // Traverse the left subtree
        inOrderTraversal(root->left);

        // Visit the root node
        cout << root->data << " ";

        // Traverse the right subtree
        inOrderTraversal(root->right);
    }
};
```

The **time complexity** is $O(N)$, and, if we don't consider the size of the stack for function calls the **space complexity** is $O(1)$, otherwise $O(h)$ with $h$ is the hight of the tree.
- It can be use to retrieve the values of binary search tree in sorted order.

### Pre-Order BT Traversal
In pre-order traversal, the **node is visited first**, followed by its **left** child and then its **right** child. This can be visualized as **Root - Left - Right**.

```c++
void preOrderTraversal(Node* root) {
    if (root == nullptr) return;

    // Visit the root node
    cout << root->data << " ";

    // Traverse the left subtree
    preOrderTraversal(root->left);

    // Traverse the right subtree
    preOrderTraversal(root->right);
}
```

The **time complexity** is $O(N)$, and, if we don't consider the size of the stack for function calls the **space complexity** is $O(1)$, otherwise $O(h)$ with $h$ is the hight of the tree.
- It can be use to create a copy of the tree (serialization)
### Post-Order BT Traversal
In post-order traversal, the **left** child is visited first, then the **right** child, and finally the **node itself**. This can be visualised as **Left - Right - Root.**

```c++
void postOrderTraversal(Node* root) {
    if (root == nullptr) return;

    // Traverse the left subtree
    postOrderTraversal(root->left);

    // Traverse the right subtree
    postOrderTraversal(root->right);

    // Visit the root node
    cout << root->data << " ";
}
```

The **time complexity** is $O(N)$, and, if we don't consider the size of the stack for function calls the **space complexity** is $O(1)$, otherwise $O(h)$ with $h$ is the hight of the tree.
- It can be use when you want to process child nodes before the parent.

### Level-Order BT Traversal
In level-order traversal, the nodes are visited level by level, starting from the root node and then moving to the next level. This can be visualized as **Level 1 - Level 2 - Level 3 - ...**

```c++
void levelOrderTraversal(Node* root) {
    if (root == nullptr) return;

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* current = q.front();
        q.pop();

        // Visit the root node
        cout << current->data << " ";

        // Enqueue left child
        if (current->left != nullptr) q.push(current->left);

        // Enqueue right child
        if (current->right != nullptr) q.push(current->right);
    }
}
```

This algorithm follow the same idea of the [[Ricerca in ampiezza (BFS)]]. And it has a **time complexity** of $O(N)$ and a **space complexity** of $O(N)$
- It can be use when we need to explore all nodes at current level before next.


# References
- https://www.geeksforgeeks.org/dsa/binary-tree-traversal/
# Leetcode
- [x] [257. Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)
- [x] [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- [ ] [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
- [ ] [107. Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)
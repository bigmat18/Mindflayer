---
Data: 2026-08-30T14:34:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Prefix Sum with Segment Tree

Let's consider the following problem. We have an array `arr[0, ..., n-1]` We should be able to:
- Find the sum of elements from index $l$ to $r$ where $0 \leq l \leq r \leq n-1$
- Change the value of a specified element of the array to a new value $x$. We need to do `arr[i]=0` where $0 \leq i \leq n-1$

To use the [[Segment Tree]] for this problem we have to slight modify its behaviour:
- **Leaf nodes** are the elements of the input array
- Each **Internal node** represents some merging of the lead nodes. The merging may be different for different problems.
- An array represent of tree is used to represent Segment Trees. For each node at index $i$, the left child is at index `(2*i + 1)`, right child at `(2*i + 2)` and the parent is at $\lfloor (i-1) / 2\rfloor$ 

![[Pasted image 20260830184907.png|378]]

### Construction of Segment Tree
We start with a segment `arr[0...n-1]`:
1. every time we divide the current segment in two
2. we call the same procedure on both halves
3. and for each such segment we store the sum in the corresponding nodes

All levels of the constructed segment tree will **be completely filled except the last level**. Also, the tree will be a **Full binary Tree** because we always divide segment in two, at every level.

Since the constructed tree is always a full binary tree with n leaves, there will be $n-1$ internal nodes. So the total number of nodes will be `2*n-1`
### Query for Sum
Once the tree is constructed, how to get the sum using the constructed segment tree. The following is the algorithm to get the sum of elements:
```c
int getSum(node, l, r)   
{  
   if the range of the node is within l and r  
        return value in the node  
   else if the range of the node is completely outside l and r  
        return 0  
   else  
    return getSum(node's left child, l, r) +   
           getSum(node's right child, l, r)  
}
```

In the above implementation there are three cases we need to take into consideration:
1. If the range of the current node while traversing the tree is not the given range then did not add the value of that nodes in ans.
2. If the range of node is partially overlapped with the given range then move either left or right according to the overlapping
3. If the range is completely overlapped by the given range then add it to the ans.

### Update a Value
Like tree construction and query operations, the update can also be done recursively. We are given an index which needs to be updated. Let __diff__ be the value to be added. We start from the root of the segment tree and add __diff__ to all nodes which have given index in their range. If a node doesn't have a given index in its range, we don't make any changes to that node

# References
- https://www.geeksforgeeks.org/dsa/segment-tree-sum-of-given-range/
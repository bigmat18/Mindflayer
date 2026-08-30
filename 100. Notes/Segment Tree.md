---
Data: 2026-08-30T12:31:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Segment Tree

The **Segment Tree** is a data structure used for storing information about **intervals or segments**. It allows querying and which of the stored segments contain a given point.

A segment tree for a set of $n$ intervals uses $O(n\log n)$ **storage complexity**  and the **time complexity** is:
- To build the tree $O(n \log n)$ 
- To query point $O(\log n + k)$ where $k$ is the number of retrieved intervals or segments

### Description
Let $I = {[x_1, x_1'], [x_2, x_2'], \dots, [x_n, x_n']}$ be a set of intervals sorted from left and right. Let's consider the line of real number by those points. 
$$
p_1 < p_2 < p_3 < \dots < p_m \text{ with } m \leq 2n
$$
The regions of this partitioning are called **elementary intervals**, it can be seen like:
$$
(-\infty, p_1), [p_1, p_1], (p_1, p_2), [p_2, p_2], \dots, (p_{m-1}, p_m), [p_m, p_m], (p_m, +\infty)
$$
The number of values on this line is $2m + 1$ divided by:
- **Opened intervals**: $(-\infty, p_1), (p_1, p_2), \dots, (p_{m-1}, p_m), (p_m, +\infty)$
- **Closed intervals degeneri (single points)**: $[p_1, p_1], [p_2, p_2], \dots, [p_m, p_m]$

This is alternated with closed intervals consisting of a single endpoint. Single points are treated themselves as intervals because the answer to a query is not necessarily the same of the interior of an elementary interval and its endpoints. 

Given a set $I$ of intervals, a segment tree $T$ for $I$ is structured as follows:
- $T$ is a **binary tree**
- **Leaves:** Its leaves correspond to the elementary intervals induced by the endpoints in $I$, in an ordered way: the element **leftmost leaf corresponds to the leftmost intervals, and so on**. The elementary intervals corresponding to a leaf $v$ denoted $Int(v)$
- **Internal nodes**: the interval $Int(n)$ corresponding to node $N$ is the union of the intervals corresponding to the leaves of the tree rooted at $N$. That implies that $Int(N)$ is the union of the intervals of two children.
- Each node or leaf $v$ in $T$ stores the interval $Int(v)$ and a set of intervals, in some data structure. This canonical subset of node $v$ contains the intervals $[x, x']$ from $I$ such that $[x, x']$ contains $Int(v)$ and does not contain $Int(parent(v))$. That is, each node in $T$ stores the segments that span through its interval, but do not span through the interval of its parent

![[Pasted image 20260830130955.png|317]]

### Construction
A segment tree from a set of segments $I$ can be built as follows:
1. Extract the **endpoints of intervals** in $I$
2. Sort all the endpoints in increasing order to obtain the **elementary intervals**
3. Now we are going to build the balanced binary tree going to determined for each node $v$ the interval $Int(v)$ it represents. Let's take as example an interval $X=[x, x']$
	1. If $Int(T)$ is contained in X the store X at T and finish
	2. Else
		- If X intersects the interval of the **left** child of T, then insert X in that child **recursively**
		- If X intersects the interval of the **right** child of T, then insert x in that **recursively**
To complete construction we need $O(n \log n)$ operations being $n$ the number of segments in $I$
### Query
A query for a segment tree receives a point $q_k$ (should be one of the leaves of tree), and retrieves a list of segments stored which contain the point $q_k$.

Given a node (subtree) $v$ and a query point $q_x$, the query can be done using the following algorithm:
1. Report all the intervals in $I(v)$. $I(v)$ represents **all the intervals that are fully contained inside the node v** (for the root it will be empty)
2. If $v$ is not a leaf:
	- If $q_x$ is in $Int(v_{left})$ then perform a query in the left child of $v$
	- If $q_x$ is in $Int(v_{right})$ then perform a query in the right child of $v$

In a segment tree that contains $n$ intervals, those containing a given query point can be reported in $O(\log n+k)$ **time complexity** where $k$ is the number of reported intervals.

![[Pasted image 20260830145137.png|409]]

### Rust Implementation
Now I report a Rust implementation of a Segment Tree. This version is slight different compared with the formal definition above, this because:
1. It use a **discrete range of values** $[0, max]$
2. The leaf are single indices not elementary intervals open and closed
3. No distinction between open/closed (only integers)
4. Dynamic structure on a fixed domain
5. Complexity $O(M)$ nodes with segments allocated

Below the code of the **Rust implementation**:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub id: usize,
    pub start: usize,
    pub end: usize,
}

pub struct SegmentTree {
    // The tree: each node contains a vector of segment IDs
    tree: Vec<Vec<usize>>, 
    // The maximum limit of our coordinate axis
    max_val: usize,
    // Store the original segments to return them later
    segments: Vec<Segment>, 
}

impl SegmentTree {
    /// Initializes the tree to cover the range [0, max_val]
    pub fn new(max_val: usize) -> Self {
        // A segment tree requires a size of approximately 4 * N
        SegmentTree {
            tree: vec![Vec::new(); 4 * (max_val + 1)],
            max_val,
            segments: Vec::new(),
        }
    }

    /// Adds a new segment to the tree
    pub fn add_segment(&mut self, start: usize, end: usize) {
        let id = self.segments.len();
        let segment = Segment { id, start, end };
        self.segments.push(segment);
        
        // Start the recursion from the root (node 1), 
        // which covers the range [0, max_val]
        self.insert_recursive(1, 0, self.max_val, start, end, id);
    }

    /// Internal recursive helper to insert the segment 
    // ID into the appropriate nodes
    fn insert_recursive(&mut self, node: usize, l: usize, r: usize, 
					    ql: usize, qr: usize, id: usize) 
	{
        // Base case: the query segment [ql, qr] COMPLETELY 
        // covers the node's range [l, r]
        if ql <= l && r <= qr {
            self.tree[node].push(id);
            return;
        }

        let mid = l + (r - l) / 2;
        let left_node = 2 * node;
        let right_node = 2 * node + 1;

        // If the requested segment overlaps with the left half
        if ql <= mid {
            self.insert_recursive(left_node, l, mid, ql, qr, id);
        }
        // If the requested segment overlaps with the right half
        if qr > mid {
            self.insert_recursive(right_node, mid + 1, r, ql, qr, id);
        }
    }

    /// Finds all segments that contain the point `x`
    pub fn query(&self, x: usize) -> Vec<Segment> {
        // If the query is outside our domain, there are no matching segments
        if x > self.max_val {
            return Vec::new();
        }

        let mut result_ids = Vec::new();
        self.query_recursive(1, 0, self.max_val, x, &mut result_ids);

        // Convert the extracted IDs back into the actual segment structs
        result_ids.into_iter()
            .map(|id| self.segments[id].clone())
            .collect()
    }

    /// Internal recursive helper to traverse the tree down to the leaf `x`
    fn query_recursive(&self, node: usize, l: usize, r: usize, 
					   x: usize, result: &mut Vec<usize>) 
	{
        // 1. Add the segments stored in the current node
        result.extend_from_slice(&self.tree[node]);

        // If we reached a leaf (l == r), we are done
        if l == r {
            return;
        }

        let mid = l + (r - l) / 2;
        let left_node = 2 * node;
        let right_node = 2 * node + 1;

        // 2. Decide whether to go down the left or right child
        if x <= mid {
            self.query_recursive(left_node, l, mid, x, result);
        } else {
            self.query_recursive(right_node, mid + 1, r, x, result);
        }
    }
}
```
##### Insertion
- **The Stop Condition:** If the inserted segment `[ql, qr]` entirely covers the current node `[l, r]`, save the segment in this node and **stop recursing**.
- **Partial Overlap:** If the segment only partially covers the node, calculate `mid` and split the insertion down to the left or right children accordingly.
- **The Reversal:** Unlike typical trees, we do not save smaller objects inside larger containers. We save _larger_ segments onto the _largest possible_ nodes they can completely blanket.
##### Query
- **The Walk:** To find all segments covering point `x`, traverse from the root directly down to the single leaf representing `x`.
- **Accumulation:** At every node along the path, grab all the saved segments and throw them into your "backpack" (the result array).
- **The Guarantee:** Because higher nodes act as protective umbrellas, any massive segment covering a wide area is naturally picked up exactly once as you walk through the parent node.

### [[Nested Segments]]
### [[Prefix Sum with Segment Tree]]
### [[Segment Tree Lazy Propagation]]

# References
- https://en.wikipedia.org/wiki/Segment_tree
- https://cp-algorithms.com/data_structures/segment_tree.html
- https://visualgo.net/en/segmenttree?slide=1
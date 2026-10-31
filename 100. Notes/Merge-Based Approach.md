---
Data: 2026-10-30T14:06:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Set Intersection]]"
Area:
  - "[[Master's Degree.base]]"
---
# Merge-Based Approach

The first basic approach to resolve the problem in [[Introduction to Set Intersection|Set Intersection]] having two ordered array is:
1. Scans $A$ and $B$ from left to right
2. Compares at each step a pair of `docIDs` (we will call them $a_i, b_j$)
	- if $a_i < b_j$ the iterator $i$ gets incremented 
	- if $a_i > b_j$ the iterator $j$ gets incremented
	- if $a_i = b_j$ we have found a doc, and both iterators get incremented

The **correctness** can be proved inductively with this observation: if $a_i < b_j$ (or $a_i > b_j$) then $a_i$ is smaller than all elements following $b_j$ in $B$ so $a_i \notin B$

The **time complexity** is concerned, we note that at each step the algorithm executes one comparison and advances at least one iterator. Given that $n = |A|$ and $m = |B|$ are the number of elements in the two sequences, we can deduce that $i$ and $j$ can advance at most $n$ or $m$, so we can conclude that this algorithm requires no more than $n+m$ comparison.

This time cost is significantly smaller that the one mentioned in [[Introduction to Set Intersection|Set Intercetion brute-force approach]] ($n\times m$). So we can say that:

> [!success] Theorem
> The algorithm based upon the merging paradigm solves the sorted set intersection problem in $O(m+n)$ time.

**Note** that the name **merge** is because this algorithm mimics the `MERGE` procedure used in `MERGESort`.

Some important notes about the **optimality**:
- In the case that $n = \Theta(m)$ this algorithm is optimal because we need to process the smallest set, so $\Theta(\min\{m,n\})$ is an obvious **lower bound**.
- This algorithm is also optimal in the disk model, because it takes $O(n/B)$ I/Os, to be more precise it is optimal whatever the memory hierarchy underlying the computation.

There can be a **problem** when $m$ and $n$ are very different. Let's say $m << n$, in this situation in which one word is much more selective that the other the classic [[Binary Search]] can be helpful because we that design an algorithm that binary search every element $b \in B$ (that is the smallest list) into sorted $A$

This bring to a **time complexity** of $O(m\log n)$ that is better than $O(n+m)$ if we have this big difference between the two sets. So we can say that:

> [!success] Theorem
> The algorithm based on the binary-search paradigm solves the sorted set intersection problem in $O(m\log n)$ time

We can design an algorithm that mech the two approach in [[Mutual Partitioning]]
# References
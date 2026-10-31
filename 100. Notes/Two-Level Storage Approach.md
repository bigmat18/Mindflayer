---
Data: 2026-10-30T14:07:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Set Intersection]]"
Area:
  - "[[Master's Degree.base]]"
---
# Two-Level Storage Approach

To bypass the issues with the high number of recursive call in [[Mutual Partitioning]] and [[Merge-Based Approach]] we can use [[Doubling Search]], but this last approach stil suffering of inefficient when it is executed in a hierarchical memory. 

In order to avoid also this last issue we can use a **Two-Level Stoarage Approach** where the main idea:
1. pre-process all lists of the collection
2. logically partition each of them into blocks of size $L$
3. copy the first element of each block into an auxiliary sequence.
4. This auxiliary sequence will be used in the query to speed up the intersection

**Example**. Let's see an example to understand the approach:
- Consider a set $A$ with length $n$ is a multiple of $L$ (say $n=hL$)
- The preprocessing of $A$ just described creates $h$ blocks $A_i$ of size $L$ each
- copies the first element of each block into auxiliary sequence $A'$ of size $h$ ($A_i[1] = A[(i-1) \times L+1]$)

A **query** will be executed given two sets $A$ and $B$ of the preprocessed collection, their intersection proceeds in two phases:
1. We apply `MERGE` procedure of the [[Merge-Sort]] to fuse together the two sorted sequence of $A'$ and $B'$ into one unique sorted sequence. This takes $O(n/L + m)$ time
	- Let's assume the we have $B_i$ contiguous sub-sequence of $B$'s elements that fall between two consecutive elements of $A'$ (say $A_i[1]$ and $A_{i+1}[1]$)
	- This means that $B_i$'s elements may occur in block $A_i$
2. So we go into the second phase where we execute [[Merge-Based Approach]] set-intersection to compute $A_i \cap B_i$ in time $O(|A_i| + |B_i|) = O(L + |B_i|)$

Summing up the time complexity of the wto pahses we can say this:

> [!success] Theorem
> The algorithm based on the two-level storage paradigm solves the sorted set intersection problem in $O(\frac{n}{L} + mL)$ time and $O(\frac{n}{LB} + \frac{mL}{B} + m)$ I/Os, where $B$ is the disk-page size of the two-level memory model



# References
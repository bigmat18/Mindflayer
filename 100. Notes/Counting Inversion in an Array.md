---
Data: 2026-08-28T23:57:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Counting Inversion in an Array

Following a quick description of the problem:
- We are given an array `A[1..n]` of $n$ positive integers.
- if $1 \leq i < j \leq n$ and $A[i] > A[j]$ the the pair $(i, j)$ is called an **inversion of A**.
- The foal is to count the number of inversion of A

We assume that the largest integer $M$ in array $A$ is in $O(n)$. This assumption is important because we are using a [[Dynamic Prefix Sums with Fenwick Tree|Fenwick Tree]] of size $M$ and building such a data structure takes $O(M)$ **time and space complexity** If, on the other hand, $M$ is too large, we need to sort array $A$ and replace each element with its rank in sorted array.
- We use a [[Dynamic Prefix Sums with Fenwick Tree|Fenwick Tree]] on an array $B$ with $M$ elements, initially all set to 0. 
- We scan array $A$ from left to right
- When processing $A[j]$, we set $B[j]$ to 1
- The number of elements larger than $A[j]$ that we have already processed can be calculated using the `range_sum(j+1, M)` function

A Rust implementation:
```rust
pub fn counting_inversions(a: &[u64]) -> usize {
    if a.is_empty() {
        return 0;
    }

    let max = *a.iter().max().unwrap() as usize;
    let mut ft = FenwickTree::with_len(max + 1);

    let mut count: usize = 0;
    for &e in a {
        count += ft.range_sum((e + 1) as usize, max) as usize;
        ft.add(e as usize, 1);
    }

    count
}
```

The **time complexity** is $O(n\log n)$ because we have to perform $n$ iteration each one we will do an `range_sum` and a `add` both $\log n$.

This algorithm works because with **Fenwick Tree** we satisfy two condition:
1. **Positional condition**: the `A[i]` element happier before `A[j]`, this is possible because we scan from left to right the array.
2. **Value condition**: the value `A[i]` is strictly greater that `A[j]`, this is possible because the Fenwick tree array `B` map each value with its frequency
	- To compute how many inversion have `j` ad right value, we have to count how many numbers yet encountered belongs to the `[e+1, M]` intervals
	- After that, we will register the element already processed, that increase the value `B[e]` to 1 for each steps along the array.

The idea behind using frequency and looking at the range `[i, max]` is that if there are larger values ​​with a frequency of 1 or greater, it means they were encountered before `i`, since we are scanning from left to right, so that constitutes a pair to be counted.
# References
- https://www.geeksforgeeks.org/dsa/inversion-count-in-array-using-merge-sort/
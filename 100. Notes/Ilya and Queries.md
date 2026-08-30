---
Data: 2026-08-23T23:55:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Ilya and Queries

This is an application of the [[Prefix Sum]] pattern, in this case we will use **Rust**. The problem has the followed characteristics:
- We have a string $s = s_1s_2 \dots s_n$ consisting only of characters $a$ and $b$ and we will answer $m$ queries
- Each query $q(l,r)$, where $1 \leq l \leq r \leq n$ asks for the number of positions $i \in [l,r]$ such that $s_i = s_{i+1}$

**Example**: Given string $s = aabbbaaba$. Consider the query $q(3,6)$ . We are interested in the substring $bbba$. So, the answer $2$ for this query is because there are three positions followed by the same symbol, namely position 1, 2, and 4 in the substring.

To resolve this problem in constant time $O(n)$ we can follow the following steps:
1. Compute a **binary vector** $B[1,n]$ such that $B[i] = 1$ if $s_i == s_{i+1}$ otherwise $0$
2. Calculate the prefix sum of $B$ called $P$
3. We will answer the query in **constant time** doing $P[r-1] - P[l-1]$

The Rust implementation is as follows.
```rust
#[derive(Debug)]
struct Ilya {
    psums: Vec<usize>,
}

impl Ilya {
    pub fn new(s: &str) -> Self {
        let psums = s
            .as_bytes()
            .windows(2) // this is used to examinate the string 2 by 2
            .map(|w| if w[0] == w[1] { 1usize } else { 0usize })
            .scan(0, |sum, e| {
                *sum += e;
                Some(*sum)
            })
            .collect::<Vec<_>>();

        Self { psums }
    }

    // Queries use 0-based indexing
    pub fn q(&self, i: usize, j: usize) -> usize {
        assert!(i < j);
        assert!(j <= self.psums.len());

        self.psums[j - 1] - if i != 0 { self.psums[i - 1] } else { 0 }
    }
}
```
# References
- https://pages.di.unipi.it/rossano/blog/2023/prefixsums/
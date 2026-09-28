---
Data: 2026-08-28T23:58:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# Update the Array

In this problem we are given an array $A[1, n]$, initially all the entries are set to 0 and we would like to support two operations:
- `access(i)`: that simply return the value $A[i]$
- `range_update(l, r, v)`: that updates the entries in $A[i..j]$ by adding $v$

In this solution, we utilize a **[[Dynamic Prefix Sums with Fenwick Tree|Fenwick Tree]]** on an array $B[1, n]$, initially populated with zeros. For a `range_update(l, r, v)`:
- we add $v$ to $B[l]$ 
- and subtract $v$ from $B[r+1]$
This ensures that the value at position $i$ is the prefix sum up to $B[i]$

A rust implementation of this solution is the following:

```rust
#[derive(Debug)]
struct UpdateArray {
    ft: FenwickTree,
}

impl UpdateArray {
    pub fn with_len(n: usize) -> Self {
        Self {
            ft: FenwickTree::with_len(n),
        }
    }

    pub fn len(&self) -> usize {
        self.ft.len()
    }

    pub fn access(&self, i: usize) -> i64 {
        self.ft.sum(i)
    }

    pub fn range_update(&mut self, l: usize, r: usize, v: i64) {
        assert!(l <= r);
        assert!(r < self.ft.len());

        self.ft.add(l, v);
        if r + 1 < self.ft.len() {
            self.ft.add(r + 1, -v);
        }
    }
}
```
# References
- https://www.spoj.com/problems/UPDATEIT/
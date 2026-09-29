---
Data: 2026-08-28T23:58:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# DPS with Range Update

In the problem [[Update the Array]] the range update with the `access(i)`  operation. This is easier that the problem we are going to solve here. Here we want to support a new query:
- `range_update(l, r, v)`: this updates the entries in $A[l..r]$ by adding $v$
- `sum(i)`: this return $\sum^{i}_{k=1}A[k]$

We notice that the `add` operation of the original [[Dynamic Prefix Sums with Fenwick Tree|Fenwick Tree]] is just a special case of the `range_update` operation. Moreover, as mentioned above, `access(i)` operations is also supported with two `sum` operations.

We can solve `range_update(l, r, v)` with $j -i + 1$ `add` quires but this will bring to a time complexity if $\Theta((j-i+1)\log n)$ and we want a **time complexity of $\Theta(\log{n})$** independent from the input indices.

The solution is to utilizes **Two Fenwick Trees**. Therefore, we require doubling the space usage to support the more powerful `range_update(l, r, v)` operation

### Single Fenwick Tree
First of all, let's consider a solution using a single Fenwick Tree, denoted as $FT_1$. In our initial approach, we follow a similar strategy used in the [[Update the Array]] problem:
- For a `range_update(r, l, v)`, we modify $FT_1$ by adding $v$ at position $l$ and subtracting $v$ at position $r+1$.
- When we will querying `sum(i)`, we multiply the result from $FT_1$ by $i$. This approach, however, led errors in the results

Let's consider a `range_update(l, r, v)` operation on a brand new Fenwick Tree. The correct result for a query `sum(i)` after the update are the following:
- if $1 \leq i \leq l$, `sum(i)` is 0
- if $l \leq i < r$, `sum(i)` is $v(i - l + 1)$
- if $r \leq i$, `sum(i)` is $v(r - l + 1)$

Instead, the results returned by our implementation of `sum(i)` are the following:
- if $1 \leq i \leq l$, `sum(i)` is 0
- if $l \leq i < r$, `sum(i)` is $v \cdot i = v(l - 1) + v(i-l+1)$
- if $r \leq i$, `sum(i)` is $(v - v)i = 0$

Our initial implementation reports the correct results for $1 \leq i < l$ but introduces errors in other cases. Specifically for **the second case** ($l \leq i \leq r$), int includes an additional term $v(l-1)$, while it erroneously reports $0$ instead of the correct value $v(r-l+1)$ in letter case.
### Second Fenwick Tree
To fix this issue we can introduce a **second Fenwick Tree** that we will call $FT_2$, which will keep track of these discrepancies. When we perform a `range_update(l, r, v)`, we add $-v(l-1)$ to position $l$ and $v\cdot r$ to position $r + 1$ in $FT_2$.

This revised approach ensures that the result of `sum(i)` can be expressed as $a \cdot i + b$, where $a$ represents the sum up to $i$ in $FT_1$ and $b$ is the sum up to $i$ in $FT_2$

The value of $b$ from the second Fenwick tree corrects the errors in the flawed solution:
- For $1 \leq i < l$, $b$ equals $0$
- For $l \leq i \leq r$, $b$ equals $-v(l - 1)$
- For $r < i$, $b$ is $v\cdot r - v(l - 1) = v(r - l + 1)$

The rust implementation of this solution is the following:
```rust
#[derive(Debug)]
struct RangeUpdate {
    ft1: FenwickTree,
    ft2: FenwickTree,
}

impl RangeUpdate {
    pub fn with_len(n: usize) -> Self {
        Self {
            ft1: FenwickTree::with_len(n),
            ft2: FenwickTree::with_len(n),
        }
    }

    pub fn len(&self) -> usize {
        self.ft1.len()
    }

    pub fn sum(&self, i: usize) -> i64 {
        self.ft1.sum(i) * i as i64 + self.ft2.sum(i)
    }

    pub fn access(&self, i: usize) -> i64 {
        self.sum(i) - if i == 0 { 0 } else { self.sum(i - 1) }
    }

    pub fn add(&mut self, i: usize, v: i64) {
        self.range_update(i, i, v)
    }

    pub fn range_update(&mut self, l: usize, r: usize, v: i64) {
        self.ft1.add(l, v);

        self.ft2.add(l, -v * (l as i64 - 1));

        if r + 1 < self.len() {
            self.ft1.add(r + 1, -v);
            self.ft2.add(r + 1, v * r as i64);
        }
    }
}
```
# References
- https://pages.di.unipi.it/rossano/blog/2023/fenwick/
---
Data: 2026-08-24T00:07:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# Little Girl and Maximum
**Problem**: we are given an array `A[1, n]` and a set `Q` of queries. Each query is a range sum query $i, j$ which returns the sum of elements in `A[i, j]`. The goal is to permute the elements in `A` in order to maximize the sum of the results of the queries i `Q`.

The main observation is that if we want to maximize the sum, we have to assign the largest values to the most frequently accessed entries.  This ca me represents as:$$
total = \sum_i A[i] \times F[i]
$$Thus, the solution consists of:
- sorting both `A` by descending values
- and the indexes of `A` by descending frequency of access and pairing them in this order. 

Therefore, once we have computed the frequencies, the solution takes time.

Thus, we are left with the problem of computing access frequencies. In other words, we want to compute the array `F[1, n]` where `F[i]` is the number of times the index $i$ belongs to a query of `Q`. Computing this vector by updating every single entry in `F` for each query takes $O(nq)$ and, is clearly infeasible.

A possible solution is to use [[Sweep Line Algorithm]]. Since the queries represent **intervals**, and calculating the frequencies equates to counting the number of overlapping intervals at each position, we can employ an approach similar to the one used in [[Sweep Line Algorithm|Maximum Number of Overlapping Intervals]] problem.

This solution has a **time complexity** of $O(q\log q)$, due to the comparison-based sorting of intervals endpoints. Since the endpoints in our problem have a maximum value of $n$, we can optimize the solution to run $O(q)$ using counting sort. However there exists an alternative solution bases on [[Prefix Sum]], simpler.

The main idea of this alternative solution is to construct an array `U[1..n]` such that its prefix sums are equal to our target array $F$. Interestingly, we need to modify just two entries of `U` to account for a query `Q`.

Initially, all the entries of `U` are set to 0. For a query `[l, r]` we add 1 to $U[l]$ and subtract `1` from `U[r+1]`. This way, the prefix sum area as follows:
- Unchanged for indexes less than `l`
- Increased by one for indexes in `[l, r]`
- Unchanged for indexes greater than `r`

Therefore, the prefix sum of `U` up to `i` equals `F[i]`. This algorithm takes $O(q + n\log n)$ **time complexity**. Below a rust implementation:

```rust
// We assumes queries are 0-based indexed
pub fn little_girl(a: &[i64], q: &[(usize, usize)]) -> i64 {
    if a.is_empty() {
        return 0;
    }

    let mut u = vec![0i64; a.len()];

    for &(l, r) in q {
        assert!(l <= r);
        assert!(r < u.len());

        u[l] += 1;
        if r + 1 < u.len() {
            u[r + 1] -= 1;
        }
    }

    let mut f = u
        .iter()
        .scan(0, |sum, e| {
            *sum += e;
            Some(*sum)
        })
        .collect::<Vec<_>>();

    // we sort both f and a in decreasing order, nothing changes
    f.sort_unstable();
    let mut a_sorted = a.to_vec();
    a_sorted.sort_unstable();

    a_sorted
        .iter()
        .zip(f)
        .fold(0, |result, (value, freq)| result + value * freq)
}
```
# References
- https://pages.di.unipi.it/rossano/blog/2023/prefixsum
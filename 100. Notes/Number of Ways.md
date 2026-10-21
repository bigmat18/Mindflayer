---
Data: 2026-08-24T00:08:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# Number of Ways

**Problem**: Given an array `A[i, n]`, count the number of ways to split the array into three contiguous parts so that they have the same sums. More formally, you need to find the number if such pairs of indices $i$ and $j$ ($2 \leq i \leq j \leq n-1$) such that:
$$
\sum_{k=1}^{i-1} A[k] = \sum_{k=i}^{j} A[j] = \sum_{k=j+1}^{n} A[k]
$$

For the solution, let `S` be the sum of the values in the array:
1. If 3 doesn't divide `S` we conclude that the result is zero.
2. Otherwise:
	1. We compute an array `C` that stores, at position `i` the number of suffixes of `A[i..n]` that sum to `S/3`
	2. That scan `A` from left to right to compute the prefix sums
		- Every time the prefix sum at position `i` is `S/3` we add `C[i+2]` to the result. 

This is because the past `A[1..i]` sums to `S/3` and can be combined with any pair of parts of `A[i+1..n]` where both parts sums to `S/3`. Since the values in `A[i+1..n]` sum to `2/3S`, the number of such pairs is the number of suffixes that sum to `S/3` in `A[i+2..n]`. Indeed, if one of this suffix sum to `S/3` say `A[j..n]`, then we are sure that `A[i+1, j-1]` sums to `S/3`

Here the Rust Implementation:
```rust
pub fn number_of_ways(a: &[i64]) -> usize {
    let sum: i64 = a.iter().sum();

    if sum % 3 != 0 {
        return 0;
    }

    let target = sum / 3;
    let mut c: Vec<_> = a
        .iter()
        .rev()
        .scan(0, |sum, e| {
            *sum += e;
            Some(*sum)
        })
        .scan(0, |counter, sum| {
            if sum == target {
                *counter += 1usize
            };
            Some(*counter)
        })
        .collect();

    c.reverse();

    let mut result = 0;
    let mut sum = 0;
    for (i, &v) in a[..a.len() - 2].iter().enumerate() {
        sum += v;
        if sum == target {
            result += c[i + 2];
        }
    }

    result
}
```

# References
- https://pages.di.unipi.it/rossano/blog/2023/prefixsums/
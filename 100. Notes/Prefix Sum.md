---
Data: 2026-07-19T22:35:00
Tags:
  - note
  - master
  - article
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
 # Prefix Sum

This pattern is often used to solve problems that ask the **sum of elements between two indices** in an array operations on subarrays. Using the prefix sum technique
- After a one-time pre-processing in $O(n)$ time, each range sum query can be answered in $O(1)$ time
- If there are $q$ queries, the overall time complexity becomes $O(n+q)$

### How to use
If we need the sum of elements in a sub range $[L, R] \in [0, N]$ where $N$ is the size of the array the result can be obtained by subtracting the prefix sum at index $L-1$ from the prefix at index $R$. **Note** that if $L=0$ the sum is simply equal to prefix in position $[R]$

We can also write this property in the following way:
$$
sum(L, R) = sum(0, R) - sum(0, L) \:\:\:\text{ where }\:\:\: sum(L, R) = [L, R]
$$

![[Pasted image 20260719224522.png | 400]]

![[Pasted image 20260719224531.png | 400]]

One important things is that sometime we can avoid an extra array, reducing the space complexity, using the input array itself, like in this python example
```python
def create_prefix_sum(arr)
	for i in range(1, len(arr)):
		arr[i] = arr[i] + arr[i-1]
		
	return arr
```

### Prefix Sum in Rust
In Rust, the combinator [`scan`](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.scan) can produce the prefix sums (and much more) from an iterator. `scan` is an iterator adapter that bears similarity to [fold](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.fold). Similar to `fold`, `scan` maintains an **internal state**, initially set to a seed value, which is modified by a closure taking both the current internal state and the current element from the iterator into account.

The distinction between `scan` and `fold` is that the **former produces a new iterator** with all the states taken by its internal state, whereas the latter only **returns the value of the final internal state**.

```rust
let a = vec![2, 4, 1, 7, 3, 0, 4, 2];

let psums = a
    .iter()
    .scan(0, |sum, e| {
        *sum += e;
        Some(*sum)
    })
    .collect::<Vec<_>>();

assert!(psums.eq(&vec![2, 6, 7, 14, 17, 17, 21, 23]));
```

### [[Subarray Sum Equals to K]]
### [[Ilya and Queries]]
### [[Little Girl and Maximum]]
### [[Number of Ways]]
### [[Dynamic Prefix Sums with Fenwick Tree]]
### [[Prefix Sum with Segment Tree]]

# References
- https://www.geeksforgeeks.org/dsa/understanding-prefix-sums/
# Leetcode
- [x] [303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/description/)
- [x] [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)
- [x] [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
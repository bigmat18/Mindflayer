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

### [[Subarray Sum Equals to K]]
### [[Ilya and Queries]]
### [[Little Girl and Maximum]]
### [[Number of Ways]]

### [[Dynamic Prefix Sums with Fenwick Tree]]


# References
- https://www.geeksforgeeks.org/dsa/understanding-prefix-sums/
# Leetcode
- [x] [303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/description/)
- [x] [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)
- [x] [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
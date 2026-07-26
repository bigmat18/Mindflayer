---
Data: 2026-07-19T22:35:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Prefix Sum

This pattern is often used to solve problems that ask the sum of elements between two indices in an array operations on subarrays. Using the prefix sum technique
- After a one-time pre-processing in $O(n)$ time, each range sum query can be answered in $O(1)$ time
- If there are $q$ queries, the overall time complexity becomes $O(n+q)$

### Compute Sum for a Query
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


### Prefix Sum Basic Example
In this section I want take as example the problem **Subarray Sum Equals K** as a starting point to explain how to apply prefix sum into a real example, and also to use it to achive from $O(N^3)$ a complexity to $O(N)$.

#### Problem Description
Given an array of integers `nums` and an integer `k`, return _the total number of subarrays whose sum equals to_ `k`. A subarray is a contiguous **non-empty** sequence of elements within an array.

> **Example 1:**
> - **Input:** nums = [1,1,1], k = 2
> - **Output:** 2
> 
> **Example 2:**
> - **Input:** nums = [1,2,3], k = 3
> - **Output:** 2

#### Brute Force solution $O(N^3)$
With these types of problem the first approach is always try the brute force approach, it will be wrong in terms of complexity, but it could help to figure out a starting point to optimize with **prefix sum**. For example for this prloblem it could be something like this:

```rust
    pub fn subarray_sum(nums: Vec<i32>, k: i32) -> i32 {
        let mut result : i32 = 0;

        for i in 0..nums.len() {
            for j in i..nums.len() {
                let mut counter : i32 = 0;
                for v in i..=j {
	                counter += nums[v]
                }
                if counter == k {
                    result += 1;
                }
            } 
        }
        return result;
    }
```

This solution bring to a computational complexity to $O(N^3)$ that is the most of the case to high to pass the basics examples. To you can optimize the inner for loop replacing it with a pre-computation. 
#### Prefix Sum solution $O(N^2)$
This is the most of the times the second stage to try for these kinds of problmes, remove the stupid inner for loop with a pre-computation of the **prefix sum**.

```rust
    pub fn subarray_sum(nums: Vec<i32>, k: i32) -> i32 {
        let mut result : i32 = 0;

        let mut sum : Vec<i32> = vec![0; nums.len()];
        sum[0] = nums[0];
        for i in 1..nums.len() {
            sum[i] = nums[i] + sum[i-1];
        }

        for i in 0..nums.len() {
            for j in i..nums.len() {
                let mut left : i32 = if i != 0 {sum[i-1]} else {0};
                let mut counter : i32 = sum[j] - left;
                if counter == k {
                    result += 1;
                }
            } 
        }
        return result;
    }
```

You can now use the **pre-computation** to reduce the time complexity to $O(N^2)$. Most of the times these kinds of sulutions are not enough.

#### Prefix Sum and HashMap $O(N)$
The last stage is to remove the second loop using an **map**. This allow to:
1. Doing a single for loop, while you compute the **prefix sum you can add it into this map**. This allow to retrive faster what do you need
2. The prefix sum is doing with an **accumulator**
3. Depending on the question, you can query the map to retrive the position where there is what you need, after that you have the range: `[query_idx, actual_idx]` and you can do whatever you want

```rust
use std::collections::HashMap;

impl Solution {
    pub fn subarray_sum(nums: Vec<i32>, k: i32) -> i32 {
        let mut result : i32 = 0;
        let mut map = HashMap::new();
        map.insert(0, 1);

        let mut sum = 0;
        for i in 0..nums.len() {
            sum += nums[i];

            if let Some(&freq) = map.get(&(sum - k)) {
                result += freq;
            }
            *map.entry(sum).or_insert(0) += 1;
        }
        return result;
    }
}
```

In this case we have to find the number of ranges where the sum is equal to $k$, to **you can ask to the map if there is an element before that summing removed it from the actual sum bring to $k$.** In this case you add it (actually you add the freq, bacause there could be more than one)

![[Pasted image 20260725230415.png | 300]] ![[Pasted image 20260725230429.png | 300]]
![[Pasted image 20260725230535.png | 300]] ![[Pasted image 20260725230547.png|300]]
![[Pasted image 20260725230558.png | 300]] ![[Pasted image 20260725230609.png | 300]]
![[Pasted image 20260725230619.png | 300]] ![[Pasted image 20260725230636.png | 300]]

### Applications
- [x] [303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/description/)
- [x] [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)
- [x] [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

# References
- https://www.geeksforgeeks.org/dsa/understanding-prefix-sums/
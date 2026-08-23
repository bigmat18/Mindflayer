---
Data: 2025-10-14T18:39:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Kadane's Algorithm

This is a classic problem that is in the [[Greedy Algorithms]] types. The problem ask to find the contiguous subarray with the largest sum.

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        curr_sum = 0
        max_sum = float('-inf')

        for num in nums:
            # Greedy choice: reset if current sum is negative
            # Negative prefix can only reduce future sums
            if curr_sum < 0:
                curr_sum = 0

            # Add current number to running sum
            curr_sum += num

            # Update maximum sum seen so far
            max_sum = max(max_sum, curr_sum)

        return max_sum
```

The approach is ti track running sum, reset to 0 when sum becomes negative (greedy choice: discard negative prefix). 
- **Time Complexity**: $O(n)$ singles ass through array
- **Space Complexity**: $O(1)$ Only tracking two variables.

# References
- [Wikipedia Page](https://en.wikipedia.org/wiki/Maximum_subarray_problem)
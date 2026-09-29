---
Data: 2026-09-29T19:37:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Dynamic Programming]]"
Area: "[[Master's Degree.base]]"
---
# Longest Bitonic Subsequence

**Problem**: Given an array **arr[]** containing n positive integers, find the length of the **longest bitonic subsequence** A subsequence of numbers is called **bitonic** if it is first strictly increasing, then strictly decreasing. 

> [!NOTE]
> Only strictly increasing (no decreasing part) or a strictly decreasing sequence should not be considered as a bitonic sequence.

**Example**: 
- **Input**: Consider the sequence `S=[2, -1, 4, 3, 5, -1, 3, 2]`
- **Output**: The bitonic subsequence is `-1, 4, 3, 1` to output `4`

- **Input**: `S = [1, 2, 5, 3, 2]`
- **Output**: We have first `[1, 2, 5]` increasing and then `[3, 2]` decreasing, the full `S` is bitonic, so `5`

The idea is to compute the [[Longest Increasing Subsequnce]] from left to right, and the [[Longest Increasing Subsequnce]] from right to left by using the $O(n\log n)$ time solution.

Then, we combine these two solution to find the longest bitonic subsequence. This is done by taking the values in a colum, adding them and subtracting one. This will be correct because `LIS[i]` computes the longest increasing subsequence of `S[1, i]` and ends with `S[i]`

![[Pasted image 20260929213653.png]]
 
So if we look the code:
1. For every index $i$, calculate:
	- `lis[i]` = length of the longest strictly increasing subsequence ending at $i$
	- `lds[i]` = length of the longest strictly decreasing subsequence starting at $i$
2. if $i$ is the peak, then `lis[i] + lds[i] - 1` gives the length of the bitonic subsequence having `arr[i]` as its peak.

Below an C++ implementation:
```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int longestBitonicSequence(vector<int>& arr) {
    int n = arr.size();

    if (n < 3)
        return 0;

    vector<int> lis(n, 1);
    vector<int> lds(n, 1);

    // Compute LIS ending at every index.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                lis[i] = max(lis[i], lis[j] + 1);
            }
        }
    }

    // Compute LDS starting at every index.
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[i]) {
                lds[i] = max(lds[i], lds[j] + 1);
            }
        }
    }

    int maxLen = 0;

    for (int i = 0; i < n; i++) {
        // Both increasing and decreasing parts must exist.
        if (lis[i] > 1 && lds[i] > 1) {
            maxLen = max(maxLen, lis[i] + lds[i] - 1);
        }
    }

    return maxLen;
}
```

This implementation consider only indices where `lis[i] > 1 && lds[i] > 1`, because both the increasing and decreasing parts must exist. This implementation has **time complexity of $O(n²)$** and **space complexity of $O(n)$**.

# References
- https://www.geeksforgeeks.org/dsa/longest-bitonic-subsequence-dp-15
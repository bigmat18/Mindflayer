---
Data: 2026-09-29T19:36:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Dynamic Programming]]"
Area: "[[Master's Degree.base]]"
---
# Longest Increasing Subsequnce

**Problem**: GIven an array `arr[]` of size `n`, find the length of the **Longest Increasing Sub-sequence (LIS)**. A LIS is the longest sub-sequence in which the element of the sub-sequence are sorted in stricly increasing order.

**Example**:
- **Input**: `S = [10, 22, 9, 21, 33, 50, 41, 60, 80]`.
- **Output**: The LIS is `[10, 22, 33, 50, 60, 80]` or `[9, 21, 33, 50, 60, 80]` so the length is `6` (as we can see in this example the LIS is not unique)

### Using [[Dynamic Programming]]
The idea is to maintain a 1D array `lis[]`, where `lis[i]` stores the length of the longest increasing sub-sequence that ends at index $i$, Initially, each element in `lis[]` is set to `1`, as the smallest possible subsequence for any element is the element itself.

1. For each element `arr[i]`, we check all previous elements `arr[0]` to `arr[i-1]`:
	- if `arr[i]` is greater than `arr[prev]` then we update `lis[i]` to maximum of its current value of `lis[prev] + 1`
2. By the time we reach the end of the array, `lis[i]` already contains the correct LIS ending at `i`, because all smaller subproblems (earlier indices) have been computed. Finally, the overall LIS is the maximum value in the lis array.

```c++
#include <vector>
#include <iostream>
using namespace std;

int lis(vector<int>& arr) {
    int n = arr.size();
    vector<int> lis(n, 1);

    // Compute optimized LIS values in
    // bottom-up manner
    for (int i = 1; i < n; i++) {
        for (int prev = 0; prev < i; prev++) {
            if (arr[i] > arr[prev]) {
                lis[i] = max(lis[i], lis[prev] + 1);
            }
        }
    }

    // Return maximum value in lis
    return *max_element(lis.begin(), lis.end());
}
```

### Using [[Binary Search]]
The idea is to simulate the process of finding a subsequence by maintaining a list of "buckets" where each bucket represents a valid subsequence. Initially, start with an empty list and iterate through the input vector arr from left to right.

For each number in arr, perform the following steps:
1. If the number is greater than the last element of the last bucket (i.e., the largest element in the current subsequence), **we append the number to the end of the list**. This indicates that we have found a longer subsequence.
2. Otherwise, perform a binary search on the list of buckets to find the **smallest element that is greater than or equal to the current number**. This step helps maintain the property of increasing elements in the buckets.
3. Once find the position to update, replace that element with the current number. This keeps the buckets sorted and ensures that we have the potential for a longer subsequence in the future.

```c++
int lengthOfLIS(vector<int>& arr) {

    int n = arr.size();
    vector<int> ans;

    // Initialize the answer vector with the
    // first element of arr
    ans.push_back(arr[0]);

    for (int i = 1; i < n; i++) {
        if (arr[i] > ans.back()) {

            // If the current number is greater
            // than the last element of the answer
            // vector, it means we have found a
            // longer increasing subsequence.
            // Hence, we append the current number
            // to the answer vector.
            ans.push_back(arr[i]);
        }
        else {

            // If the current number is not
            // greater than the last element of
            // the answer vector, we perform
            // a binary search to find the smallest
            // element in the answer vector that
            // is greater than or equal to the
            // current number.

            // The lower_bound function returns
            // an iterator pointing to the first
            // element that is not less than
            // the current number.
            int low = lower_bound(ans.begin(), ans.end(), arr[i]) - ans.begin();

            // We update the element at the
            // found position with the current number.
            // By doing this, we are maintaining
            // a sorted order in the answer vector.
            ans[low] = arr[i];
        }
    }

    // The length of the answer vector
    // represents the length of the
    // longest increasing subsequence.
    return ans.size();
}
```

This implementation has a **time complexity** of $O(n\log n)$ and **space complexity** of $O(n)$
# References
- https://www.geeksforgeeks.org/dsa/longest-increasing-subsequence-dp-3/
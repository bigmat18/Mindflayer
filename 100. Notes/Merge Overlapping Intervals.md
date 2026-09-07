---
Data: 2026-08-23T19:29:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Merge Overlapping Intervals

To analyze better this pattern let's take this problem: Given an array of time intervals where `arr[i] = [start_i, end_i]` our task is to merge all the overlapping intervals into one and output the result which should have only mutually exclusive intervals. 

**Examples:**

> **Input:** `arr[] = [[1, 3], [2, 4], [6, 8], [9, 10]]` 
> **Output:** `[[1, 4], [6, 8], [9, 10]]`  
> **Explanation:** In the given intervals, we have only two overlapping intervals `[1, 3]` and `[2, 4]`. Therefore, we will merge these two and return `[[1, 4]], [6, 8], [9, 10]]`.
> 
> **Input:** `arr[] = [[7, 8], [1, 5], [2, 4], [4, 6]]`  
> **Output:** `[[1, 6], [7, 8]]`  
> **Explanation:** We will merge the overlapping intervals `[[1, 5], [2, 4], [4, 6]]` into a single interval `[1, 6]`.

### Naive Approach
A simple approach is to group all the intervals by sorting them then start from the first interval and compare it with all other intervals for overlaps. If the first intervals overlaps with any other interval, then remove the other interval from the list and mrge the other into the first one.

```c++
vector<vector<int>> mergeOverlap(vector<vector<int>> &arr) {
    int n = arr.size();

    sort(arr.begin(), arr.end());
    vector<vector<int>> res;

    // Checking for all possible overlaps
    for (int i = 0; i < n; i++) {
        int start = arr[i][0];
        int end = arr[i][1];

        // Skipping already merged intervals
        if (!res.empty() && res.back()[1] >= end)
            continue;

        // Find the end of the merged range
        for (int j = i + 1; j < n; j++) {
            if (arr[j][0] <= end)
                end = max(end, arr[j][1]);
        }
        res.push_back({start, end});
    }
    return res;
}
```

This approach is simple but take at least $O(n^2)$ **Time complexity** and $O(n)$ space complexity.

###  Expected Approach
In the previous approach, for each range we are checking for possible overlaps by iterating over all the remaining ranges till the end. 

We can optimize this by checking only **those intervals that overlap with the last merged interval**. Since the intervals will be sorted based on starting point, so if we encounter an interval whose starting time lies outside the last merged interval, then all further intervals will also lie outside it.

![[Pasted image 20260816000545.png|400]]

1. [Sort](https://www.geeksforgeeks.org/dsa/sorting-algorithms/) the intervals based on their starting points so that overlapping intervals appear consecutively
![[Pasted image 20260816000307.png|400]]

2. Iterate through the sorted intervals while maintaining the last merged interval.
![[Pasted image 20260816000338.png|400]]

3. If the current interval overlaps with the last merged interval, merge them by updating the ending point.
![[Pasted image 20260816000404.png|400]]

![[Pasted image 20260816000509.png|400]]

4. Otherwise, append the last merged interval to the result and start a new merged interval with the current interval.
![[Pasted image 20260816000450.png|400]]

The code for this approach has a **time complexity** of $O(n\log{n})$ and **space complexity** of $O(n)$
```c++
vector<vector<int>> mergeOverlap(vector<vector<int>>& arr) {

    // Sort intervals based on start values
    sort(arr.begin(), arr.end());
  
    vector<vector<int>> res;
    res.push_back(arr[0]);

    for (int i = 1; i < arr.size(); i++) {
        vector<int>& last = res.back();
        vector<int>& curr = arr[i];

        // If current interval overlaps with the last merged
        // interval, merge them 
        if (curr[0] <= last[1]) 
            last[1] = max(last[1], curr[1]);
        else 
            res.push_back(curr);
    }

    return res;
}
```

# References
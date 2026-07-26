---
Data: 2026-07-26T00:19:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Sliding Window

This pattern is useful when there are problems that ask for **finding subarrays with a specific sum**, finding the **longest substring with unique characters**, or solving problems that require a **fixed-size window to process elements efficiently**. Many of these problems can easly be solved in $O(n^2)$ complezity, with this pattern it decrease to $O(n)$.

The key pointers of this patter are:
- Instead of repeatedly iterating over the same elements, the sliding window **maintains a range (or “window”) that moves step-by-step through the data**, updating results incrementally.
- The main idea is to **use the results of previous window to do computations for the next window**.

### How to use Sliding Window
There are basically two types of sliding window that could be possibily be identified from a native solution.
1. **Fixed Size Sliding Window**
	- Find the size of the window requires (say K)
	- Compute the result for the 1st window (initialize the data structure for the first K elements)
	- Then loop to slide the window by 1 and keep computing the result window by window
	
2. **Variable Size Sliding Window**
	- **Increase right:** in this type of siding window we increase out right pointer one by one till out condition is true or we achieve the end of array
	- **Increase left:** if the condition does not match, we shrink the size of our window by increasing left pointer and restore the property

### Sliding Window Basic Example
We analize the example called **Maximum Sum of a Subarray with K Elements** to understand how to applay this pattern correctly. This is the problem statements:

Given an array `arr[]` and an integer `k`, we need to calculate the maximum sum of a subarray having size exactly `k`.

> *Input :* `arr[] = [5, 2, -1, 0, 3], k = 3`  
> *Output :* `6`  
> *Explanation :* We get maximum sum by considering the subaarray `[5, 2 , -1]`
> 
> *Input  :*  `arr[] = [1, 4, 2, 10, 23, 3, 1, 0, 20], k = 4`   
> *Output :* `39`  
> ****Explanation**** : We get maximum sum by adding subarray `[4, 2, 10, 23]` of size 4

#### Native Approach $O(nk)$
The first, as always, is to write a naive approach, this facilitate to understand the type of the problem, and allow to unlock the situations.
```c++
int maxSum(vector<int>& arr, int k) {
    int n = arr.size();
    int max_sum = INT_MIN;

    // Consider all blocks starting with i
    for (int i = 0; i <= n - k; i++) {
        int current_sum = 0;

        // Calculate sum of current subarray of size k
        for (int j = 0; j < k; j++)
            current_sum += arr[i + j];

        // Update result if required
        max_sum = max(current_sum, max_sum);
    }

    return max_sum;
}
```

This is a very simple brute force solution, we iterate over the array and the check the sum of the elements inside each possibile subarray of size $k$. It's clear that the inner loop can be removed.

#### Sliding Window $O(n)$
We use the sliding window patter in the case 1
1. We compute the sum of the first k element using `window_sum` var
2. Then we will traverse linearly over the array till it reaches the end
3. For each iteration the decrease the `window_sum` with the element out the window, and we add to it the new element in the window, and we check if this is the new max.

```c++
int maxSum(vector<int>& arr, int k){
    int n = arr.size();

    // n must be greater
    if (n <= k) {
        cout << "Invalid";
        return -1;
    }

    // Compute sum of first window of size k
    int max_sum = 0;
    for (int i = 0; i < k; i++)
        max_sum += arr[i];

    // Compute sums of remaining windows by
    // removing first element of previous
    // window and adding last element of
    // current window.
    int window_sum = max_sum;
    for (int i = k; i < n; i++) {
        window_sum += arr[i] - arr[i - k];
        max_sum = max(max_sum, window_sum);
    }

    return max_sum;
}
```

With this implementation we achieve easy the $O(n)$ time complexity with $O(1)$ space compleaxity, so, without any additional space.

![[Pasted image 20260726222711.png | 300]]

![[Pasted image 20260726222728.png | 300]]

![[Pasted image 20260726222743.png | 300]]

### Applications
- [x] [643. Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)
- [x] [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [ ] [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/)

# References
- https://www.geeksforgeeks.org/dsa/window-sliding-technique/
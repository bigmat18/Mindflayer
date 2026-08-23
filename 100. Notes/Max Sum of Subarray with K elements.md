---
Data: 2026-08-23T00:18:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Max Sum of Subarray with K elements

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

# References
---
Data: 2025-09-18T16:50:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Trapping Rain Water

###### Problem 
Given an array `arr[]` of size n consisting of non-negative integers, where each element represents the height of a bar in an elevation map and the width of each bar is 1, determine the total amount of water that can be trapped between the bars after it rains.

![[Pasted image 20250918165303.png]]

**Input:** height = `[0,1,0,2,1,0,1,3,2,1,2,1]`
**Output:** 6
**Explanation:** The above elevation map (black section) is represented by array `[0,1,0,2,1,0,1,3,2,1,2,1]`. In this case, 6 units of rain water (blue section) are being trapped.

### Brute Force
In this approach for each elements in array we find the highest bars on left and right sides. We take the smaller of two. The difference from smaller and the current in the amount of water in this location (a pillar of water).

```c++
int maxWater(vector<int>& arr) {
    int res = 0;

    // For every element of the array
    for (int i = 1; i < arr.size() - 1; i++) {

        // Find the maximum element on its left
        int left = arr[i];
        for (int j = 0; j < i; j++)
            left = max(left, arr[j]);

        // Find the maximum element on its right
        int right = arr[i];
        for (int j = i + 1; j < arr.size(); j++)
            right = max(right, arr[j]);

        // Update the maximum water
        res += (min(left, right) - arr[i]);
    }

    return res;
}
```

- **Time Complexity**: $O(n²)$
- **Space Complexity**: $O(1)$
### Prefix and suffix max for each index
A different approach to reduce time complexity is avoid double loop and pre-calculate left and right max value for each element in array, and, subsequently calculate the water pillar for each element.

![[Pasted image 20250918172426.png | 450]]

![[Pasted image 20250918172503.png | 450]]

```c++
int maxWater(vector<int>& arr) {
    int n = arr.size();

    // left[i] contains height of tallest bar to the
    // left of i'th bar including itself
    vector<int> left(n);

    // right[i] contains height of tallest bar to
    // the right of i'th bar including itself
    vector<int> right(n);

    int res = 0;

    // fill left array
    left[0] = arr[0];
    for (int i = 1; i < n; i++)
        left[i] = max(left[i - 1], arr[i]);

    // fill right array
    right[n - 1] = arr[n - 1];
    for (int i = n - 2; i >= 0; i--)
        right[i] = max(right[i + 1], arr[i]);

    // calculate the accumulated water element by element
    for (int i = 1; i < n - 1; i++) {
        int minOf2 = min(left[i], right[i]);
            res += minOf2 - arr[i];
    }

    return res;
}
```

- **Time Complexity**: $O(3n) \approx O(n)$
- **Space Complexity**: $O(2n) \approx O(n)$ because we need two array with same size of input.
### Using Two Pointers
In this approach we do the following considerations:
- Consider a sub-array of input `arr[left...right]` we can decide the amount of water either for `arr[left]` or `arr[right]` if we know the left max (max element in `arr[0...left-1]`) and right max (max element in `arr[right+1 ... n-1]`)

- if left max is less than right max, we can decide for `arr[left]`, else for `arr[right]`
![[Pasted image 20250918174644.png | 450]]

- If we decide for `arr[left]`, then the amount of water would be `left max - arr[left]` and if we decide for `arr[right],` then the amount of water would be `right max - arr[right]`.

```c++
int maxWater(vector<int> &arr) { 
    int left = 1;
    int right = arr.size() - 2;

    // lMax : Maximum in subarray arr[0..left-1]
    // rMax : Maximum in subarray arr[right+1..n-1]
    int lMax = arr[left - 1];
    int rMax = arr[right + 1];

    int res = 0;
    while (left <= right) {
      
        // If rMax is smaller, then we can 
        // decide the amount of water for arr[right]
        if (rMax <= lMax) {
          
            // Add the water for arr[right]
            res += max(0, rMax - arr[right]);

            // Update right max
            rMax = max(rMax, arr[right]);

            // Update right pointer as we have 
            // decided the amount of water for this
            right -= 1;
        } else { 
            // Add the water for arr[left]
            res += max(0, lMax - arr[left]);

            // Update left max
            lMax = max(lMax, arr[left]);

            // Update left pointer as we have 
            // decided water for this
            left += 1;
        }
    }
    return res;
}
```

- **Time Complexity**: $O(3n) \approx O(n)$
- **Space Complexity**: $O(1)$ Because we don't need to store all left/right max for each element but we iterate moving the two pointer
### Using Stack
This approach we use **[next greater](https://www.geeksforgeeks.org/dsa/next-greater-element/)** and **[previus greater](https://www.geeksforgeeks.org/dsa/previous-greater-element/)** problems to solve this one. For each element the water trapped can be determined by the minimum height between the previous and next greater element.

```c++
int maxWater(vector<int>& arr) {
    stack<int> st;  
    int res = 0;
    for (int i = 0; i < arr.size(); i++) {
       
        // Pop all items smaller than arr[i]
        while (!st.empty() && arr[st.top()] < arr[i]) {          
            
            int pop_height = arr[st.top()];
            st.pop();
          
            if (st.empty())
                break;

            // arr[i] is the next greater for the removed item
            // and new stack top is the previous greater 
            int distance = i - st.top() - 1;
          
            // Take the minimum of two heights (next and prev greater)
            // and find the amount of water that we can fill in all
            // bars between the two
            int water = min(arr[st.top()], arr[i]) - pop_height;

            res += distance * water;
        }
        st.push(i);
    }
    return res;
}
```

- **Time Complexity**: $O(n)$
- **Space Complexity**: $O(n)$ because we need to store the stack of elements with size of input.
# References
- [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/trapping-rain-water/)
- [Leetcode](https://leetcode.com/problems/trapping-rain-water/description/)
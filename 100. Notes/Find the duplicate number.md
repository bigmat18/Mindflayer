---
Data: 2026-09-28T18:38:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# Find the duplicate number

The problem is the following: We are given an array `A[0, ..., n]` of `n+1` numbers in `{0, ..., n-1}`. By pigeonhole principale, there must be at least a duplicated elements `e`. Own goal is to find the duplicate number (or more)

### Naive Approach
The main Idea is to iterate through each element and check if it already exists in the result. If not, we scan the rest of the array to see if it appears again. If it does, we add it to the result. This bring to a **Time Complexity** of $O(n²)$ and a **Space Complexity** of $O(1)$

### Frequency Array
The main Idea is to traverse the array once and count the occurrences of each element using a frequency array. Then, we iterate through the array to collect elements whose frequency 2, indicating they are duplicates.

```c++
int findDuplicates(vector<int>& arr) {
    
    int n = arr.size();
    
    // frequency array of size n+1 (1-based indexing)
    vector<int> freq(n + 1, 0);  

    // collect elements that appear exactly twice
    for (int i = 0; i < n; i++) {
        if (freq[i] == 2) {
            return i;
        } else {
	        freq[i]++
        }
    }

    return ans;
}
```

This bring to a **Time Complexity** of $O(n)$ and a **Space Complexity** of $O(n)$

### Negative Marking Approach
The main Idea is to iterate over the array and use the value of each element as an index (after subtracting 1).
- If the element at that index is positive, we negate it to mark the number as visited.
- If it's already negative, it means we've seen it before, so we add it to the result.

This avoids extra space and works in linear time.

```c++
vector<int> findDuplicates(vector<int>& arr) {

    for (int i = 0; i < arr.size(); i++) {
        
        // convert value to index (1-based to 0-based)
        int idx = abs(arr[i]) - 1; 

        // if already visited, it's a duplicate
        if (arr[idx] > 0){
            
            // mark as visited by negating
            arr[idx] = -arr[idx]; 
        }
        else{
            return idx;
        }
    }
    return ans;
}
```

The **step-by-step** approach is:
1. Traverse the array once, using each value `arr[i]` to compute an index `idx = abs(arr[i]) - 1`.
2. Check the value at `arr[idx]`:  
	- If it is positive, mark it as visited by setting `arr[idx] = -arr[idx]`.  
	- If it is already negative, it means `arr[i]` has been seen before, so it's a duplicate.
3. Return the duplicate value idx

This bring to a **Time Complexity** of $O(n)$ and a **Space Complexity** of $O(1)$. This approach is more efficient in terms of space complexity but **it must destroy the array structure and this can be an issue**.

![[Pasted image 20260928185918.png|426]]

### Slow Fast Pointer
We can use the same approach above the viewing the array input as a list with at least one cycle, we need to identify this cycle. This can be done with [[Floyd's Slow and Fast Pointers]]. In **Time Complexity** of $O(n)$ and a **Space Complexity** of $O(1)$ without destroying the array structure.

# References
- https://www.geeksforgeeks.org/dsa/duplicates-elements-in-an-array/
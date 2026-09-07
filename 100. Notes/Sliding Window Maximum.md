---
Data: 2025-09-23T18:15:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Sliding Window Maximum

You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

**Input:** `nums = [1,3,-1,-3,5,3,6,7], k = 3`
**Output:** `[3,3,5,5,6,7]`
**Explanation:** 
```
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       4
 1  3  -1  -3  5 [3  6  7]      7
```

### Brute-Force Approach
The simplest approach to address this problem involves handling **each of the windows independently**. Within each window, we calculate its maximum by scanning through all its elements, which takes $O(k)$ times. Consequently this approach operates in $O(nk)$ times

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {

	vector<int> result;
	vector<int> slides;
	
	for (int i = 0; i < nums.size(); i++) {
	
		// inizialization of slides window
		if (slides.size() < k) {
			slides.push_back(nums[i]);	
			
		// after inizialization
		} else if (slides.size() == k) {
			slides.erase(slides.begin());
			slides.push_back(nums[i]);
		}
		
		// if slides window is inizialized found the max value
		if (slides.size() == k) {
			int max = slides[0];
	
			for (int j = 1; j < slides.size(); ++j) 
				max = std::max(max, slides[j]);
	
			result.push_back(max);
		}
	}
	return result;
}
```


### BST-based Solution
When we transition form one window to the next, only two elements change: the first element of the first widow exits from the scene, and the last element of the second one enters. Consequently, we can derive the multiset of the new window from the multi-set of the previous window by simply adding one element and removing another one.

A possible data-structure capable of performing three crucial operations on a multiset (**insert** new element, **deleting** an arbitrary element, and efficiently **retrieving the maximum** element within the multiset). A possible solution is to use is a **[[Balanced Binary Search Tree (BST)]]** that supports the there operation in $O(\log |M|)$ where $|M|$ is t< number of elements in the multiset.

```rust
use std::collections::BTreeSet;

pub fn bst(nums: &Vec<i32>, k: usize) -> Vec<i32> {
    let n = nums.len();
    if k > n { 
        return Vec::<i32>::new();
    }

    let mut maxs = Vec::with_capacity(n - k + 1);

    let mut set = BTreeSet::new();
    let mut max_sf = nums[0];

    for (i, &v) in nums.iter().enumerate() {
        set.insert((v, i));

        // keep track of the max so far to avoid a costly query to the set
        max_sf = max_sf.max(v); 

        if i >= k {
            set.remove(&(nums[i - k], i - k));
            if max_sf == nums[i - k] {
                max_sf = set.last().unwrap().0;
            }
        }
        if i >= k - 1 {
            maxs.push(max_sf);
        }
    }

    maxs
}
```

This is a Rust implementation of this strategy, it use **BTreeSet** that is a BST-like data structure that represents a set of unique, ordered elements.

### Heap-based Solution
An alternative to BST solution is using an **Heap** data-structure. Theoretically is slightly less efficient then BST-based version, however, this solution in many cases is more efficient.

In this version we use a **Max-Heap queue** with the following operations:
- Insert an element $O(\log n )$
- Take the max element $O(1)$
- Pop max element $O(\log n)$

```c++
// Method to find the maximum for each
// and every contiguous subarray of size k.
vector<int> maxOfSubarrays(const vector<int>& arr, int k) {
    int n = arr.size();

    // to store the results
    vector<int> res;

    // to store the max value
    priority_queue<pair<int, int> > heap;

    // Initialize the heap with the first k elements
    for (int i = 0; i < k; i++)
        heap.push({ arr[i], i });

    // The maximum element in the first window
    res.push_back(heap.top().first);

    // Process the remaining elements
    for (int i = k; i < arr.size(); i++) {

        // Add the current element to the heap
        heap.push({ arr[i], i });

        // Remove elements that are outside the current
        // window
        while (heap.top().second <= i - k)
            heap.pop();

        // The maximum element in the current window
        res.push_back(heap.top().first);
    }

    return res;
}
```

We simply use the same approach used is BST-based solution replacing  BST with heap, and each time chack if the top element is outside the window.
### Linear Time Solution
We can optain a best solution using a simple **Deque** of size $k$ that stores only useful elements of current window of $k$ elements. An elements is useful if it is in the current window and is greater than all other elemetns on tight side of it in current window.

![[Pasted image 20251014222932.png]]

The **Deque** must support **insert**, **remove** and access in costant time at the begin and end. We start with an empty deque, and start to add and remove element based on their usefulness


```c++
// Method to find the maximum for each
// and every contiguous subarray of size k.
vector<int> maxOfSubarrays(vector<int>& arr, int k) {

    // to store the results
    vector<int> res;
  
    // create deque to store max values
    deque<int> dq(k);

    // Process first k (or first window) elements of array
    for (int i = 0; i < k; ++i) {
      
        // For every element, the previous smaller elements 
        // are useless so remove them from dq
        while (!dq.empty() && arr[i] >= arr[dq.back()]) {
          
            // Remove from rear
            dq.pop_back();
        }

        // Add new element at rear of queue
        dq.push_back(i);
    }

    // Process rest of the elements, i.e., from arr[k] to arr[n-1]
    for (int i = k; i < arr.size(); ++i) {
      
        // The element at the front of the queue is the largest 
        // element of previous window, so store it
        res.push_back(arr[dq.front()]);

        // Remove the elements which are out of this window
        while (!dq.empty() && dq.front() <= i - k) {
          
            // Remove from front of queue
            dq.pop_front();
        }

        // Remove all elements smaller than the currently being 
        // added element (remove useless elements)
        while (!dq.empty() && arr[i] >= arr[dq.back()]) {
            dq.pop_back();
        }

        // Add current element at the rear of dq
        dq.push_back(i);
    }

    // store the maximum element of last window
    res.push_back(arr[dq.front()]);

    return res;
}
```

#### Time Complexity
First we have a loop with $n$ iterations, the cost of each iteration is based on the cost of pop operations. However, in a iteration we should extract every elements in the deque. There is until $n$ elements in the deque then an iteration can be $O(n)$. With this analysis the algorithm has a time complexity in bad case $O(n²)$.

But with empiric analysis the time become linear because the heavy iterations are rare, and they are depreciated by small iterations (very often).

# References
- https://pages.di.unipi.it/rossano/blog/2023/swm/
- https://leetcode.com/problems/sliding-window-maximum/description/
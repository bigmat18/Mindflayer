---
Data: 2025-09-23T18:15:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Slinding Window Maximum

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

### Heap-based Solution

### Linear Time Solution
# References
- [Notes](https://pages.di.unipi.it/rossano/blog/2023/swm/)
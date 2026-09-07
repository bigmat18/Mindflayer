---
Data: 2026-07-26T00:19:00
Tags:
  - note
  - master
  - article
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Sliding Window

This pattern is useful when there are problems that ask for **finding subarrays with a specific sum**, finding the **longest substring with unique characters**, or solving problems that require a **fixed-size window to process elements efficiently**. Many of these problems can easly be solved in $O(n^2)$ complezity, with this pattern it decrease to $O(n)$.

The key pointers of this patter are:
- Instead of repeatedly iterating over the same elements, the sliding window **maintains a range (or “window”) that moves step-by-step through the data**, updating results incrementally.
- The main idea is to **use the results of previous window to do computations for the next window**.

### How to use
There are basically two types of sliding window that could be possibily be identified from a native solution.
1. **Fixed Size Sliding Window**
	- Find the size of the window requires (say K)
	- Compute the result for the 1st window (initialize the data structure for the first K elements)
	- Then loop to slide the window by 1 and keep computing the result window by window
	
2. **Variable Size Sliding Window**
	- **Increase right:** in this type of siding window we increase out right pointer one by one till out condition is true or we achieve the end of array
	- **Increase left:** if the condition does not match, we shrink the size of our window by increasing left pointer and restore the property

### [[Max Sum of Subarray with K elements]]
### [[Sliding Window Maximum]]


# References
- https://www.geeksforgeeks.org/dsa/window-sliding-technique/
# Leetcode
- [x] [643. Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)
- [x] [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [ ] [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/)
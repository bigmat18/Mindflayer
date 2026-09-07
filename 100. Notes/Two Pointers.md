---
Data: 2025-10-14T18:24:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Two Pointers
The Two-Pointers Technique is a simple yet powerful strategy where you use two indices (pointers) that traverse a data structure. Two pointers is really an easy and effective technique that is typically used for: **Two Sum in Sorted Arrays**, **[[Trapping Rain Water]]** or **Three/Four Sum**. This approach has pointers starting at opposite ends of the data structure and moving inward toward each other:

![[Pasted image 20251014183041.png | 350]]

### How to use
The pointers move toward the center, adjusting their positions based on comparisons, until a certain condition is met, or they meet/cross each other. This is ideal for problems where we need to compare elements from different ends of a data structure.

This pattern is very useful in ***Sorted Input***, If the array or list is already sorted (or can be sorted), two pointers can efficiently find pairs or ranges. Example: Find two numbers in a sorted array that add up to a target.

It can be also used as general case for the followings problems:
- [[Sliding Window]]: 
	- **(Fixed Window)** When you need to maintain a window of elements that grows/shrinks based on conditions. Example: Find smallest subarray with sum ≥ K, move all zeros to end while maintaining order.
	- **(Variable Window)** When the problem asks about two elements, subarrays, or ranges instead of working with single elements. Example: Longest substring without repeating characters, maximum consecutive ones, checking if a string is palindrome.
- [[Fast and Slow Pointers]] Detecting cycles, finding the middle node, or checking palindrome property. Example: Floyd’s Cycle Detection Algorithm (Tortoise and Hare).

### [[Trapping Rain Water]]


# References
- https://www.geeksforgeeks.org/dsa/two-pointers-technique/
# Leetcode
- [x] [167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
- [x] [15. 3Sum](https://leetcode.com/problems/3sum/)
- [ ] [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
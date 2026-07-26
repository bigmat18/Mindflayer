---
Data: 2025-10-14T18:24:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Two Pointers
The Two-Pointers Technique is a simple yet powerful strategy where you use two indices (pointers) that traverse a data structure. Two pointers is really an easy and effective technique that is typically used for: **Two Sum in Sorted Arrays**, **[[Trapping Rain Water]]** or **Three/Four Sum**.
## Same Direction
Ideal for problem where we process or scan data in a single pass like in Liked List. In this approach, both pointers start at the same end of the data structure (usually the beginning) and move in the same direction:

![[Pasted image 20251014183120.png]]

These pointers generally serve two different but supplementary purposes. A common application of this is when we want one pointer to find information (usually the right pointer) and another to keep track of information (usually the left pointer).

> **Fast-Slow Pointers**
> Classic case of two pointers in same direction used in liked lists. With this technique, also called **runner** technique, we have a fast pointer and a slow pointer. The **fast** be ahead by a fixed amount, or it might be hopping multiple nodes for each node that the **slow** pointer iterates though.

## Opposite Direction
This approach has pointers starting at opposite ends of the data structure and moving inward toward each other:

![[Pasted image 20251014183041.png]]

The pointers move toward the center, adjusting their positions based on comparisons, until a certain condition is met, or they meet/cross each other. This is ideal for problems where we need to compare elements from different ends of a data structure.

## When To Use Two Pointers
- ***Sorted Input :*** If the array or list is already sorted (or can be sorted), two pointers can efficiently find pairs or ranges. Example: Find two numbers in a sorted array that add up to a target.
- **Pairs or Subarrays :** When the problem asks about two elements, subarrays, or ranges instead of working with single elements. Example: Longest substring without repeating characters, maximum consecutive ones, checking if a string is palindrome.
- ***Sliding Window Problems :*** When you need to maintain a window of elements that grows/shrinks based on conditions. Example: Find smallest subarray with sum ≥ K, move all zeros to end while maintaining order.
- **Linked Lists (Slow–Fast pointers) :*** Detecting cycles, finding the middle node, or checking palindrome property. Example: Floyd’s Cycle Detection Algorithm (Tortoise and Hare).

### Applications
- [x] [167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
- [x] [15. 3Sum](https://leetcode.com/problems/3sum/)
- [ ] [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

# References
- https://www.geeksforgeeks.org/dsa/two-pointers-technique/
---
Data: 2026-07-27T21:27:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Fast and Slow Pointers

This approach is a side versione of the [[Two Pointers]] pattern where, instead of using two pointer one for the start and one for the end with let them start both to the zero position, increasing them in two different way one **slower** and one **faster**.

Ideal for problem where we process or scan data in a single pass like in **Linked Lists** In this approach, both pointers start at the same end of the data structure (usually the beginning) and move in the same direction:

![[Pasted image 20251014183120.png]]

These pointers generally serve two different but supplementary purposes. A common application of this is when we want one pointer to find information (usually the right pointer) and another to keep track of information (usually the left pointer).

Is usefull in particular to answer at the following question:
- We need to find if there is a **cycles** inside a list or an array
- We need to find the **middle value** con a list 

### [[Floyd's Slow and Fast Pointers ]]


# References
- [Github Page with Pattern Explanation](https://github.com/Chanda-Abdul/Several-Coding-Patterns-for-Solving-Data-Structures-and-Algorithms-Problems-during-Interviews/blob/main/%E2%9C%85%20%20Pattern%2003:%20Fast%20%26%20Slow%20pointers.md)
- [Floyd's Slow and Fast Algorithm](https://www.geeksforgeeks.org/dsa/how-does-floyds-slow-and-fast-pointers-approach-work/)
- [Wikipedia Page](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)
# Leetcode
- [x] [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- [x] [202. Happy Number](https://leetcode.com/problems/happy-number/)
- [ ] [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

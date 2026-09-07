---
Data: 2026-07-27T21:36:00
Tags:
  - note
  - master
  - article
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Monotonic Stack

A **monotonic stack** is a special type of **stack** where elements are kept in either **increasing** or **decreasing** order. The idea is to maintain this order while pushing and popping elements.

![[Pasted image 20260815130037.png | 500]]

When a new element is pushed, it is compared with the top of the stack. If the order is violated, elements are popped until the property is restored, and then the new element is pushed.

### [[Next Greater or Smaller Element in Array]]


# References
- https://www.geeksforgeeks.org/dsa/introduction-to-monotonic-stack-2/
# Leetcode
- [x] [496. Next Grater Element I](https://leetcode.com/problems/next-greater-element-i/)
- [x] [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- [ ] [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
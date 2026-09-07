---
Data: 2026-08-30T19:05:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Segment Tree Lazy Propagation

In this example [[Prefix Sum with Segment Tree]] I used segment tree to resolve Sum of given range problem with a strategy of **Lazy propagation**.

> [!NOTE]
> In syntesis **Lazy propagation** is an technique to update ranges in $O(\log n)$ instead of $O(n)$ or $O(k\log n)$. The core idea is to post-pone the update until it is strictly necessary
> 

The problem is in that example, update function was called to update only a single value in array. Please note that a single value update in array may **cause multiple updates in Segments Tree**, as there may be many segment tree nodes that have a single array element in their ranges.

To optimise this problem using **Lazy propagation** we can do the following thing: when there are many updates and updates are done on a range, we can postpone some update and do those update only when required.

![[Pasted image 20260830192044.png|298]]

**Example**
Let's consider the node with value 27 in above diagram, this node stores sum of values at indexes from 3 to 5. If our update query is for range 2 or 5, then **we need to update this node and all descendants of this node.** 

With Lazy propagation we update **only node with value 27** and postpone updates to its children by storing this update information in separate nodes called lazy nodes or values.

We create an array `lazy[]` which represents lazy node. Size of `lazy[]` is same as array that represents segment tree.
- Initialise all values inside lazy as 0, that means that there are not update
- If the value is not zero means that this amount needs to be added to node i.

Below the pseudo-code that explain this approach:
```
// To update segment tree for change in array
// values at array indexes from us to ue.
updateRange(us, ue)
1) If current segment tree node has any pending
   update, then first add that pending update to
   current node.
2) If current node's range lies completely in 
   update query range.
....a) Update current node
....b) Postpone updates to children by setting 
       lazy value for children nodes.
3) If current node's range overlaps with update 
   range, follow the same approach as above simple
   update.
...a) Recur for left and right children.
...b) Update current node using results of left 
      and right calls.
```

This approach takes a **time complexity of $O(n)$** instead of $O(n\log n)$ 
# References
- https://www.geeksforgeeks.org/dsa/lazy-propagation-in-segment-tree/
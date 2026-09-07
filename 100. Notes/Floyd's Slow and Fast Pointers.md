---
Data: 2026-08-23T19:26:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Floyd's Slow and Fast Pointers

This is one of the famousness use case of the Fast and Slow pointers patterns. In this problem we need to find if there are cycles in a list of elements

![[Pasted image 20260727213015.png | 400]]

Image we have two pointer, **the slow pointer moves on step** and **the fast pointer moves two steps**. This gives us two conclusions:
1. If the **linked list doesn't have a cycle**:
	- the fast pointer will reach the end before the slow pointer 
	- The slow pointer will never be able to catch up the fast pointer it there is no cycle
2. If the **linked list has cycles**
	- The faster pointer will enter in a cycles first, followed by slow pointer
	- if at a certain point both pointer point the same **we can conclude that exists a cycle**

To be sure that the condition to detectet a cycle are always true let's a look this:
1. If the _fast pointer_ is one step behind the _slow pointer_: The _fast pointer_ moves two steps and the _slow pointer_ moves one step, and they both meet.
2. If the _fast pointer_ is two steps behind the _slow pointer_: The _fast pointer_ moves two steps and the _slow pointer_ moves one step. After the moves, the _fast pointer_ will be one step behind the _slow pointer_, which reduces this scenario to the first scenario. This means that the two pointers will meet in the next iteration.

```js
class Node {
  constructor(value, next = null) {
    this.value = value;
    this.next = next
  }
}

function hasCycle(head) {
  let slow = head
  let fast = head
  while(fast !== null && fast.next !== null) {
    fast = fast.next.next;
    slow = slow.next
    
    if(slow === fast) {
      //found the cycle
      return true
    }
  }
  return false
}
```

Based on the image above we can express this problem with some formulations:
- Distance traveled by slow pointer
$$
(m + n*x + k)
$$
- Distance traveled by fast pointer is $2 \cdot \text{ slow pointer }$
$$
(m + n*x + k) = 2 \cdot (m + n*x + k)
$$
We can also define:
- $x$ = Number of complete cyclic rounds made by fast pointer before they meet first time.
- $y$ = Number of complete cyclic rounds made by slow pointer before they meet first time.

From the above equation, we can conclude below:

### Finding the Start of the Cycle
If we know the length of the **LinkedList** cycle, we can find the start of the _cycle_ through the following steps:
1. Apply the above algorithm once the slow and fast pointers meet within the loop
2. Reset one of the pointers, for example the **slow pinter to the head**, keep the other to the current pos
3. Now move both pointer **one node at a time**
4. The **slow** pointer will cover the distance $m$ to the start of the loop
5. Also the **faster** pointer will cover the distance $m$ but inside the loop
6. Since both did the same steps, the will meet at the start of the loop
# References
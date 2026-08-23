---
Data: 2026-07-27T21:34:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Linked List Reversal

In a lot of problems, we are asked to reverse the links between a set of nodes of **linked list**. Often the constraint is that we need to do this in-place, ie without using extra memory

The basic approach is to reverse one node at time. We will start with a variable `curret` which will initially point to the head of the **Linked List** and a variable `previus` which will point to the previus node that we have processed, this will point to `null` at the first iteration.

From this the code is pretty simple:
```js
class Node {
  constructor(value, next=null) {
    this.value = value;
    this.next = next
  }
  
  printList() {
    let result = ""
    let temp = this
    while(temp !== null) {
      result += temp.value + " "
      temp = temp.next
    }
    return result
  }
}

function reverse(head) {
  let current = head
  let previous = null
  
  while(current !== null) {
    //temporarily store the next node
    next = current.next
    
    //reverse the current node
    current.next = previous
    
    //before we move to the next node, 
    //point previous to the current node
    previous = current
    
    //move on to the next node
    current = next
  }
  
  return previous
}
```

And we achieve the following results:
- The time complexity of our algorithm will be `O(N)` where `N’` is the total number of nodes in the **LinkedList**.
- We only used constant space, therefore, the space complexity of our algorithm is `O(1)`.


# References
- [Github Page with Linked List Reversal](https://github.com/Chanda-Abdul/Several-Coding-Patterns-for-Solving-Data-Structures-and-Algorithms-Problems-during-Interviews/blob/main/%E2%9C%85%20%20Pattern%2006:%20In-place%20Reversal%20of%20a%20LinkedList.md)
- https://www.geeksforgeeks.org/dsa/reverse-a-linked-list/
# Leetcode
- [x] [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- [x] [92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)
- [ ] [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
---
Data: 2026-08-23T19:23:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Next Greater or Smaller Element in Array

Usually this pattern is used when we are asking the **next greater** or the **next smaller** elements in an array. For example, given an array `arr[]` of integers, determine the **Next Greater Elements** for every element in the array, maintaining the order of appearance

**Examples**:
**Input**: `arr[] = [1, 3, 2, 4]`  
**Output**: `[3, 4, 4, -1]`  
**Explanation**: The next larger element to 1 is 3, 3 is 4, 2 is 4 and for 4, since it doesn't exist, it is -1.

**Input**: `arr[] = [6, 8, 0, 1, 3]`  
**Output**: `[8, -1, 1, 3, -1]`  
**Explanation**: The next larger element to 6 is 8, for 8 there is no larger elements hence it is -1, for 0 it is 1 , for 1 it is 3 and then for 3 there is no larger element on right and hence -1.

The idea is to use a **monotonic decreasing stack** (stack that maintains elements in decreasing order). We traverse the array from right to left. For each element, we pop elements from the stack that are smaller than or equal to it, since they cannot be the next greater element. If the stack is not empty, the top of the stack is the next greater element. Finally, we push the current element onto the stack.

![[Pasted image 20260815131406.png | 320]]   ![[Pasted image 20260815131458.png | 320]]

![[Pasted image 20260815131543.png | 320]]   ![[Pasted image 20260815131557.png | 320]]

![[Pasted image 20260815131609.png | 320]]  ![[Pasted image 20260815131627.png | 320]]

This is the code to implement a classic example for next greater:
```c++
vector<int> nextLargerElement(vector<int> &arr) {
    int n = arr.size();
    vector<int> res(n, -1);
    stack<int> stk;

    for (int i = n - 1; i >= 0; i--) {

        // Pop elements from the stack that are less
        // than or equal to the current element
        while (!stk.empty() && stk.top() <= arr[i]) {
            stk.pop();
        }

        // If the stack is not empty, the top element
        // is the next greater element
        if (!stk.empty()) {
            res[i] = stk.top();
        }

        // Push the current element onto the stack
        stk.push(arr[i]);
    }

    return res;
}
```
# References
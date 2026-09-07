---
Data: 2026-08-15T18:33:00
Tags:
  - note
  - master
  - article
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Top K Elements

These are a sets of problems that asks to find the **top/smallest/frequest K** elements among a given set. The best data structure that to keep track of K elements is an **Heap**.

Let's do a simple example. Given an array `arr[]` and a positive integer `k`, Find the top `k` elements which have the **highest frequency** in the array.

**Examples:**
> **Input:**  `[1, 23, 12, 9, 30, 2, 50], k = 3`  
> **Output**: `[50, 30, 23]`
> 
> **Input**:  `[11, 5, 12, 9, 44, 17, 2], k = 2`  
> **Output**: `[44, 17]`

The first **Naive Approach** is to sort the array and take the first `k` elements. In this case we have a time complexity of $O(n\log{n})$ that is the time complexity of the sort.

To improve this solution in $O(n\log{k})$ we can use a **Priority queue (Heap)**. The idea is:
1. as we **iterate** through the array, we keep track of the **k largest** elements at each step. To do this, we use a **min-heap**. 
2. First, we insert the **initial k** elements into the min-heap. 
3. After that, for each next element, we compare it with the **top** of the heap. 
4. Since the **top** element of the min-heap is the **smallest** among the k elements, 
5. if the current element is **larger** than the top, it means the top element is no longer one of the k largest elements. In this case, we **remove** the top and **insert** the larger element. After completing the entire traversal, the heap will contain exactly the k largest elements of the array.

```c++
vector<int> kLargest(vector<int> &arr, int k) {
  
    // Min Priority Queue (Min-Heap) with first k
    // elements of the array
    priority_queue<int, vector<int>, greater<int>>
                  minH(arr.begin(), arr.begin() + k);

    // Travers n - k elements
    for (int i = k; i < arr.size(); i++) {

      	
      	// If the top of heap is less than the arr[i]
      	// then remove top element and insert arr[i] 
      	if(minH.top() < arr[i]) {
         	minH.pop();
          	minH.push(arr[i]);
        }
    }

    vector<int> res;
  
  	// Min heap will contain only k 
  	// largest element
    while (!minH.empty()) {
        res.push_back(minH.top());
        minH.pop();
    }
  	
  	// Reverse the result array, so that all
  	// elements are in decreasing order
	reverse(res.begin(), res.end());
   	return res;
}
```

### [[Quickselect Algorithm]]
An alternative approach to archive a time complexity of $O(n)$ is to use [[Quickselect Algorithm]]. This approach is slight hard to implement and it is based on the concept to use the **partitioning step** of **QuickSort** to find the `k` largest elements in the array without sorting the entire array.


# References
- [Github Page with Top K Elements](https://github.com/Chanda-Abdul/Several-Coding-Patterns-for-Solving-Data-Structures-and-Algorithms-Problems-during-Interviews/blob/main/%E2%9C%85%20Pattern%2013:%20Top%20%27K%27%20Elements.md)
- https://www.geeksforgeeks.org/dsa/find-k-numbers-occurrences-given-array/

# Leetcode
- [x] [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [x] [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [ ] [373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)
---
Data: 2026-08-16T00:07:00
Tags:
  - note
  - master
  - "#article"
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Binary Search

This is a famous searching algorithms that could be applied in many other sub problems. It has need of two many conditions:
- The data structure must be **sorted**.
- **Access** to any element of the data structure should take **constant time**.

The classic problems where use binary search is useful to achive $O(\log{n})$ or, if you also need to order the array $O(\log{n})$, are:
- Searching in a **nearly sorted** array
- Searching in a **rotated sorted** array
- Searching in a list with **unknown length**
- Searching in an array with duplicates
- Finding the **first or last occurrence** of an element
- Finding the **square root** of a number
- Finding a **peek** elements

### Binary Search Implementation
As said above it works on a sorted or monotonic search space, and allow to achive a $O(\log{n})$ **time complexity**. T implement in there are the following steps:
1. Divide the search space into two halves by **finding the middle index "mid"**.
2. Compare the middle of the search space with the **key**.
3. If the **key** is found at middle, the process is terminated.
4. If the **key** is not found at middle, choose which half will be used as the next search space.
	- If the **key** is smaller than the middle, then the **left** side is used for next search.
	- If the **key** is larger than the middle, then the **right** side is used for next search.
5. This process is continued until the **key** is found or the total search space is exhausted.

![[Pasted image 20260816132156.png | 400]]

![[Pasted image 20260816132306.png | 400]]

![[Pasted image 20260816132316.png|400]]

![[Pasted image 20260816132331.png|400]]

Here we use a while loop to continue the process of comparing the key and splitting the search space in two halves.

```c++
int binarySearch(vector<int> &arr, int x) {
    int low = 0;
    int high = arr.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;

        // Check if x is present at mid
        if (arr[mid] == x)
            return mid;

        // If x greater, ignore left half
        if (arr[mid] < x)
            low = mid + 1;

        // If x is smaller, ignore right half
        else
            high = mid - 1;
    }

    // If we reach here, then element was not present
    return -1;
}
```

Another fancy implementation is in rust, as following:
```rust
fn binary_search<T: Ord>(arr: &[T], key: T) -> Option<usize> {
    let mut low = 0;
    let mut high = arr.len();

    while low < high {
        let middle = low + (high - low)/2;

        match key.cmp(&arr[middle]) {
            std::cmp::Ordering::Equal   => return Some(middle),
            std::cmp::Ordering::Less    => high = middle,
            std::cmp::Ordering::Greater => low = middle + 1,
        }
    }
    None
}
```

### Binary Search the Answer
Let'try to image a generic version of the binary search. We can consider a problem where all the possible answer are inside a interval $[low, high]$, and we also have a $pred$ that it used to indicate with cadidate is the right one. 

If the can not do hypothesis on the right candidate we can only try all the possibility indeed therefore $O(n)$. But if we have the **monotonic property**, we can use the binary search and achive a $O(\log(n))$.

```rust
fn binary_search_range<T, F>(low: T, high: T, pred: F) -> Option<T>
where
    T: Num + PartialOrd + FromPrimitive + Copy,
    F: Fn(T) -> bool,
{
    let mut low = low;
    let mut high = high;

    let mut ans = None;

    while low < high {
        let middle = low + (high - low) / FromPrimitive::from_u64(2).unwrap();

        match pred(middle) {
            true => {
                low = middle + T::one();
                ans = Some(middle)
            }
            false => high = middle,
        }
    }

    ans
}
```

An example for the **sqrt** problem:

```rust
fn sqrt(v: u64) -> u64 {
    binary_search_range(0, v + 1, |x| x * x <= v).unwrap()
}
```


# References
- https://www.geeksforgeeks.org/dsa/binary-search/
- https://pages.di.unipi.it/rossano/blog/2023/binarysearch/
# Leetcode
- [x] [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [x] [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- [ ] [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)

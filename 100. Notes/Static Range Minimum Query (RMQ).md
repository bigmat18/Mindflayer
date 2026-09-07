---
Data: 2026-08-30T20:18:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Static Range Minimum Query (RMQ)

We have an array `arr[0...n-1]`. We should be able to efficiently find the minimum value from index $L$ (query start) to $R$ (query end) where $0 \leq L \leq R \leq n-1$. Consider a situation when there are many range queries.

**Example:**
```
Input:  arr[]   = {7, 2, 3, 0, 5, 10, 3, 12, 18};  
        query[] = [0, 4], [4, 7], [7, 8]  
Output: Minimum of [0, 4] is 0  
        Minimum of [4, 7] is 3  
        Minimum of [7, 8] is 12
```

A **Naive Solution** is to run a loop from $L$ to $R$ and find the minimum element in the given range. This solution takes $O(n)$ time to query in the worst case.

Another approach is to use [[Segment Tree]]. With segment tree, pre-processing time is $O(n)$ and time to for range minimum query is $O(\log n)$. We also need extra space to $O(n)$. With this approach the updates are allowed in $O(\log n)$.

### Lookup table
Let' stat with a simple solution, so we create a 2D array `lookup[][]` where an entry lookup `lookup[i][j]` stores the minimum value in range `arr[i..j]`. The minimum of given range can now be calculated in $O(1)$ time.

![[Pasted image 20260830231840.png|244]]

This approach support queries in $O(1)$ but preprocessing takes $O(n^2)$ **time complexity**. Also this approach needs $O(n^2)$ extra space which may become huge for large input arrays.

### Square Root Decomposition
We can use **Spare Root Decomposition** to reduce space required in the above method.
###### Preprocessing
1. Divide the range `[0, n-1]` into different blocks of $\sqrt{n}$ each.
2. Compute the minimum of every block of size $\sqrt{n}$ and store the results.

![[Pasted image 20260830232452.png|272]]

Preprocessing takes $O(\sqrt{n} \cdot \sqrt{n}) = O(\sqrt{n})$ **time complexity** and $O(\sqrt{n})$ **space complexity**
###### Query
1. To query a range `[L, R]` we take a minimum of all blocks that lie in this range. 
2. For left and right corner blocks which may partially overlap with the given range we linearly scan the to find the minimum

The **time complexity** is $O(\sqrt{n})$. Note that we have a minimum of the middle block directly accessible and there can be at most $O(\sqrt{n})$ middle blocks. Moreover there can be at most two corder blocks that we my have to scan, so we may have to scan $2*O(\sqrt{n})$ elements of corner blocks

Below an implementation:
```c
// input data
int n;
vector<int> a (n);

// preprocessing
int len = (int) sqrt (n + .0) + 1; // size of the block and the number of blocks
vector<int> b (len);
for (int i=0; i<n; ++i)
    b[i / len] += a[i];

// answering the queries
for (;;) {
    int l, r;
  // read input data for the next query
    int sum = 0;
    for (int i=l; i<=r; )
        if (i % len == 0 && i + len - 1 <= r) {
            // if the whole block starting at i belongs to [l, r]
            sum += b[i / len];
            i += len;
        }
        else {
            sum += a[i];
            ++i;
       }
}
```

### Sparse Table
The above solution requires only $O(\sqrt{n})$ space but takes $O(\sqrt{n})$ time to query. The sparse table method supports query time $O(1)$ with extra space $O(n\sqrt{n})$

The core idea is to pre-compute a minimum of all subarrays of size $2^j$ where $j$ varies from $0$ to $\log{n}$. Like the first version with lookup table, we make a **lookup table**. Here `lookup[i][j]` contains a minimum of range starting from $i$ and size $2^j$. 

**Example**: Let's look as an example `lookup[0][3]` contains a minimum of range `[0, 7]` (starting with 0 and of size $2^3$)
###### Pre-Processing
The idea to fill this lookup table is **bottom-up** manner, using previously computed values. For example, to find a minimum of range `[0, 7]` we can use a minimum of following two:
1. Minimum of range `[0, 3]`
2. Minimum of range `[4, 7]`

Based on the above example, below is the formula
```c
// If arr[lookup[0][2]] <=  arr[lookup[4][2]],   
// then lookup[0][3] = lookup[0][2]  
if arr[lookup[i][j-1]] <= arr[lookup[i+2j-1][j-1]]  
   lookup[i][j] = lookup[i][j-1]  
   
// If arr[lookup[0][2]] >  arr[lookup[4][2]],   
// then lookup[0][3] = lookup[4][2]  
else 
   lookup[i][j] = lookup[i+2j-1][j-1]
```

![[Pasted image 20260830235159.png|236]]

###### Query
For any arbitrary range `[L, R]` we need to use ranges that are in powers of 2. The idea is to use the closest power of 2. We always need to do at most one comparison (compare a minimum of two ranges which are powers of 2).
- One range start with `L` and end with `L + closest-power-of-2`
- The other range ends with `R` and start with `R - same-closest-power-of-2+1`

**Example**: if the given range is `[2, 10]` we compare minimum of two ranges `[2, 9]` and `[3, 10]`

Based of the above example, below is the formula:
```c
// For (2,10), j = floor(Log2(10-2+1)) = 3  
j = floor(Log(R-L+1))  

// If arr[lookup[0][3]] <=  arr[lookup[3][3]],   
// then RMQ(2,10) = lookup[0][3]  
if arr[lookup[L][j]] <= arr[lookup[R-(int)pow(2,j)+1][j]]  
   RMQ(L, R) = lookup[L][j]  
   
// If arr[lookup[0][3]] >  arr[lookup[3][3]],   
// then RMQ(2,10) = lookup[3][3]  
else
   RMQ(L, R) = lookup[R-(int)pow(2,j)+1][j]
```

Since we do only one comparison, the time complexity of the query is O(1).
# References
- https://www.geeksforgeeks.org/dsa/range-minimum-query-for-static-array/
- https://cp-algorithms.com/sequences/rmq.html
- https://epubs.siam.org/doi/10.1137/090779759
---
Data: 2026-08-23T19:38:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Dynamic Prefix Sums with Fenwick Tree

The **Fenwick Tree**, also know as **Binary Indexed Tree (BIT)**, is a popular and elegant data structure that maintains the prefix sums of a dynamic array. That means that with this DS we can update values in the original array and still answer prefix sum queries. **Both operations runs in logarithimc time**

More precisely, this DS solves the following problems:

> [!NOTE]
> We have an array $A[1, n]$ of integers, and we would like to support the following operations:
> - `sum(i)` return the sum of the elements in $A[1..i]$
> - `add(i, v)` return the values $v$ in the entry $A[i]$

These queries are efficiently handles in $O(\log n)$ **time complexity** while using **linear space complexity**. It is an **implicit data structure** that means it requires only $O(1)$ additional space in addition to the space needed to sore the input data.

To describe it's behavior we are going to use the following array $A$
![[Pasted image 20260824234414.png]]

### Trivial Solutions
We can describe two trivial solutions for the problem above:
1. In the first one we store directly $A$ and it answer to `sum(i)` in $O(n)$ scannig the array and `add(i, v)` in $O(1)$
2. Instead in the second solution we store the **[[Prefix Sum]]** of A, in this way `sum(i)` is solved in $O(1)$ time and `add(i, v)` is solved in $O(n)$ modifying all entries up to position $i$

### FT Level by Level
With the **Fenwick tree** we can achieve a better tradeoffs for this problem. First of all we need to understand how to build this tree. And to do that we will procede level by level understaing the roles behind each one.
#### First Layer
Let's start considering just to solving `sum` queries only for positions that are powers of 2. In ower example we have positions 1, 2, 4, and 8. For each element **we store the prefix some of this range**

![[Pasted image 20260825000546.png]]

The figure above show how the tree has been structured. We address the queries of the simplified problems as follows:
- the `sum(i)` query is simple. We simple access node $i$, this only works for  $i$ are power of 2.
- For the `add(i, v)` query we need to add $v$ to all nodes covering ranges that include position $i$. For example for `add(3,10)` we add the value 10 to nodes 4 and 8. In general **we need to find the smallest power of 2 grater than $i$, let's call it $j$**. We will add $v$ to nodes $j, 2j, 2^2j, 2^3j, \dots$

With this first layer the `sum` takes $O(1)$ and `add` takes $O(\log n)$. Now we have to extend the solution to support `sum` queries on more positions.

#### Second Layer
To extend the capability of `sum` we have to add a second layer. First of all we want try to support rages **between consecutive powers of 2**. For example, if we need to query the range $[5, 7]$ which fall between $2^2, 2^3$ it is just a **smaller instance of our original problem**. So for the second layer we can apply the same strategy, the second layer will contain for a subarray $A[l..r]$ all the queries for any $i$ such that $i-l +1$ is a power of 2.

![[Pasted image 20260828223857.png]]

Each of elements in this layer will follow the pattern for the value that it will contain, so this value will be the **prefix sum of the elements inside the range**, NOT THE LAYER but the single range. 

Now **we can handle `sum(i)` queries also for positions that are the sum of two powers of 2**. Let' consider a position $i$ that can be expressed as $2^{k'} + 2^k$ where $k' > k$.
1. We can decompose the range $[i, k]$ into $[1, 2^{k'}]$ and $[2^{k'} + 1, 2^{k'+2^k} = 1]$
2. Both of these subranges are covered by the first layer, the first range
3. Now we can just queries these two range and sum the result

**Example**: we want evaluate `sum(5)`, we have $5 = 2^2 + 2^0$, so the subranges are $[1, 4]$ and $[5, 5]$. 
#### Third Layer
Now, with our representation we can't support positions that are neither powers of 2 nor the sum of two power of 2 yer. In our the only value is the position $7 = 2^2 + 2^1 + 2^0$ so it needs not two power of 2 but three. 

To resolve this problem we can simple add another layer to our tree with the same concept.
![[Pasted image 20260828230134.png]]

That's all from the build prospective. Let's make some **observation**:
1. We can represents our tree as an **array** of size $n+1$
2. We no longer require the original array
3. Let b $h$ equal to $\lfloor \log n +1 \rfloor$  which is the length of the binary representation of any position in the range $[1, n]$. Since any position can be expressed as the sum of at most $h$ powers of 2, **the tree has no more than $h$ level.**

If want describe a **generic role**, to determinate which node is the parent of node $k$ to build the tree we can apply these roles:
- **Role 1**: First of all we have to determinate the interval $[L, R]$ for each node, this interval determinate the prefix sum for that node.
	- `R = k`
	- `L = k - len + 1` where `len` is the greatest power of 2 that divide $k$
- **Role 2**: to determinate the parent it's enough to calculate with node has $R_{parent} = L_{children} - 1$ 

#### Answering `sum`
Let' start by discussing the `sum(i)` query. To solve this function we need to start ad node $i$ and traversing up the tree to reach node 0. This operations takes time proportional to the height of the tree, resulting in a **time complexity** of $O(\log n)$

Now, we will consider and example with $i=7$, so we will start at node 7 and:
1. move to its parent that is node 6
2. mode to its grandparent that is node 4
3. mode to its great-grandparent that is node 0
4. Summing values among the way

To navigate extracting the parent node we can use some **bit-tricks**. Let's do this without represeting the structure of the tree.

![[Pasted image 20260828231716.png | 200]]

Above there is the binary representation of all IDs involved in answering the previous query. The pattern is clear, the binary representation of the parent can be obtain **removing the triling (rightmost bit set to 1) one from the binary representation of its children**.

To do this operations in a easy way we can simply comput this bit operations `k = i & - i` thus `i-k` is the parent node. In fact, negative numbers are represented in [two’s complement](https://en.wikipedia.org/wiki/Two%27s_complement) form. In this representation, the two’s complement of a number is obtained by taking the bitwise complement of the number and then adding one to it.

![[Pasted image 20260828233103.png]]

#### Answering `add`
Now let's analyse the `add(i, v)` function. We need to add the value `v` to each node whose range include the position `i`

It might seem like we have to modify a large number of nodes, however a simple observation reveals that this number is at most $\log n$, this is because each time we mode from anode to its right sibling or to the right sibling of its parent, the size of the covered range at least doubles, and the range can not double more than $\log n$.

![[Pasted image 20260828234433.png]]

In the figure above there is an example with `add(5, _)`.  To know which are the nodes to modify for a generic `add(i, _)`  we can use another **bit-tricks**. 

For this example starting from `i=5` the next one is `6`.

![[Pasted image 20260828234636.png|200]]

![[Pasted image 20260828235306.png | 200]]

The pattern is that we need **to isolate the trailing one in 5**, which is `0001` and add to 5 to obtain 6.  This method is correct because the binary representation of a node and its sibling  matches **except for the position of the trailing one**. When we move from a node to its right sibling this triling one shifts one position to the left.

Now, consider the ID of a node that is the last child of its parent. In this case, the rightmost and second trailing one are adjacent. To obtain the right sibling of its parent, **we need to remove the trailing one and shift the second trailing one one position to the left.**

The **time complexity** for this operation still remain $O(\log n)$ as we observe that each time we move the right sibling of the current node or the right sibling of its parent, the triling on in its binray represnetion shift at leat on position to the left.

Below an implementation of **Fenwick tree in rust**

```rust
#[derive(Debug)]
pub struct FenwickTree {
    tree: Vec<i64>,
}

impl FenwickTree {
    pub fn with_len(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }

    pub fn len(&self) -> usize {
        self.tree.len() - 1
    }

    /// Indexing is 0-based, even if internally we use 1-based indexing
    pub fn add(&mut self, i: usize, delta: i64) {
        let mut i = i + 1; 
        assert!(i < self.tree.len());

        while i < self.tree.len() {
            self.tree[i] += delta;
            i = Self::next_sibling(i);
        }
    }

    /// Indexing is 0-based, even if internally we use 1-based indexing
    pub fn sum(&self, i: usize) -> i64 {
        let mut i = i + 1;  

        assert!(i < self.tree.len());
        let mut sum = 0;
        while i != 0 {
            sum += self.tree[i];
            i = Self::parent(i);
        }

        sum
    }

    pub fn range_sum(&self, l: usize, r: usize) -> i64 {
        self.sum(r) - if l == 0 { 0 } else { self.sum(l - 1) }
    }

    fn isolate_trailing_one(i: usize) -> usize {
        if i == 0 {
            0
        } else {
            1 << i.trailing_zeros()
        }
    }

    fn parent(i: usize) -> usize {
        i - Self::isolate_trailing_one(i)
    }

    fn next_sibling(i: usize) -> usize {
        i + Self::isolate_trailing_one(i)
    }
}
```

### [[Counting Inversion in an Array]]
### [[Nested Segments]]
### [[Update the Array]]
### [[DPS with Range Update]]
# References 
- https://pages.di.unipi.it/rossano/blog/2023/fenwick/
- https://en.wikipedia.org/wiki/Fenwick_tree
- https://blog.mitrichev.ch/2013/05/fenwick-tree-range-updates.html
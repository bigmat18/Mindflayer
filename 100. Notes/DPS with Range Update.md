---
Data: 2026-08-28T23:58:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# DPS with Range Update

In the problem [[Update the Array]] the range update with the `access(i)`  operation. This is easier that the problem we are going to solve here. Here we want to support a new query:
- `range_update(l, r, v)`: this updates the entries in $A[l..r]$ by adding $v$
- `sum(i)`: this return $\sum^{i}_{k=1}A[k]$

We notice that the `add` operation of the original [[Dynamic Prefix Sums with Fenwick Tree|Fenwick Tree]] is just a special case of the `range_update` operation. Moreover, as mentioned above, `access(i)` operations is also supported with two `sum` operations.

We can solve `range_update(l, r, v)` with $j -i + 1$ `add` quires but this will bring to a time complexity if $\Theta((j-i+1)\log n)$ and we want a **time complexity of $\Theta(\log{n})$** independent from the input indices.

The solution is to utilizes **Two Fenwick Trees**. Therefore, we require doubling the space usage to support the more powerful `range_update(l, r, v)` operation

### Single Fenwick Tree
First of all, let's consider a solution using a single Fenwick Tree, denoted as $FT_1$. In our initial approach, we follow a similar strategy used in the [[Update the Array]] problem:
- For a `range_update(r, l, v)`, we modify $FT_1$ by adding $v$ at position $l$ and subtracting $v$ at position $r+1$.
- When we will querying `sum(i)`, we multiply the result from $FT_1$ by $i$. This approach, however, led errors in the results

Let's consider a `range_update(l, r, v)` operation on a brand new Fenwick Tree. The correct result for a query `sum(i)` after the update are the following:
### Second Fenwick Tree


# References
- https://pages.di.unipi.it/rossano/blog/2023/fenwick/
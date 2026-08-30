---
Data: 2026-08-30T19:32:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Mo's Algorithm

The **Mo's Algorithm** is a powerful and efficient technique for solving a wide variety of [[Overlapping Intervals|range query problems]].  It becomes particularly useful for kind of queries where the use of a [[Segment Tree]] or similar data structures is not feasible. This typically occurs when the query **is non-associative**, meaning that the **result of a query on a range cannot be derived by combining the answers of the subranges that cover the original range**.

Mo's Algorithm typically achieves a **Time Complexity** of $O((n+1)\sqrt{n}$, where $n$ represents the size of the dataset, and $q$ is the number of queries.

To explain this algorithm let's consider the following problem: 
- We are given an array $A[1,n]$ of integers and our goal is to solve $q$ queries `power`.
- For a query `power(l,r)` we have to compute the power of the subarray $A[l,r]$
- For each integer $s$ within this subarray, let $K_s$ represent the number of occurrences. 
- The subarray's power is defined as the sum of products $s\cdot K_s \cdot K_s$ for every positive integer $s$ that appears in the subarray

Our goal is to achieve a **Time Complexity** of $O((x+q)\sqrt{n})$ to solve all the $q$ queries.

### A Easier Problem
For many types of ranges queries exist suitable data structure to answer queries efficiently and **online**. Solving a query **online** means that the data structure answers the query as soon as it is presented, without any delay. However, for some more complex query types, there doesn't exist such online-efficient data structures.

For certain query types, the best we can hope for is an efficient solution that works effectively only when handling a sufficiently large batch of queries. This way, the solution can process the queries in the order it deems most favorable. With such solutions, **the time complexity of an individual query is low only in an amortised sense.**

The **Mo's Algorithm** is one of these strategies: if the batch consist of $q=\Omega(n)$ queries, each query can be solved in $O(\sqrt{n})$ amortised time.

Consider now the following problem.
- We are given an array $A \left[0 , n - 1\right]$ consisting of colors, with each color represented by an integer within $\left[0 , n - 1\right]$. 
- Additionally, we are given a set of $q$ range queries called `three_or_more`. T
- he query `three_or_more(l, r)` aims to count the colors that occur at least three times within the subarray $A \left[l , r\right]$.

Let’s begin by examining a straightforward algorithm that addresses a query `three_or_more(l, r)` by scanning the subarray $A \left[l , r\right]$. The algorithm maintains an array of `counters` to track the number of occurrences of each color within the query range. Whenever a color reaches three occurrences, the `answer` is incremented by one.

Below is a **Rust implementation** of this strategy.

```rust
pub fn three_or_more_slow(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> 
{
    let mut counters: Vec<usize> = vec![0; a.len()];
    let mut answers = Vec::with_capacity(queries.len());

    for &(l, r) in queries {
        let answer = a[l..=r].iter().fold(0, |ans, &color| {
            counters[color] += 1;
            if counters[color] == 3 {
                ans + 1
            } else {
                ans
            }
        });

        answers.push(answer);

        a[l..=r].iter().for_each(|&color| counters[color] = 0);
    }

    answers
}
```

Observe that, after each query, **it’s essential to reset the vector of counters**. In the above implementation, this reset is done using the code snippet
```
a[l..=r].iter().for_each(|&color| counters[color] = 0)
```

What’s noteworthy is that this method selectively resets only the counters associated with colors within the queried subarray. This approach ensures that the time spent on resetting is proportional to the size of the queried range, rather than the length of `counters`.  Indeed, it’s evident that it has a **time complexity** of $\Theta \left(q n\right)$. 

The figure below illustrates an input that showcases the worst-case running time. We have $n$ queries. The first query range has a length of $n$ and spans the entire array. Then, the subsequent query ranges are each one unit shorter, until the last one, which has a length of one. The total length of these ranges is $\Theta \left(n^{2}\right)$, which is also the time complexity of the solution.

![425](https://pages.di.unipi.it/rossano/assets/img/mos/Mos_1.svg)

### Mo's Algorithm
Let’s now introduce a different way to implementing the inefficent algorithm above. Suppose we have just answered the query for the range $\left[l^{'} , r^{'}\right]$ and are now addressing the query for the range $\left[l , r\right]$.

Instead of starting from scratch, **we can update the previous answer and counters by adding or removing the contributions of colors that are in the new query range but not in the previous one, or vice versa**. Specifically, for the left endpoints, we must remove all the colors in $A \left[l^{'} , l - 1\right]$ if $l^{'} < l$, or we need to add all the colors in $A \left[l , l^{'} - 1\right]$ if $l < l^{'}$. The same applies to the right endpoints $r$ and $r^{'}$.

The Rust implementation below utilizes two closures, `add` and `remove`, to keep `answer` and `counters` updated as we adjust the endpoints.

```rust
pub fn three_or_more(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> {
    let mut counters: Vec<usize> = vec![0; a.len()];
    let mut answers = Vec::with_capacity(queries.len());

    let mut cur_l = 0;
    let mut cur_r = 0; // here right endpoint is excluded
    let mut answer = 0;

    for &(l, r) in queries {
        let mut add = |i| {
            counters[a[i]] += 1;
            if counters[a[i]] == 3 {
                answer += 1
            }
        };

        while cur_l > l {
            cur_l -= 1;
            add(cur_l);
        }

        while cur_r <= r {
            add(cur_r);
            cur_r += 1;
        }

        let mut remove = |i| {
            counters[a[i]] -= 1;
            if counters[a[i]] == 2 {
                answer -= 1
            }
        };

        while cur_l < l {
            remove(cur_l);
            cur_l += 1;
        }

        while cur_r > r + 1 {
            cur_r -= 1;
            remove(cur_r);
        }

        answers.push(answer);
    }

    answers
}
```

The **time complexity** of this algorithm remains $\Theta \left(q n\right)$. However, we observe that a query now executes **more quickly if its range significantly overlaps with the range of the previous query**.

This effect is perfectelly explained by the input of the previosu figure. This is input becomes a best-case for the new implementation as it takes $\Theta \left(n\right)$ time. Indeed, after spending linear time on the first query, any subsequent query is answered in constant time.

This implementation is highly sensitive to the ordering of the queries. It is enough to modify the ordering of the above queries, as shown in the figure below, to revert to quadratic time. In the example below, we rearrange the queries to alternate between a long and a short query. With this ordering, the new implementation takes $\Theta \left(n^{2}\right)$ time.

![427](https://pages.di.unipi.it/rossano/assets/img/mos/Mos_2.svg)

These considerations lead to a question: 
> *if we have a sufficient number of queries, can we rearrange them in a way that exploits the overlap between successive queries to gain an asymptotic advantage in the overall running time*?

Mo’s algorithm answers positively this question by providing a **reordering of the queries such that the time complexity reduces to $\Theta \left(\left(q + n\right) \sqrt{n}\right)$.**

The idea is to conceptually partition the array **$A$ into $\sqrt{n}$ buckets**, each with a size of $\sqrt{n}$, named $B_{1} , B_{2} , \ldots , B_{\sqrt{n}}$. A query belongs to bucket $B_{k}$ if and only if its left endpoint $l$ falls into the $k$ -th bucket, which can be expressed as $\lfloor l / \sqrt{n} \rfloor = k$.
1. we group the queries based on their corresponding buckets, 
2. and within each bucket, the queries are solved in ascending order of their right endpoints.

The figure shows this bucketing approach and the queries of one bucket sorted by their right endpoints.

![510](https://pages.di.unipi.it/rossano/assets/img/mos/Mos_3.svg)

Now, let’s analyze the time complexity of the algorithm with this query reordering. It’s sufficient to count the number of times we move the indexes `cur_l` and `cur_r`. This is because both `add` and `remove` take constant time, and, thus, the **time complexity is proportional to the overall number of moves of these two indexes.**

Let’s concentrate on a specific bucket:
- As we process the queries in ascending order of their right endpoints, the index `cur_r` moves a total of at most $n$ times. 
- On the other hand, the index `cur_l` can both increase and decrease but, it is constrained within the bucket, and, thus, it cannot move more than $\sqrt{n}$ times per query. 

Thus, for a bucket with $b$ queries, the overall time to process its queries is $\Theta \left(b \sqrt{n} + n\right)$.

Summing up over all buckets, the time complexity is $\Theta \left(q \sqrt{n} + n \sqrt{n}\right)$, which results in $\Theta \left(\sqrt{n}\right)$ amortized time per query when $m = \Omega \left(n\right)$.

Here’s a Rust implementation of the reordering process. We have to compute a `permutation` to keep track of how the queries have been reordered. This permutation is essential for returning the answers to their original ordering.

```rust
pub fn mos(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> {
    // Sort the queries by bucket and get the permutation induced
    // by this sorting.
    // The latter is needed to permute the answers back 
    // to the original ordering
    let mut sorted_queries: Vec<_> = queries.iter().cloned().collect();
    let mut permutation: Vec<usize> = (0..queries.len()).collect();

    let sqrt_n = (a.len() as f64) as usize + 1;
    sorted_queries.sort_by_key(|&(l, r)| (l / sqrt_n, r));
    permutation.sort_by_key(|&i| (queries[i].0 / sqrt_n, queries[i].1));

    let answers = three_or_more(a, &sorted_queries);

    let mut permuted_answers = vec![0; answers.len()];
    for (i, answer) in permutation.into_iter().zip(answers) {
        permuted_answers[i] = answer;
    }

    permuted_answers
}
```

# References
- https://pages.di.unipi.it/rossano/blog/2023/mosalgorithm/
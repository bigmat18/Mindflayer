---
Data: 2026-08-23T19:32:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Sweep Line Algorithm
This is a type of algorithm to solve problems of [[Overlapping Intervals]] pattern.
### Maximum Number of Overlapping intervals
For example let's start to describe this paradigm using a problem:
- We are given a set of `n` intervals `[s_i, e_i]` on a line
- We say that two intervals `[s_i, e_i]` and `[s_j, e_j]` overlaps if and only if their intersection is not empty, i.e., if there exist at least a point belonging to both intervals
- The goal is to compute the maximum number of overlapping intervals.

![[Pasted image 20260823195458.png]]

If we consider the image above, we have a set of 10 intervals. The maximum number of overlapping intervals is 5.

The **sweep line algorithm** employs an imaginary **vertical line** sweeping over the x-axies. As it progresses, we maintain a running solution to the problem at hand.

The solution is updated when the vertial line reaches certain key point where some event happen. The type of event tells us how to update the current solution.

- We will mode the sweep line **from left to right** and **stop at the beginning or the end of the intervals**. 
- We can see the two importants events: **new interval start**, **new interval end**
- We also maintain a counter which keeps track of the number of intervals that are currently intersecting the sweep line, along with the maximum value reached by the counter so far
- For each point, we first add to the counter the number of intervals that begin at that point, 
- and then we substract the number of intervals that end at that point

![[Pasted image 20260823195722.png]]

Note that the sweep line **touches only points on the x-axis where an event occurs**. For example, points and are not taken into consideration. This is important because the number of considered points, and thus the time complexity, **is proportional to the number of intervals and not to the size of the x-axis**.

Here is a rust implementation:
```rust
#[derive(PartialOrd, Ord, PartialEq, Eq, Debug)]
enum Event {
    Begin,
    End,
}

pub fn max_overlapping(intervals: &[(usize, usize)]) -> usize {
    let mut pairs: Vec<_> = intervals
        .iter()
        .flat_map(|&(b, e)| [(b, PointKind::Begin), (e, PointKind::End)])
        .collect();

    pairs.sort_unstable();

    pairs
        .into_iter()
        .scan(0, |counter, (_, kind)| {
            if kind == Event::Begin {
                *counter += 1;
            } else {
                *counter -= 1;
            }
            Some(*counter)
        })
        .max()
        .unwrap()
}
```

### Closest Pair of Points
Let's look a second problem to apply **sweep line paradigm** to a two-dimensional problem.
- We are given a set of $n$ point in the plane
- The gol is to find the closest pair of points in the set. The distance between two points $(x_1, y_1)$ and $(x_2, y_2)$ is the **Euclidean distance** $d((x_1, y_1),(x_2, y_2)) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$

A **brute-force** approach is to calculate the distances between all possible pairs of points, but this result in a **time complexity** of $O(n^2)$

![[Pasted image 20260823203056.png]]

A better approach use the **sweep line paradigm**. Now below the steps:
###### Initialisation
- We start by sorting the array of points in increasing order of their **$x$-coordinates**. If two points share the same $x$-coordinate, they are sorted by their $y$-coordinate.
- We maintain the shortest distance found so far, denoted as $\delta$. Initially, $\delta$ is set to $\infty$ (or the distance between the first two points).
- We initialize an active set (a `std::set` in C++ or a `BTreeSet` in Rust) to store the candidate points currently inside the bounding strip. **Crucial detail:** the set stores and orders the points primarily by their **$y$-coordinates** to allow efficient vertical range queries.
###### Iterations
We iterate over the points $P[i] = (x_i, y_i)$ from left to right, ordered by $x$:
1. We remove from the active set all points that respect these two condition:
	- outside the horizontal range $[x_i - \delta, x_i]$ 
	- outside the vertical range $[y-\delta, y+\delta]$

2. Using the active set, we perform a range query to inspect only the points $q = (x_q, y_q)$ whose $y$-coordinates fall within the vertical window:
$$[y_i - \delta, \; y_i + \delta]$$
4. For each candidate point $q$ found in the vertical range:
    - Compute the Euclidean distance $d(P[i], q)$.        
    - If $d(P[i], q) < \delta$, update the shortest distance $\delta \leftarrow d(P[i], q)$ and record $(P[i], q)$ as the closest pair.
5. Insert $P[i]$ into the active set, then proceed to the next point $P[i+1]$.

To understand why, consider the squares in the figure above. Each of these squares, including its perimeter, **can contain at most one point**.  Assume, for the sake of contradiction, that a square contains two points, denoted as $q$ and $q'$. The distance between $q$ and $q'$ is smaller than $\delta$. If point exists, **it would have already been processed by the sweep line because it has an x-coordinate smaller than that of $p$.** However, this is not possible, because otherwise the value of would be smaller than its current value.

The following is a Rust implementation of this algorithm.

```rust
pub fn distance_squared(p: (i64, i64), q: (i64, i64)) -> i64 {
    (p.0 - q.0).pow(2) + (p.1 - q.1).pow(2)
}

use std::collections::BTreeSet;
use std::ops::Bound::Included;

// Returns the (squared) Euclidean distance between the closest pair of 
// points in `points`
pub fn closest_pair(points: &mut [(i64, i64)]) -> Option<i64> {
    if points.len() < 2 {
        return None;
    }

    points.sort_unstable_by_key(|p| (p.1, p.0)); // sort by y

    let min_y = points[0].1;
    let max_y = points.last()?.1;

    let mut delta = distance_squared(points[0], points[1]);

    let mut set: BTreeSet<(i64, i64)> = BTreeSet::new();
    for &point in points.iter() {
        // Search by x and select the points with too small y-coordinate that we remove
        // to not touch them again in the future
        let to_delete: Vec<_> = set
            .range((
                Included(&(point.0 - delta, min_y)),
                Included(&(point.0 + delta, max_y)),
            ))
            .filter(|p| p.1 < point.1 - delta)
            .cloned()
            .collect();

        // Remove those points
        for p in to_delete {
            set.remove(&p);
        }

        // Search again and compute the distances with survived points.
        // Update delta if needed.
        delta = set
            .range((
                Included(&(point.0 - delta, min_y)),
                Included(&(point.0 + delta, max_y)),
            ))
            .fold(delta, |acc, &p| acc.min(distance_squared(point, p)));

        set.insert(point);
    }

    Some(delta)
}
```

There are two differences from the above description. First, we compute the squared Euclidean distance. This way, we avoid the computation of the square root, which is slow and results in a floating-point value The second difference is that **we swap the roles of x and y**. Therefore, we process the points by ascending y-coordinate and use a horizontal sweep line.

With this implementation we achieve a **time complexity** of $O(n\log{n})$.
# References
- https://pages.di.unipi.it/rossano/blog/2023/sweepline/
---
Data: 2026-08-28T23:58:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Nested Segments

First of all let's go to describe the problem. 
- **Input:** We are given segments: $[l_1. r_1], [l_2, r_2], \dots, [l_n, r_n]$ on a line. There are no coinciding endpoints among the segments.
- **Output**: We have to determine and report the number of other segments each segment contains (**fully contain**). That is the same  thing as saying, for each segment $i$ we want to count the number of segments $j$ such that the following conditions hold: $l_i < l$ and $r_i < r$

To solve this problem we can use a combination between [[Sweep Line Algorithm]] and [[Dynamic Prefix Sums with Fenwick Tree]].
1. We initialise a **Fenwick tree** with a size of 2n.
2. We build the **Fenwick tree** by adding all right values adding 1 in each position that correspond to right endpoint of a segment
3. Then we let a **Sweep Line process** the segments in **increasing order** to their left endpoints. To do this we need first to order the array of segment by left value.
4. When we process the segment $[l_i, r_i]$ we compute `sum(r_i - 1)` as the result for the current segment
5. Before to moving to the next segments -1 at position $r$ to remove the contribution of the right endpoint of the current segment

The claim is that `sum(r_i - 1)` **is the number of segments contained in $[l_i, r_i]$** because all the segments that start before $l_i$ have already been processed, and their right endpoints have been removed from the Fenwick Tree. Therefore `sum(r_i - 1)` is the number of segments that start after $l_i$ and end before $r_i$

Below the final implementation using the Fenwick tree implemented in [[Dynamic Prefix Sums with Fenwick Tree]]
```rust
use std::io::{self, Read};

#[derive(Clone, Copy, Debug)]
struct Segment {
	// this is used to map segments in the ordered list per left with the original one
    original_id: usize, 
    l: i64,
    r: i64,
    r_rank: usize, // This is used to map r to sorted array per fenwick tree
}

fn main() {
    let mut in_buffer = String::new();
    io::stdin().read_to_string(&mut in_buffer).unwrap();

    let mut values = in_buffer.split_whitespace();
    let n: usize = match values.next() {
        Some(val) => val.parse().unwrap(),
        None => return,
    };

	// Read value from inputs
    let mut segments: Vec<Segment> = Vec::with_capacity(n);
    for original_id in 0..n {
        let l: i64 = values.next().unwrap().parse().unwrap();
        let r: i64 = values.next().unwrap().parse().unwrap();
        segments.push(Segment {
            original_id,
            l,
            r,
            r_rank: 0,
        });
    }

    // Now we map each r value for each segment with 
    // the position in an ordered array
    let mut mapping: Vec<usize> = (0..n).collect();
    mapping.sort_unstable_by_key(|&idx| segments[idx].r);

    let mut acc = 0;
    for (i, &idx) in mapping.iter().enumerate() {
	    // this is used to handle segments with same r value, 
	    // we will asign both same acc
        if i > 0 && segments[idx].r != segments[mapping[i - 1]].r {
            acc += 1; 
        }
        segments[idx].r_rank = acc;
    }

    // Ordering the segments for l value
    segments.sort_unstable_by_key(|seg| seg.l);

    // Init fenwick tree
    let max_rank = if n > 0 { acc + 1 } else { 0 };
    let mut ft = FenwickTree::with_len(max_rank);

    for seg in &segments {
        ft.add(seg.r_rank, 1);
    }

    // Application of sweep algorithm from left to right with the tree
    let mut results = vec![0; n];
    for seg in &segments {
        let r = seg.r_rank;
        if r > 0 {
            results[seg.original_id] = ft.sum(r - 1);
        } else {
            results[seg.original_id] = 0;
        }
        ft.add(r, -1);
    }

    for res in results {
        println!("{}", res);
    }
}
```

### Using Segments Tree
A more natural way to solve nested segments is to use the [[Segment Tree]]. This algorithm follow the same initial idea of the previous one but using the different data structure.
1. First of all we collect all the left and right values, removing the duplicated values and we order them in ascending order 
2. During this operation we keep track of the mapping so in the segments array we have 
```
// the original value - the value in the compressed array
left - left_map 
right - right_map
```
3. The we order the new segments array (with the values compressed) in decreasing order for left and, in case of equals values, right in ascending order
4. We init the segment tree as an array of frequencies (we use the [[Segment Tree#Rust Implementation|this]] implementation)
5. We iterate over the ordered segments array, we calculate the sum for the interval and we update the value
6. Return the updated values

```rust
pub fn solve_nested_segments(raw_segments: &[(usize, usize)]) -> Vec<usize> {
    let n = raw_segments.len();
    if n == 0 {
        return Vec::new();
    }

    // Step 1: Collect all distinct endpoints for Coordinate Compression
    let mut coords = Vec::with_capacity(2 * n);
    for &(l, r) in raw_segments {
        coords.push(l);
        coords.push(r);
    }
    coords.sort_unstable();
    coords.dedup();

    let get_compressed = |val: usize| -> usize {
        coords.binary_search(&val).unwrap() + 1 // 1-based index
    };

    // Internal representation with original indices preserved
    #[derive(Clone, Copy)]
    struct InputSegment {
        id: usize,
        l: usize,
        r: usize,
    }

    let mut segments: Vec<InputSegment> = raw_segments
        .iter()
        .enumerate()
        .map(|(id, &(l, r))| InputSegment {
            id,
            l: get_compressed(l),
            r: get_compressed(r),
        })
        .collect();

    // Step 2: Sort segments:
    //   - Primary: `l` in descending order (largest `l` first)
    //   - Secondary: `r` in ascending order (smallest `r` first)
    segments.sort_by(|a, b| {
        if a.l != b.l {
            b.l.cmp(&a.l)
        } else {
            a.r.cmp(&b.r)
        }
    });

    // Step 3: Use the SegmentTree to count how many segments 
    // have r_j <= r_curr
    let max_coord = coords.len() + 1;
    let mut tree = SegmentTree::new(max_coord);
    let mut ans = vec![0; n];

    for seg in segments {
        // Query point `seg.r`: retrieves all segments previously 
        // inserted with range [seg.r, max_coord]
        // That is equivalent to finding how many inserted endpoints satisfy
        // r_inserted <= seg.r
        let contained_segments = tree.query(seg.r);
        ans[seg.id] = contained_segments.len();

        // Register the current segment's right boundary 
        // across [seg.r, max_coord]
        tree.add_segment(seg.r, max_coord);
    }

    ans
}
```
# References
- https://codeforces.com/problemset/problem/652/D?locale=en
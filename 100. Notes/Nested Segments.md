---
Data: 2026-08-28T23:58:00
Tags:
  - note
  - master
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
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
5. Before to moving to the next segments -1 at position $r$ to remove the contribution of the right endpoint of the current segment. **Cosa è rimasto dentro il Fenwick Tree in questo preciso istante?**
	- I segmenti già esaminati in precedenza sono già stati tolti.
	- Il segmento corrente $i$ è appena stato tolto.
	- **Dentro l'albero ci sono SOLTANTO i segmenti futuri!**

The claim is that `sum(r_i - 1)` **is the number of segments contained in $[l_i, r_i]$** because all the segments that start before $l_i$ have already been processed, and their right endpoints have been removed from the Fenwick Tree. Therefore `sum(r_i - 1)` is the number of segments that start after $l_i$ and end before $r_i$

Below the final implementation using the Fenwick tree implemented in [[Dynamic Prefix Sums with Fenwick Tree]]
```rust
// assumendo che i valori r, l sono al massimo n (lunghezza segments vector)
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub l: i32,
    pub r: i32,
    pub idx: usize,
}

pub fn count_contained_segments(segments: &[Segment]) -> Vec<usize> {
    let n = segments.len();
    if n == 0 {
        return vec![];
    }

    // 1. Troviamo il massimo valore di r per dimensionare l'albero 0-based
    let max_r = segments.iter().map(|s| s.r).max().unwrap().max(0) as usize;
    let mut ft = FenwickTree::with_len(max_r + 1);

    // 2. Popoliamo l'albero inserendo tutti gli estremi destri (r)
    for seg in segments {
        ft.add(seg.r as usize, 1);
    }

    // 3. Creiamo una copia da ordinare per la sweep-line (l crescente)
    let mut sorted_segs = segments.to_vec();
    sorted_segs.sort_unstable_by_key(|s| s.l);

    let mut result = vec![0; n];

    // 4. Sweep-line: elaboriamo i segmenti in ordine di l crescente
    for seg in sorted_segs {
        let r_idx = seg.r as usize;

        // Rimuoviamo il contributo del segmento corrente
        ft.add(r_idx, -1);

        // Contiamo i segmenti con estremo destro <= r - 1
        let count = if seg.r > 0 {
            ft.sum(r_idx - 1) as usize
        } else {
            0
        };

        // Salviamo il conteggio nella posizione originale indicata da idx
        result[seg.idx] = count;
    }

    result
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
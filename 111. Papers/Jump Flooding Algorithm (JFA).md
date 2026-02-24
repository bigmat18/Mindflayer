**Data time:** 15:01 - 30-06-2025

**Status**: #note #youngling #paper

**Tags:** [[CSG on Mesh using Voxelization and SDF]]

**Area**: [[Master's degree]]
# Jump Flooding Algorithm (JFA)

This algorithm born to compute [[Voronoi Decoposition|Voronoi Diagram]] for a 2D grid of size $n \times n$. The idea is: starting from a set of seeds at some grid point and propagate content to each grid point, to achive this we flood the content in creasing distance from s outward. A grid point $(x, y)$ propagates its information (e.g., closest seed ID and its position) to its immediate neighbors $(x+i, y+j)$ where $i, j \in \{-1, 0, 1\}$. Propagating this across an $n \times n$ grid typically requires $O(n)$ rounds, which is inefficient for large grids.

![[Pasted image 20250703172618.png]]

### JFA Algorithm
Assume an $n \times n$ grid, where $n$ is a power of 2 (e.g., 64x64, 128x128). We can expand this concept to a 3D grid without any problem. 
#### 1. Initialization Phase
*   Each grid point designated as a seed $ s $ at position $(x, y)$ stores `(seed_id: s, position: (x, y))` This signifies the closest seed found so far.
*   All other grid points (non-seeds) store `(seed_id: null, position: null)`.
```python
# Pseudocode: Initialization
grid_data = initialize_grid(n) # n x n grid
for seed_pos in seed_positions:
    seed_id = get_seed_id(seed_pos)
    grid_data[seed_pos] = (seed_id, seed_pos)
```
#### 2. Flood Rounds Phase (Iterative)
The algorithm performs $\log_2 n$ rounds. The step lengths $k$ *decrease* across rounds: starting from $n/2$ and halving each time, down to 1. For a 64x64 grid, steps would be: 32, 16, 8, 4, 2, 1.

* **During each round (with step $ k $):**
    * **Broadcasting:** Each grid point $(x, y)$ broadcasts its current information `(current_seed, current_pos)` to all points $(x', y')$ that are $ k $ steps away (i.e., $x' \in \{x-k, x, x+k\}$ and $y' \in \{y-k, y, y+k\}$, excluding $(x,y)$ itself).
    * **Receiving and Updating:** Each grid point $(x', y')$ collects all information broadcast from its $ k $-distance neighbors. It then compares this received information with its own `current_seed`.
    * If a received seed `(received_seed, received_pos)` is closer to $(x', y')$ than `current_seed` (usually determined by Euclidean distance), the grid point $(x', y')$ updates its stored information to `(received_seed, received_pos)`.
    * To ensure that updates within the same round do not affect subsequent calculations in that same round, a common implementation uses a double buffering technique: a `current_grid` and a `next_grid`. Updates are written to `next_grid` based on `current_grid`, and then `next_grid` becomes `current_grid` for the next round.

```python
# Pseudocode: Flood Round (simplified conceptual view)
for k in [n/2, n/4, ..., 1]:
    # Using double buffering: read from 'current_grid', write to 'next_grid'
    next_grid_data = copy(grid_data) # Prepare for the next state

    for x_src from 0 to n-1:
        for y_src from 0 to n-1:
            src_seed, src_pos = grid_data[(x_src, y_src)]

            if src_seed is not None: # Only propagate if it has a valid seed
                # Broadcast to neighbors at distance k
                for dx in [-k, 0, k]:
                    for dy in [-k, 0, k]:
                        if dx == 0 and dy == 0: continue # Skip self

                        x_dest, y_dest = x_src + dx, y_src + dy

                        # Check bounds
                        if 0 <= x_dest < n and 0 <= y_dest < n:
                            dest_seed, dest_pos = next_grid_data[(x_dest, y_dest)]

                            # Calculate distance from destination to the source's seed
                            dist_to_src_seed = calculate_distance((x_dest, y_dest), src_pos)

                            # Update if this seed is closer than the current one at destination
                            if dest_seed is None or 
                               dist_to_src_seed < calculate_distance((x_dest, y_dest), dest_pos):
                                next_grid_data[(x_dest, y_dest)] = (src_seed, src_pos)

    grid_data = next_grid_data # Update the grid for the next round
```

## Advantages

* **Path Dependence:** For a grid point $p$ to correctly associate with seed $s_0$, there must exist a path of information propagation from $s_0$ through a sequence of grid points $s_1, s_2, \dots, s_k$, where each $s_i$ considers $s_0$ its closest seed at the time it receives it.
* **Error Minimization:** JFA excels because it allows grid points to temporarily accept multiple seeds as their "closest." This means points don't prematurely finalize their association. This flexibility is crucial.
    * **Contrast with Increasing Steps:** An algorithm that starts with step 1 and doubles ($1, 2, 4, \dots, n/2$) often fails. Pixels near seeds will quickly lock onto their correct seed in early, small-step rounds. Later, when steps are large, these "locked" pixels may not propagate information effectively, hindering optimal paths for distant seeds.
    * **JFA's Advantage:** By using *decreasing* steps ($n/2, n/4, \dots, 1$), JFA gives distant seeds a better chance to reach their targets. The smaller steps in later rounds allow for finer adjustments and corrections, reducing the likelihood of incorrect associations.
      
### Disadvantages
* **No Guarantee of Absolute Correctness:** The primary disadvantage is that JFA does *not always* compute the perfectly correct Voronoi diagram. Cases can occur where certain grid points record a seed that is not actually the closest. This happens when the necessary information propagation path from the truly closest seed is interrupted. **Cause:** This issue arises if the intermediate points along the path required to reach a pixel $p$ from its correct seed $r$ have already "finalized" another closer seed (e.g., $g$ or $b$) in earlier rounds. If these intermediate points no longer record $r$ as their closest seed, they cannot propagate $r$'s information towards $p$.

![[Pasted image 20250703172555.png]]

* **Special Errors and Metric Dependence:** Despite its effectiveness, JFA can make errors. However, with the Euclidean distance metric, these errors are described as "very few" and "only very special.". This implies that errors tend to occur in very specific seed configurations or certain regions of the grid, rather than being widespread.

# References
- [Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform](https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf)
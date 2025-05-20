**Data time:** 01:57 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Combining-Tree Barrier

The problem with [[Centralized Barrier]] is that due to all the entities repeatedly accessing the global variable for pass/stop, the communication traffic is rather high.

This problem can be resolved by a **multi-level barrier**. The **Idea** is a hierarchical way of implementing barrier to resolve the scalability by avoiding the case that all entities are spinning at the same location.

![[Pasted image 20250520020332.png | 350]]

- Each node of the tree is in a dedicated cache line
- **Node**: local `counter` (no of children k>1) `sense` boolean, pointer to the `parent` node (`null` for the root)
- Entities are divided into groups, each assigned to a leaf of the tree
- Each entity updates the state of its leaf.
- The last one to arrive continues up to the tree by updating the state of its parent

Assume for simplicity that the entities synchronizing in the barrier are $N = k^{h}$ where N is the number of involved entities, $k$ is the ariety of the tree, and $h$ is the height of the tree.

We can see the pseudo-code below. We use `node` struct and two primitives: `barrier_wait` and `barrier_wait_node`

![[Pasted image 20250520020839.png | 450]]

- The code assumes the tree has been already built and the struct **barrier** only contains the pointers to the leaves of the tree
- Depth of the barrier is $h$
- Each barrier notifies exactly $k$ children
- The critical path is $O(\log_kN)$
- Other solutions (not analyzed here) like dissemination, tournament, MCS barriers and so forth.
# References
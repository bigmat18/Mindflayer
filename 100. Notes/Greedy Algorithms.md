---
Data: 2026-08-16T20:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Greedy Algorithms

There are many problems where the solution is to "just be greedy". At it's core, a greedy algorithms builds up a solution step by step, always **choosing the best available option at each moment**. The hope is that, these locally optimal decisions will eventually lead to a globally optimal solution.

Sometimes this assumption is true, sometimes not, but it's imoprtant to keep in mind that greedy is not about speed. It's about the "mindset": **grab what looks best right now, and trust the structure of the problem to carry you the rest of the way.**

Greedy doesn't have a universal template but it's more like a philosophy. Some common greedy patterns are the following:
1. **Interval Scheduling**: Sorting intervals by end/start and picking non-overlapping ones.
2. **Fractional Knapsack**: Pick items based on best value/weight ratio.
3. **Job Sequencing / Deadline Scheduling**: Pick jobs by deadlines or profits.
4. **Frequency Merging / Huffman-Type Greedy**: Repeatedly combining items with smallest cost using a heap.
5. **Coin Change (Greedy Variant)**: Works only for canonical coin systems.
6. **Sorting + Greedy Decision**: Optimize based on prefix/suffix choices.
7. And more, read this [link](https://leetcode.com/discuss/post/7344979/15-core-greedy-patterns-for-coding-inter-a1wp/)

### [[Kadane's Algorithm]]

# References
- https://medium.com/@hanxuyang0826/mastering-the-greedy-algorithm-from-leetcode-puzzles-to-infrastructure-50c586a6518f
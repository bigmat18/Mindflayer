---
Data: 2026-08-16T14:46:00
Tags:
  - note
  - master
  - article
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
Area: "[[Master's degree]]"
---
# Backtracking

**Backtracking** is a problem-solving algorithmic technique that involves finding a solution incrementally by trying **different options** and **undoing** them if they lead to a **dead end**.
- Backtracking is used to explore multiple possibilities in problems such as finding a path in a maze, or solving puzzles like sudoku, by systematically trying different choices
- When a choice leads to a dead end, the algorithm backtracks to the previous decision point and tries a different path, avoiding unnecessary explorations

![[Pasted image 20260816203422.png]]

In general in work building a solution step by step and uno whenever we hit dead end. The steps are:
1. **Choose**: start by making a choice that could lead toward a solution
2. **Explore**: recursively mode forward with this choice
3. **Check validity**: if the choice leads to an invalid state, **undo** it and try another option
4. **Repeat**: continue this process until all possibilities are explored ora a valid solution is found

### How to use
##### Use Backtracking
- **Constraint satisfaction problems:** When you need to build a solution step by step and must satisfy certain conditions
- **Search problems**: When the solution space is large, but invalid or infeasible branches can be pruned early.
- **Multiple solutions**: When you need to explore all possible valid solutions, not just one.
- **Combinatorial problems**: When you must generate all valid combinations, permutations, or subsets under constraints.

##### Not Use Backtracking
- **[[Greedy Algorithms|Greedy]] or [[Dynamic Programming|DP]] fits better**: If the problem can be solved directly using a greedy strategy or dynamic programming, backtracking is overkill.
- **No pruning possible:** If all branches must be explored anyway (no constraints to cut early), brute force or iterative methods may be simpler.
- **Large input size**: Backtracking can be exponential in time. For very large inputs without strong pruning opportunities, it becomes impractical.
- **Single optimal solution**: If the task only needs one best solution with clear optimization criteria, algorithms like DP, greedy, or graph search may be faster.


# References
- https://www.geeksforgeeks.org/dsa/backtracking-algorithms/
- https://www.geeksforgeeks.org/dsa/introduction-to-backtracking-2/
# Leetcode
- [ ] [46. Permutations](https://leetcode.com/problems/permutations/)
- [ ] [78. Subsets](https://leetcode.com/problems/subsets/)
- [ ] [51. N-Queens](https://leetcode.com/problems/n-queens/)
---
Data: 2026-08-16T21:01:00
Tags:
  - note
  - master
  - article
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Algorithms Patterns]]"
  - "[[Dynamic Programming]]"
Area: "[[Master's degree]]"
---
# Dynamic Programming Pattern

This is a very powerful technique used to solving optimisation problem by breaking them down into smaller sub-problems and storing ther solutions to avoid repetitive work. It has the following properties:
- It is mainly an **optimisation over plain recursion**. Wherever we see a recursive solution that has repeated calls for the same inputs, we can optimize it using Dynamic Programming
- The idea is to simply **store the results of subproblems so that we do not have to re-compute them when needed later**. This simple optimization typically reduces time complexities from exponential to polynomial.

![[Pasted image 20260816211813.png | 400]]

![[Pasted image 20260816211826.png|400]]

### [[Fibonacci Sequence]]
### [[Longest Common Subsequence (LCS)]]
### [[Conf & Perm with Backtracking]]
### [[Binomial Coefficient]]
### [[Rod Cutting Problem]]
### [[Edit Distance (Distanza di Levenshtein)]]
### [[Zaino 0-1 (0-1 Knapsack Problem)]]
### [[Longest Palindromic Substring]]


# References
- https://www.geeksforgeeks.org/dsa/dynamic-programming/
# Leetcode
- [ ] [70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
- [ ] [322. Coin Change](https://leetcode.com/problems/coin-change/)
- [ ] [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [ ] [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
- [ ] [416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
- [ ] [312. Burst Balloons](https://leetcode.com/problems/burst-balloons/)
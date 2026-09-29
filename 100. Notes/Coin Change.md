---
Data: 2026-09-29T19:12:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
  - "[[Dynamic Programming]]"
Area: "[[Master's Degree.base]]"
---
# Coin Change

We have $n$ types of coins available in infinite quantities where the value of each coin is given in the array $C=[c_1, c_2, \dots, c_n]$. The goal is find out how many ways we can make the change of the amount $K$ (sum) using the coins given.

**Example 1**:
- **Input**: `sum = 4, coins[] = [1, 2, 3]`  
- **Output:** `4`  
- **Explanation:** There are four solutions: `[1, 1, 1, 1], [1, 1, 2], [2, 2]` and `[1, 3]`

**Example 2**:
- **Input:** `sum = 10, coins[] = [2, 5, 3, 6]`  
- **Output:** `5`  
- **Explanation:** Five solutions:  `[2, 2, 2, 2, 2], [2, 2, 3, 3], [2, 2, 6], [2, 3, 5]` and `[5, 5]`

The solution is similar to [[Zaino 0-1 (0-1 Knapsack Problem)]] and [[Subset Sum]] problems. The goal is to build a $(n+1) \times (K+1)$ matrix $W$. This will be use to compute the number of ways to change any amount smaller that or equal to $K$ by using coins in any prefix of $C$.

1. The easy cases are when $K=0$. We have just one way to change this amount. Thus, the first column of $W$ contains $1$ but $W[0,0] = 0$, which is the number of ways to change $0$ no coin.
2. Now for every coin we have an option to include it in the solution or exclude it
	- if we decide to include the i-th coin, we reduce the amount by coin value and use the sub-problem solution ($K-c[i]$)
	- If we decide to exclude the i-th coin, the solution for the same amount without considering that coin is on the entry above.

We can describe this formulation by this formula:
$$
DP[i][j] = 
\begin{cases}
0 & i = 0 \text{ and } j \neq 0\\
0 & i \neq 0 \text{ and } j = 0\\
1 & i = 0 \text{ and } j = 0\\
DP[i-1][j] + DP[i][j - coins[i-1]] & otherwise
\end{cases}
$$

The code solution:
```c++
#include <iostream>
using namespace std;

int count(vector<int>& coins, int sum) {
    int n = coins.size();

    vector<vector<int> > dp(n + 1, vector<int>(sum + 1, 0));

    dp[0][0] = 1;
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= sum; j++) {

            // Add the number of ways to make change without
            // using the current coin,
            dp[i][j] += dp[i - 1][j];

            if ((j - coins[i - 1]) >= 0) {

                // Add the number of ways to make change
                // using the current coin
                dp[i][j] += dp[i][j - coins[i - 1]];
            }
        }
    }
    return dp[n][sum];
}
```

# References
- https://www.geeksforgeeks.org/dsa/coin-change-dp-7/
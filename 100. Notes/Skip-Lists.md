---
Data: 2026-10-29T11:56:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
Area:
  - "[[Master's Degree.base]]"
---
# Skip-Lists

This type of data structure have many of the usual desirable properties of balanced binary search trees, but their structure is very different. At a high level, a skip list is just a sorted linked list with some random shortcut.

TO do a search in a normal singly-linked list of length $n$ we obviously need to took $n$ i terms in the worst case, so the **time complexity** will be $O(n)$. 

The idea of the skip-list is to reduce this complexity adding a second-level list that contains roughly half the items from the original list, specifically, for each item in the original list, we duplicate it with a probabiliy of 1/2

![[Pasted image 20261029124909.png]]

We can find a value in this structure using a two stage algorithm:
1. we scan for x in the shortcut list, starting at the −∞ sentinel node
	- If we find x, we’re done
	- otherwise, we reach some value bigger than x and we know that x is not in the shortcut list. 
2. Let $w$ be the largest item less than $x$ in the shortcut list, in this second phase we scan for x in the original list, starting from w. Again, if we reach a value bigger than x, we know that x is not in the data structure.

![[Pasted image 20261029125104.png|474]]

In terms of complexity we can split the number of comparison for the two levels:
1. For the first scan we will have a list with a node with probability $1/2$ so it will have at most $n/2$ nodes and so elements to scan.
2. In the second scan the probability that any node is followed by $k$ nodes without duplicates is $2^{-k}$, so the expected number of nodes examined in the second scan is at most:
$$
1 + \sum_{k \geq 0} 2^{-k} = 2
$$
So the total cost of a search go from $n$ to $n/2 + 2$, roughly a factor of two i savings.

From here we can extend this idea **add shortcuts to the shortcuts** and repeat recursively. That's exactly how skip list are constructed. So we do the following things:
1. For each node in the original list we flip a coin over and over until we get tails
2. For each heads we make a duplicate of the node
3. The duplicates are stacked up in levels, and the nodes on each level are strung toghether into sorted linked list.
4. Each node $v$ stores:
	- A search key `key(v)`
	- A pointer to its next lower copy `down(v)`
	- A pointer to the next node in its level `right(v)`

The search algorithm is very simple:
![[Pasted image 20261029131131.png]]

![[Pasted image 20261029131137.png|468]]

Regards the **time complexity**:
- Since each level of the skip lists has about half the number of nodes as the previous level, the total number of levels should be about $O(\log n)$. 
- Similarly, each time we add another level of random shortcuts to the skip list, we cut the search time roughly in half, so after $O(\log n)$ levels, we should have a search time of $O(\log n)$

### Number of Levels
Now let's formalize the intuition for the time complexity. First of all let's assume the keys are the integers 1 through $n$, and let $L(x)$ be the number of levels of the skip list that contain some search key $x$ (not counting the bottom level).

Each new copy of a node $x$ is created with probability $1/2$ from the previous level. We can compute the expected value of $L(x)$ recursively:
$$
E[L(x)] = \frac{1}{2} \cdot 0 + \frac{1}{2}(1 + E[L(x)])
$$
This means:
- With probability $1/2$ we flip tails (on a imaginary coin) so $L(x) = 0$
- And with a probability $1/2$ we flip heads, increase $L(x)$ by one and recurse

Solving this equation fives us $E[L(x)] = 1$.  To analyze the worst case we need to estimate the max high of the skip list, it is defined as:
$$
L = \max_x L(x) 
$$
- La probabilità che una specifica chiave $x$ raggiunga almeno il livello $\ell$ richiede $\ell$ Teste consecutive: $$ Pr[L(x) \ge \ell] = 2^{-\ell} $$
- La Skip List ha almeno $\ell$ livelli se e solo se almeno una delle $n$ chiavi raggiunge il livello $\ell$. Applicando l'**Union Bound**: $$ Pr[L \ge \ell] \le \sum_{x=1}^{n} Pr[L(x) \ge \ell] = n \cdot 2^{-\ell} $$
- Scegliendo $\ell = c \log_2 n$ per una costante $c > 1$: $$ Pr[L \ge c \log_2 n] \le \frac{n}{2^{c \log_2 n}} = \frac{n}{n^c} = \frac{1}{n^{c-1}} $$
We conclude that **with high probability, a skip list has $O(\log n)$ levels**.  This high-probability bound indirectly imples a bound on the expected number of levels. Some simple algebra gives us the following alternate defintion for expectation:
$$
E[L] = \sum_{\ell \ge 1} \ell \cdot Pr[L = \ell] = \sum_{\ell \ge 1} Pr[L \ge \ell]
$$
Spezziamo la sommatoria in due parti: la prima fino a $\lg n$ (dove $\lg n = \log_2 n$) e la seconda da $\lg n + 1$ in poi: 
$$
E[L] = \sum_{\ell=1}^{\lg n} Pr[L \ge \ell] + \sum_{\ell = \lg n + 1}^{\infty} Pr[L \ge \ell]
$$
- **Prima parte (**$\ell \le \lg n$**):** Maggioriamo ogni probabilità con $1$ (poiché le probabilità non possono superare 1): $$ \sum_{\ell=1}^{\lg n} Pr[L \ge \ell] \le \lg n $$
- **Seconda parte (**$\ell \ge \lg n + 1$**):** Sostituiamo il bound derivato dall'Union Bound $Pr[L \ge \ell] \le \frac{n}{2^\ell}$: $$ \sum_{\ell = \lg n + 1}^{\infty} Pr[L \ge \ell] \le \sum_{\ell = \lg n + 1}^{\infty} \frac{n}{2^\ell} $$
Ponendo il cambio di indice $i = \ell - \lg n$ (ovvero $\ell = \lg n + i$), la sommatoria diventa: 
$$
\sum_{i=1}^{\infty} \frac{n}{2^{\lg n + i}} = \sum_{i=1}^{\infty} \frac{n}{n \cdot 2^i} = \sum_{i=1}^{\infty} \frac{1}{2^i} = 1
$$
Sommando i due contributi otteniamo: 
$$
E[L] \le \lg n + 2
$$

### Logarithmic Search Time
It's a little easier to anlyze the cost of search if we imagine running the algorithm backward. We will have an inverse procedure call `InvSkipListFind` that takes the output from `SkipListFind` as input and traces back through the data structure to the upper left corner. 

Skip lists dont really have up and left poitners, but we'll pretend that they do so we don't have to write the inverse operation
![[Pasted image 20261029134954.png]]

Now for every node $v$ in the skip list `up(v)` exists with probability $1/2$. So for purposes of analysis `InvSkipListFind` is equivalent to the following algoriithm:

![[Pasted image 20261029135049.png]]

Obviously, the expected number of heads is exactly the same as the expected number of TAILs. Thus, the expected running time of this algorithm is twice the expected number of upward jumps. Since we already know that the number of upward jumps is $O(\log n)$ with high probability, we  can conclude that **the worst-case search time is $O(\log n)$ with hight probability**
# References

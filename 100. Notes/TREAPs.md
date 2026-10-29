---
Data: 2026-10-29T13:52:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
Area:
  - "[[Master's Degree.base]]"
---
# TREAPs

A **Treap** is a **binary tree** with an extra property for each node by using a **Min-Heap**. So for each node we have:
- **search key**: where the sequence of search keys is sorted
- **priority value**: where each node's priority is smaller than the priorities of its children

In other words, a treap is simultaneously a binary search tree for the search keys and a (min-)heap for the priorities. In our examples, we will use letters for the search keys and numbers for the priorities.

In left a implementation with a **Min-Heap**, right an implementation with a **Max-Heap**:

![[Pasted image 20261029214633.png|565]]

First of all let's assume that all keys and priorities are distinct, under this assumption we can prove by induction the structure of a treap is completely determined by the search keys and priorities of its node. 

> [!success] Proof
> What we want demostrate in detail is that **exists one and only one valid Treap** that contains the element $(key, value)$
> - **Caso Base: Insieme vuoto (**$|S| = 0$**)**: Un insieme di zero nodi può essere rappresentato solo ed esclusivamente dal Treap vuoto. La struttura è univocamente determinata
> - **Ipotesi Induttiva**: Assumiamo che per qualsiasi insieme di nodi $S'$ con dimensione $|S'| < n$ (con chiavi e priorità tutte distinte), la struttura del Treap sia **unica e completamente determinata** dalle sue chiavi e priorità.
> - **Passo Induttivo** ($|S| = n \ge 1$): Consideriamo un insieme $S$ composto da $n$ elementi con chiavi e priorità distinte. Dobbiamo mostrare che esiste una sola ed unica posizione per ciascun elemento:
> 	1. **Scelta univoca della Radice**: Per la proprietà di **Min-Heap**, la radice dell'albero deve contenere l'elemento con la **priorità minima globale** in tutto l'insieme $S$, Poiché tutte le priorità sono per ipotesi distinte, il minimo è unico. Quindi **esiste un solo candidato possibile per essere la radice dell'albero**, ossia il nodo $v$.
> 	2. **Partizione univoca dei Sottoalberi**: Poiché tutte le chiavi di ricerca sono distinte, per ogni nodo $u \neq v$ la condizione $key(u) < key(v)$ (per decidere se pozionare a destra o sinsitra) è o strettamente vera o strettamente falsa. Di conseguenza, l'insieme dei nodi rimanenti $S \setminus \{v\}$ si suddivide in modo **rigido ed univoco** in due insiemi disgiunti $L$ e $R$
> 	3. **Applicazione dell'Ipotesi Induttiva sui Sottoalberi**: L'insieme $L$ contiene solo gli elementi con chiave minore della radice, mentre $R$ contiene quelli con chiave maggiore, Poiché la radice $v$ è stata rimossa, entrambi i sottoinsiemi hanno dimensione strettamente minore di $n$ ($|L| < n$ e $|R| < n$). Inoltre, sia $L$ sia $R$ soddisfano l'ipotesi di avere chiavi e priorità tutte distinte. Per l'**ipotesi induttiva**, la struttura del sottoalbero sinistro per l'insieme $L$ e la struttura del sottoalbero destro per l'insieme $R$ sono **completamente e univocamente determinate**

### Cartesian Trees
Another interesting description interprets the keys and priorities as the coordinates of a set of points in the plane:
- The root corresponded to a $T$ whose joint lies on the topmost point
- The $T$ splits the plane into three parta:
	- The top part is (by definition empty)
	- The left and right parts are split recursively

This interpretation has some interesting application in [[Mesh Data Structures|computational geometry]].

![[Pasted image 20261029220150.png]]

### Treap Operations
##### Search
The search algorithm is the usual one for binary search trees, that means:
- The time for a **successful** search is proportional to the depth of the node.  So for **key $k$** takes $O(depth(v))$ time, where $v$ is the node with $key(v) = k$
- The time for an **unsuccessful** search is proportional to the depth of either its successor or its predecessor. So:
	- let $v^{-}$ be the in-order **predecessor** of $k$ (the node whose key is just barely smaller than $k$), 
	- and let $v^+$ be the in-order **successor** of $k$ (the node whose key is just barely larger than $k$)
	Since the last node examined by the binary search is either $v^-$ or $v^+$, the time for an unsuccessful search is either $O(depth(v⁺))$ or $O(depth(v⁻))$

##### Insert/Delete
To insert a new node z, we start by using the **standard binary search tree insertion algorithm** to insert it at the bottom of the tree. At the point, we will have: the search keys still form a search tree, but the priorities may no longer form a heap

![[Pasted image 20261029231858.png|700]]

To fix this issue in the heap property we can perform a **rotation** at $z$, a local operation that decreases the depth of $z$ by one and increases its parent's depth by one, while maintain the search tree property. Rotation **can be performed in const time** since they only involve simple pointer manipulation.

![[Pasted image 20261029224105.png]]

Inserting a new node with key k takes either $O(depth(v⁺))$ time or $O(depth(v⁻))$ time, where $v⁺$ and $v⁻$ are the predecessor and successor of the new node. **Deletion is just insertion in reverse**
##### Split/Merge
We may want to **merge** two treaps $T_<$ and $T_>$ where every node is $T_<$ has a smaller search key than any node in $T_>$, into one super treap.

In this case merge is just splitting in reverse: create a dummy root whose left sub-treap is $T_<$ and whose right sub-treap is $T_>$, rotate the dummy node down to a leaf and the cut it off.

Splitting a treap at pivot value $k$ takes either $O(depth(v⁺))$ time or $O(depth(v⁻))$ time, since it costs the same as inserting a new dummy root with search key $k$ and priority $-\infty$. **Merging** is just splitting in reverse.

### Random Priorities

> [!NOTE] Definition
> A **randomized treap** is a treap in which the **priorities are independently and uniformly distributed continuous random variable**, that means that whenever we insert a new search key into the treap, we generate a random real number between (say) 0 and 1 and use that number as the priority of the new node.

The cost of all the operations we discussed, search, insert, delete, split, join, is proportional to the depth of some node in the tree. Here we’ll see that the expected depth of any node is $O(\log n)$, which implies that the expected running time for any of those operations is also $O(\log n)$

Let $x_k$ denote the node with the kth smallest search key. To analyze the expected depth, we define an indicator variable:
$$
A_k^i = \begin{cases} 1 & \text{se } x_i \text{ è un antenato proprio di } x_k \\ 0 & \text{altrimenti} \end{cases}
$$
Since the depth of v is just the number of proper ancestors of v, we have the following identity:
$$
\text{depth}(x_k) = \sum_{i=1}^{n} A_k^i
$$
Now we can express the expected depth of a node in terms of these indicator variables as follows.
$$
E[\text{depth}(x_k)] = E[\sum^k_{i=1} A^i_k] = \sum_{i=1}^{n} E[A_k^i] = \sum_{i=1}^{n} \mathbb{P}(A_k^i = 1) = \sum_{i=1}^{k-1} \mathbb{P}(A_k^i = 1) + \sum_{i=k+1}^{n} \mathbb{P}(A_k^i = 1)
$$
Where we define $X(i,k)$ that is the subset of nodes between $x_i$ and $x_k$ (inclusi). in the final part we have:
- Se $i < k$, $X(i, k) = \{x_i, x_{i+1}, \dots, x_k\}$
- Se $i > k$, $X(i, k) = \{x_k, x_{k+1}, \dots, x_i\}$

$X(i,k)$ and $X(k,i)$ always denote prceisly the same subset, and this subset contains $|k-1| + 1$ nodes. So from here we can say that $X(1,n) = X(n,1)$ contains all $n$ nodes in the trep.


> [!success] Lemma
> For all $i\neq k$, $x_i$ is a proper ancestor of $x_k$ if and only if $x_i$ has the smallest priority among all nodes in $X(i, k)$

Since each node in $X(i,k)$ is equality likely to have smallest priority, we immediately have the probability we wanted:
$$
\Pr[A_k^i = 1] = \frac{[i \neq k]}{|k - i| + 1} = \begin{cases} 
\frac{1}{k - i + 1} & \text{se } i < k \\[8pt]
0 & \text{se } i = k \\[8pt]
\frac{1}{i - k + 1} & \text{se } i > k 
\end{cases}
```
$$
To compute the expected depth of a node $x_k$, we just plug this probability into our formula and we resolve:
$$E[\text{depth}(x_k)] = \sum_{i=1}^{k-1} \frac{1}{k - i + 1} + \sum_{i=k+1}^{n} \frac{1}{i - k + 1} $$
Effettuando i cambi di variabile nei due indici di sommatoria:

1. Nella prima somma (per $i < k$), poniamo $j = k - i + 1$. Al variare di $i$ da $1$ a $k-1$, l'indice $j$ varia da $k$ a $2$: 
$$
\sum_{j=2}^{k} \frac{1}{j} = H_k - 1
$$
2. Nella seconda somma (per $i > k$), poniamo $j = i - k + 1$. Al variare di $i$ da $k+1$ a $n$, l'indice $j$ varia da $2$ a $n - k + 1$: 
$$
\sum_{j=2}^{n-k+1} \frac{1}{j} = H_{n-k+1} - 1
$$
_(dove_ $H_z = \sum_{j=1}^z \frac{1}{j}$ _è il_ $z$_-esimo numero armonico)_. Sommando i due contributi: 
$$
E[\text{depth}(x_k)] = (H_k - 1) + (H_{n-k+1} - 1)
$$
Poiché per i numeri armonici vale la disuguaglianza $H_z - 1 < \ln z$: 
$$
E[\text{depth}(x_k)] < \ln k + \ln(n - k + 1) - 2 < 2 \ln n - 2
$$
Poiché $2 \ln n = 2 (\ln 2) \log_2 n \approx 1.386 \log_2 n$, la profondità attesa di qualsiasi nodo è strettamente delimitata da: 
$$
E[\text{depth}(x_k)] < 2 \ln n - 2 = O(\log n)
$$
# **References**
- [[04__EricksonNotes-Treap-SL.pdf]]
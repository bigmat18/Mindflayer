---
Data: 2026-05-03T23:17:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Vector Norms

### Vector Norms
A **norm** is a function that a non negative real number to a vector. It is also way to measure the **length** or **distance from 0** of a vector. A norm functon is:
$$
||\cdot||: X \to \mathbb{R}
$$
and It generalises the absolute value and it is valid if these properties are valid:
- $||v||> 0$ with equality iff $v=0$
- $||v\alpha|| = ||v|||\alpha|$ for $\alpha \in \mathbb{C}$
- $||v+w|| \leq ||v|| + ||w|| \forall v,w$ this is call **Triangle Inequality**

There are many norms and all are defined by:
$$
||x||_p = \bigg(\sum^n_{i=1} |x_1|^p \bigg)^{1/p}
$$
with $p\in [1, +\infty)$ . With $n=1$ all the norms are the the abosulte value. With $p\in(1,2)$ the **Triangle inequality is invalid**.

##### Euclidean Norm
The (Euclidean) norm represents the "length" of a vector, induced by the scalar product:
$$||x||_2 = \| x \| := \sqrt{\langle x , x \rangle} = \sqrt{x_1^2 + \dots + x_n^2}$$
**Properties:**
1.  **Positivity:** $\| x \| \ge 0$, and $\| x \| = 0 \iff x = 0$.
2.  **Homogeneity:** $\| \alpha x \| = | \alpha | \| x \|$ for any scalar $\alpha \in \mathbb{R}$.
3.  **Triangle Inequality:** $\| x + z \| \le \| x \| + \| z \|$. This is crucial: the direct path is always the shortest.
###### Examples
- Euclidean norm: $v \in \mathbb{R}^n, v=\begin{bmatrix}v_1\\\vdots\\v_n\end{bmatrix}$ 
$$
||v||_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} = \sqrt{v^Tv} = \sqrt{<v,v>}
$$
- $||v||_1 = |v_1| + |v_2| + \dots + |v_n|$
- $||v||_{max} = \max_{i=1,\dots,n}{|v_1|}$ 

Fact: for any vector $v\neq 0$, you can write $v=\alpha\cdot w$ with $\alpha=||v||, w=\frac{v}{||v||}$
###### Distance and Neighborhoods 
Once we have a norm, we can define how "far apart" two points (solutions) are. 
 **Euclidean Distance:** $$d( x , z ) := \| x - z \| = \sqrt{( x_1 - z_1 )^2 + \dots + ( x_n - z_n )^2}$$
 **Formal Properties of the Distance ($d$):** The distance $d(x, z) = \| x - z \|$ must satisfy:
1.  **Positivity:** $d( x , z ) \ge 0 \quad \forall x , z \in \mathbb{R}^n$, and $d( x , z ) = 0 \iff x = z$
2.  **Homogeneity (relative to origin):** $d( \alpha x , 0 ) = | \alpha | d( x , 0 ) \quad \forall x \in \mathbb{R}^n, \alpha \in \mathbb{R}$
3.  **Triangle Inequality:** $d( x , w ) \le d( x , z ) + d( z , w ) \quad \forall x, w, z \in \mathbb{R}^n$


 **The Ball (The "Neighborhood"):** For a center $x \in \mathbb{R}^n$ and radius $r > 0$, the ball $B( x , r )$ is the set of all points within distance $r$: $$B( x , r ) = \{ z \in \mathbb{R}^n: \| z - x \| \le r \}$$
**Topology and Optimization:** The norm defines the **Ball** $B(x, r)$, which in turn defines the **topology** of the space. This is not just theoretical: it tells us "what is next to what", which is essential for algorithms to decide where to move next. 

> **Key Insight:** There are other scalar products and norms (e.g., for matrices or functions), but in this course, we focus on the Euclidean ones. However, the properties above are universal for any valid inner product space.


# References
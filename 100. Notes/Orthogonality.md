---
Data: 2025-11-20T11:27:00
Tags:
  - note
  - youngling
Connection:
  - "[[Linear Algebra]]"
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Orthogonality

### Vector Norms
A **norm** is a way to measure the **length** or **distance from 0** of a vector. It generalises the absolute value. Properties:
- $||v||> 0$ with equality iff $v=0$
- $||v\alpha|| = ||v|||\alpha|$ for $\alpha \in \mathbb{C}$
- $||v+w|| \leq ||v|| + ||w|| \forall v,w$ this is call **Triangle Inequality**

###### Examples
- Euclidean norm: $v \in \mathbb{R}^n, v=\begin{bmatrix}v_1\\\vdots\\v_n\end{bmatrix}$ 
$$
||v||_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} = \sqrt{v^Tv} = \sqrt{<v,v>}
$$
- $||v||_1 = |v_1| + |v_2| + \dots + |v_n|$
- $||v||_{max} = \max_{i=1,\dots,n}{|v_1|}$ 

Fact: for any vector $v\neq 0$, you can write $v=\alpha\cdot w$ with $\alpha=||v||, w=\frac{v}{||v||}$

### Orthogonality
The 2-norm is nice because there are many matrice $U$ that preserve it: A **square** $U\in \mathbb{R}^{m\times n}$ is called **orthogonal** if:
- $U^TU = Id$
- $UU^T = Id$
- $U^{-1}=U^T$
Each of these properties implies the other

**Property**: If $U$ is orthogonal, then $||Ux||_2 = ||x||_2$ and more generally
$$
(Ux)^T(Uy) = x^Ty
$$
The **proof** is the following:
$$
||U_x|| = \sqrt{(Ux)^T(Ux)} = \sqrt{x^TU^TUx} = \sqrt{x^Tx} = ||x||
$$

The **geometric idea** is that the transformation associated to $U$ is a rotation or a mirror symmetry: length and angles are preserved.

**Definition**: two vector $u,v$ are orthogonal to each other if $u^Tv=0$ 

**Property**: for any matrices $A,B$ "that make sense":
1. $(AB)^T = B^TA^T$
2. $(AB)^{-1} = B^{-1}A^{-1}$ where $(AB)^{-1} = B^{-1}A^{-1}(AB) = B^{-1}B= Id$

**Lemma**: if $U_1, U_2$ are orthogonal matrixes fo size nxn then $U_1U_2$ is also an orthogonal matrix, because:
$$
(U_1U_2)^T(U_1U_2) = U_2^TU_1^TU_1U_2 = U_2^TU_2 = Id
$$
##### Ortonormal
The columns $u_1, u_2, \dots, u_m$ of an orthogonal matrix $U=[\begin{smallmatrix}u_1&u_2&\dots&u_m\end{smallmatrix}]$ are **orthonormal**:
$$
u_i^Tu_j = \begin{cases}0&i\neq j\\1&1=j\end{cases}
$$
and so are its row.

Sometimes vectors $u_1, u_2, \dots, u_m$ such that $u_i^Tu_j = 0$ when $i\neq j$ without the second condition, are called **orthogonal**. This may be confusing.

##### Product of orthogonal metrics
The product of two orthogonal matrices is an orthogonal matrix. Simple to verify if you remember a couple of facts from linear algebra:

**Property**: for any two matrix $A,B$:
- $(AB)^T = B^TA^T$
- $(AB)^{-1} = B^{-1}A^{-1}$ (when $A,B$ are square invertible)
Indeed,
$$
(AB)^{-1} = B^{-1}A^{-1} = A^TB^T = (AB)^T
$$

##### Orthogonal Column
We will often work with tall thin rectangular matrices with orthonormal columns:
$$
U_0=[\begin{smallmatrix}u_1&u_2&\dots&u_m\end{smallmatrix}] \in \mathbb{R}^{m\times n}. \:\:\:\: (m\geq n)
$$
They are the first block of an orthogonal matrix: $U=[U_0\:\:U_c]$

# References


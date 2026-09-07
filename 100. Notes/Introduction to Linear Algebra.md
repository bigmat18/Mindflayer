---
Data: 2025-11-19T16:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Introduction to Linear Algebra

### Matrix-Vector products
**The operational way**: row-by-column $y_i  = \sum_{j}A_{ij}x_j$
$$
\begin{bmatrix}
A_{11}&A_{12}&A_{13}\\
A_{21}&A_{22}&A_{23}\\
A_{31}&A_{32}&A_{33}\\
A_{41}&A_{42}&A_{43}\\
\end{bmatrix}
\begin{bmatrix}
x_{1}\\
x_{2}\\
x_{3}\\
\end{bmatrix}
=
\begin{bmatrix}
y_{1}\\
y_{2}\\
y_{3}\\
y_{4}\\
\end{bmatrix}
$$
**The smart way:** **linear combination** of columns of A
$$
\begin{align}
\begin{bmatrix}
A_{11}\\
A_{21}\\
A_{31}\\
A_{41}\\
\end{bmatrix}
x_1 +
\begin{bmatrix}
A_{12}\\
A_{22}\\
A_{32}\\
A_{42}\\
\end{bmatrix}
x_2 +
\begin{bmatrix}
A_{31}\\
A_{31}\\
A_{33}\\
A_{34}\\
\end{bmatrix}
x_3 =
\begin{bmatrix}
y_{1}\\
y_{2}\\
y_{3}\\
y_{4}\\
\end{bmatrix}\\
v_1 \:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\: 
v_2 \:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:  
v_3 \:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:
\end{align}

$$
The entries of $x$ are coordinates used to write $y$ as a **linear combination** of $v_1, v_2, v_3$

### Bases and Linear Indipendency
A **bases** is a tuple of vectors $v_1, v_2, \dots, v_n$ such that we can write each vector $y$ of a certain space, **uniquely**, as a linear combination of them.
- **Uniquely**: coordinates for each vector are unique/well defined. We can also say that the vector in the base are **Linear Indipendency**
- It generate all space space (the **span** of the bases)

###### Example
$$
e_1 = \begin{bmatrix}1\\0\\0\\0\end{bmatrix}, \:\:\:
e_2 = \begin{bmatrix}0\\1\\0\\0\end{bmatrix}, \:\:\:
e_3 = \begin{bmatrix}0\\0\\1\\0\end{bmatrix}, \:\:\:
e_4 = \begin{bmatrix}0\\0\\0\\1\end{bmatrix}
$$
Coordinates (w.r.t.) with reference to this basis $\Leftrightarrow$  vector entries:
$$
y = e_1y_1 + e_2y_2 + e_3y_3 + e_4y_4
$$
**Canonical basis**: vectors with only one 1, for example n=4
### Linear systems
**Problem**: find **coordinates** $x_1, \dots, x_n$ needed to write $y$ as linear combinations of the columns of $A \in \mathbb{R}^{m\times n}$ or
$$
Ax = y
$$
sometimes there are multiple solutions, or none, for example:
$$
A = \begin{bmatrix}2&0&1\\0&1&1\\0&0&0\\0&0&0\end{bmatrix},\:\:\:
y_1 = \begin{bmatrix}4\\4\\0\\0\end{bmatrix}, \:\:\:
y_2 = \begin{bmatrix}4\\4\\1\\0\end{bmatrix}
$$
$Im(A)$: **Image** of $A$: the set of vectors $y$ that we can obtain
$Ker(A)$: **Kernel** of $A$: possible choices of $x$ that produce $Ax=0$

**Main problem** (initially): find $x$ that reaches a given $y$ exactly, or gets as close as possible.

##### Square Linear systems
$A$ is called **Invertible** if $Ax = y$ has a unique solution, i.e., its columns are a basis or $\mathbb{R}^n$. (it must be a **square** for this hold)

In this case, the solution is given by another matrix: $x = A^{-1} y$
$$
AA^{-1} = A^{-1}A = I = \begin{bmatrix}1\\&1\\&&\ddots\\&&&1\end{bmatrix}
$$
**Warning**: $inv(A) \cdot y$ is not the best way to solve $Ax = y$ numerically. Most languages have a specialised instruction.

### Rank
The word **rank** has a precise linear-algebra meaning. While, for some computer scientists, rank = number of indices of an array.

**Definition**: Rank of a matrix $A$ is equals to the minimum $r$ so that is possible to find vectors $v_1, \dots, v_r$ such that all the columns of $A$ are linear combinations fo these vectors.
###### Example
$$
ww^T = \begin{bmatrix}
v_1w_1&v_1w_2&v_1w_3\\
v_2w_1&v_2w_2&v_2w_3\\
v_3w_1&v_3w_2&v_3w_3
\end{bmatrix}
$$
har rank $r=1$: all columns are multiples of $v$

**Theorem**: column rank = row rank: if you replace "columns" with "row" in the definition, you get the same value $r$, For instance, in the example above, all rows are multiples of $w^T$.

### Full column Rank
**What is going on:** To understand why we need full column rank, look at what happens when a matrix _lacks_ it. If there is a non-zero vector $z \in \ker A$ (for example, $A\begin{bmatrix}1\\ 1\\ -1\end{bmatrix}=0$), we lose uniqueness. If $x$ is a solution to a system, then so is $x+z, x+2z, x-37z...$ because $A(x+z) = Ax + Az = Ax + 0$.

> **Definition** We say that $A\in\mathbb{R}^{m\times n}$ has **full column rank** if $\ker A=\{0\}$, or, equivalently: $\text{rank } A=n$ or, equivalently: there is no $z\in\mathbb{R}^{n}$, $z\ne0$ such that $Az=0$.

**Why it matters:** We shall see, via several equivalent conditions, that the least squares problem $\min||Ax-y||$ has a unique solution if and only if $A$ has full column rank. Without this property, the problem would result in infinitely many equally optimal solutions.

### Triangular Linear Systems and substitution
**Idea** if $A$ is **lower triangular** (ie, square with all zeros above the main diagona), then we can solve $Ax=y$ one entry at a time by **forward-substituition**

![[Screenshot 2025-11-19 at 17.49.33.png | 500]]
Cost: $O(n^2)$
- This is **cheaper** that computing $A^{-1}$ which costs $O(n^3)$
- Another instance of an important principle: never form inverses explicitly
- The same computations hold if the above quantities are **blocks**:
$$
x_1 = A^{-1}_{11}y; \:\:\: x_2= A^{-1}_{22}(y_2 - A_{21}x_1), \dots
$$
### The Scalar Product (Inner Product)
The (Euclidean) scalar product of $x \in \mathbb{R}^n$ and $z \in \mathbb{R}^n$ is defined as:
$$\langle x , z \rangle = \sum_{i=1}^n x_i z_i = x_1z_1 + \dots + x_n z_n$$

* **Geometric Interpretation:** $\langle x , z \rangle = \| x \| \cdot \| z \| \cdot \cos( \theta )$, where $\theta$ is the angle between the two vectors.
	![[Pasted image 20260217205150.png | 200]]   ![[Pasted image 20260217205159.png | 200]]   ![[Pasted image 20260217205225.png | 200]]


    * **Orthogonality:** $\langle x , z \rangle = 0 \iff x \perp z$ (the vectors are perpendicular).
	![[Pasted image 20260217210120.png | 200]]

	- **Directional Info:** * $\langle x , z \rangle > 0 \implies$ they point in a similar direction ($\theta < 90^\circ$).
        * $\langle x , z \rangle < 0 \implies$ they point in opposite directions ($\theta > 90^\circ$).
	![[Pasted image 20260217210229.png | 200]]
	

* **Cauchy-Schwarz Inequality:** $| \langle x , z \rangle | \le \| x \| \| z \|$. 
    Equality holds only if $x$ and $z$ are linearly dependent (pointing in the exact same or opposite direction).

The scalar product $\langle x, z \rangle$ is defined by four fundamental properties:
1.  **Symmetry:** $\langle x, z \rangle = \langle z, x \rangle \quad \forall x, z \in \mathbb{R}^n$
2.  **Positivity:** $\langle x, x \rangle \ge 0 \quad \forall x \in \mathbb{R}^n$, and $\langle x, x \rangle = 0 \iff x = 0$
3.  **Homogeneity:** $\langle \alpha x, z \rangle = \alpha \langle x, z \rangle \quad \forall x, z \in \mathbb{R}^n, \alpha \in \mathbb{R}$
4.  **Additivity:** $\langle x + w, z \rangle = \langle x, z \rangle + \langle w, z \rangle \quad \forall x, w, z \in \mathbb{R}^n$

### The Determinant
**Definition:** Let $A$ be an $n \times n$ square matrix with entries $a_{ij}$. The determinant, denoted as $\det(A)$ or $|A|$, is a scalar value uniquely defined by the Leibniz formula:
$$ \det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^n a_{i,\sigma(i)} $$
Where:
- $S_n$ is the set of all permutations $\sigma$ of the set $\{1, 2, \dots, n\}$.
- $\text{sgn}(\sigma)$ is the signature of the permutation ($+1$ for even permutations, $-1$ for odd).
    

**Geometric Interpretation:**
$\det(A)$ represents the signed $n$-dimensional volume of the parallelepiped spanned by the column (or row) vectors of the matrix $A$.

### Invertibility and Singularity
**Definition (Invertibility):** An $n \times n$ matrix $A$ is **invertible** (or **non-singular**) if there exists an $n \times n$ matrix $B$ such that:
$$ AB = BA = I_n $$
where $I_n$ is the $n \times n$ identity matrix. The matrix $B$ is denoted as $A^{-1}$.

**Definition (Singularity):** An $n \times n$ matrix $A$ is **singular** if it is not invertible (i.e., no such matrix $A^{-1}$ exists).

###### The Fundamental Theorem of Invertibility
The relationship between the determinant and the invertibility of a matrix is strictly governed by the following theorem:

**Theorem:** Let $A$ be an $n \times n$ square matrix over a field.
- $A$ is **invertible (non-singular)** $\iff \det(A) \neq 0$.
- $A$ is **singular** $\iff \det(A) = 0$.

**Formal Explanation:**
From the property of determinants, we know that for any two $n \times n$ matrices $A$ and $B$:
$$ \det(AB) = \det(A)\det(B) $$
If $A$ is invertible, then $A A^{-1} = I_n$. Taking the determinant of both sides yields:
$$ \det(A A^{-1}) = \det(I_n) \implies \det(A)\det(A^{-1}) = 1 $$
For the product $\det(A)\det(A^{-1})$ to equal $1$, it is strictly mathematically required that $\det(A) \neq 0$. Furthermore, this proves that $\det(A^{-1}) = \frac{1}{\det(A)}$.

**Algebraic and Geometric Consequence:**
- **If $\det(A) = 0$:** The column vectors of $A$ are linearly dependent. The matrix maps the $n$-dimensional space $\mathbb{R}^n$ onto a subspace of strictly lower dimension (the volume collapses to zero). Because multiple input vectors map to the same output vector, the mapping is not injective (one-to-one), and therefore an inverse function cannot exist.    
- **If $\det(A) \neq 0$:** The column vectors form a basis for $\mathbb{R}^n$ (they are linearly independent). The matrix represents a bijective (one-to-one and onto) linear transformation, meaning every point in the target space has exactly one corresponding point in the domain, allowing the transformation to be perfectly reversed by $A^{-1}$.

# References
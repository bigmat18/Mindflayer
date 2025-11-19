---
Data: 2025-11-19T16:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Linear Algebra]]"
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

### Bases
A **bases** is a tuple of vectors $v_1, v_2, \dots, v_n$ such that we can write each vector $y$ of a certain space, **uniquely**, as a linear combination of them.
- **Uniquely**: coordinates for each vector are unique/well defined
- **Canonical basis**: vectors with only one 1, for example n=4
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

### Square Linear systems
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
# References
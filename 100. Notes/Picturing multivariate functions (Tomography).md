---
Data: 2026-02-18T20:24:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Picturing multivariate functions (Tomography)

Visualizing the graph of a function $gr(f) \in \mathbb{R}^{n+1}$ becomes impossible as soon as $n > 2$ (which would require a 4D plot). To study these functions, we use **Tomography**: instead of looking at the whole landscape, we look at a "slice" along a line.

* **The Concept:** Given a point $x$ and a direction $d$, we define a univariate function $\phi$ that represents the values of $f$ as we move along the line passing through $x$ in direction $d$.
    $$\phi_{x,d} ( \alpha ) = f ( x + \alpha d ) : \mathbb{R} \rightarrow \mathbb{R}$$
* **Key Properties:**
    * **Scale:** Changing the length of $d$ ($\| d \|$) only stretches or compresses the graph of $\phi$. For this reason, we usually use a **normalised direction** ($\| d \| = 1$).
    * **Coordinate Restriction:** The simplest tomography is moving along a single axis (the $i$-th coordinate). This means varying $x_i$ while keeping all other $x_j$ constant. This is the basis for partial derivatives.
    * **Utility:** While we can't see the whole function, we can always plot $gr( \phi_{x,d} )$ on a 2D plane to understand the behavior of $f$ locally.

In sintesi we set a point and direction and we draw only the graph along this line.
# References
---
Data: 2026-02-18T16:38:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Non-Homogeneous (separable)
A special case of quadrati funzione is the separable **non-homogeneous** quadratic function.

**Definition:** A separable (non-homogeneous) quadratic function is defined as the sum of $n$ independent univariate quadratic functions:
$$f(x) = \sum_{i=1}^n f_i(x_i) = \sum_{i=1}^n (a_i x_i^2 + b_i x_i)$$where the coefficients $(a, b) \in \mathbb{R}^{2n}$ are fixed.

**Example (Euclidean Norm)**  
A fundamental example is the squared Euclidean norm, where $a_i=1$ and $b_i=0$ for all $i$:
$$f(x) = \| x \|^2 = \sum_{i=1}^n x_i^2$$


To visualize the behavior, consider a 2D function $f(x_1, x_2) = a x_1^2 + x_2^2$ (with $b=0$). The shape of the **level sets** (contour lines) depends on the coefficient $a$:

* **Perfect Circles ($a = 1$):**
    If the coefficients are equal ($a=1$), the level sets are perfect circles ($x_1^2 + x_2^2 = c$).
![[Pasted image 20260218144424.png | 250]]

* **Ellipses ($a \neq 1$):** Changing $a$ stretches or compresses the graph, turning circles into ellipses.
    * **Larger $a$ ($a > 1$):** The function grows faster along $x_1$. The level sets become **vertically elongated** ($\updownarrow$).
	![[Pasted image 20260218144451.png | 250]]

    * **Smaller $a$ ($a < 1$):** The function grows slower along $x_1$. The level sets become **horizontally elongated** ($\leftrightarrow$).
	![[Pasted image 20260218144519.png | 250]]

How hard is it to find the minimum of these functions?

**Effect of Non-homogeneous terms ($b \neq 0$):** The linear terms $b_i x_i$ simply shift the location of the minimum (the center of the ellipses/circles).
$$[0, 0] \xrightarrow{\text{shift}} \left[ -\frac{b_1}{2a_1}, -\frac{b_2}{2a_2} \right]$$
*Reason:* For each independent $f_i(x_i) = a_i x_i^2 + b_i x_i$, the derivative is $2a_i x_i + b_i = 0 \implies x_i = -b_i / 2a_i$.

**Complexity ($O(n)$):** Since the function is separable, we solve $n$ independent univariate problems.
- This takes linear time, **$O(n)$**.
- **Note:** This is the last time optimization will be this simple. The next step (general quadratic functions) will introduce interactions between variables, making complexity much higher.

# References
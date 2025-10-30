---
Data: 2025-10-26T01:04:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Computer Science Metrics]]"
Area: "[[Master's degree]]"
---
# Gustafson's Law

**Gustafson’s Law** is a particular case of the **Scaled Speedup** that can be used to predict the theoretically achievable speedup using multiple processors when the parallelised part scales linearly with the problem size (Weak scaling), while the serial part remains constant.

It assumes the parallel part scales linearly with the amount of resources, while the serial part does not increase with the problem size.

![[Screenshot 2025-05-25 at 22.17.07.png|600]]

If we introduce a different form of Amdahl's Law we have the following formula:

![[Screenshot 2025-05-25 at 22.19.24.png | 450]]

Using different functions for $\gamma$ yields to the following two notable cases:
- $\gamma=1$ (ie $\gamma=\beta$) we have **Amdahl's Law**
- $\gamma=p$ (eg $\alpha=1; \beta=p$) i.e., the parallelizable part grows linear in p while the non-parallelizable part remains constant. We have **Gustafson’s law**:

![[Screenshot 2025-05-25 at 22.21.42.png| 350]]

**Weak scalability** is:
$$S'(n) = T_{C-\Sigma}(1, n\cdot w)/T_{C-\Sigma}(n,n\cdot w)$$
where $T_{C-\Sigma}(1, n\cdot w)$ is the execution time with parallelism 1 and problem size n-times greater than the reference one w.

![[Pasted image 20250511193325.png | 550]]

If you increase the problem size, the serial part remain the same (dark blue) and the parallel part grows. The time is same time but with growing data.
# References
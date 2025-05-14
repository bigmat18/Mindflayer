**Data time:** 15:35 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Optimal parallelism degree with single inputs

[[Introduction to Data Parallelism|Data parallelism]] can be applied both on **streams** and on a **single inputs**, but **how is the [[Optimal Parallelism Degree]] determined in these two cases?**

With the single inputs, we do not have an [[Inter Calculation Time|inter-departure time]] so the concept of [[Utilization Factor]] does not exists.

![[Pasted image 20250514155024.png | 400]]

### Cost Model
To study the [[Optimal Parallelism Degree]] with single inputs we have a very general approach. The goal is to find the **parallelism degree** such that the **completion time of the program is minimized**. We have to study (analytically) the completion time to find its minimum

**Example**: $\forall i=0\dots L-1 \::\: B[i] = F(A[i])$

![[Pasted image 20250514155654.png | 300]]

**Limited pipeline effect** between the three phases. The gather collective is partially overlapped with the internal calculation in the worker processes for this **all the sends in the gather are overlapped except the last one**.

The completion time is a function of the parallelism degree. To minimize it, we should find the parallelism degree such that **the first-order derivative is zero**.

![[Pasted image 20250514155935.png | 400]]

- If the optimal parallelism degree is greater than $N_{max}$ we use $N_{max}$
- Otherwise, we use $N_{opt}$ since we have a sufficient number of processors of exploit it.
# References
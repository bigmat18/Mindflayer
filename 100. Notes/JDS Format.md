**Data time:** 22:17 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# JDS Format

**Jagged Diagonal Storage (JDS)** is a format that can reduce **control divergence** without introducing any **padding bytes**. Rows of the matrix are sorted from the longest to the shortest.

![[Pasted image 20250601221932.png]]

First, nonzeros are grouped by row. Next, the rows are sorted by filling `row` with the row indexes. Finally, we store the elements of `column` and `value` in column major.

![[Pasted image 20250601222040.png | 600]]

As usual, each **CUDA thread** is assigned to a row of the matrix, and iterates through the nonzeros of that row by doing the dot product and computing its element of c.

![[Pasted image 20250601222110.png | 350]]

This format, as in [[Coordinate Format (COO)|COO]] and [[ELL Data Layout|ELL]], allows memory accesses by CUDA threads to be **coalesced**. The unique feature of JDS is that rows are sorted. Therefore, threads of the same warp tend to be assigned to rows with similar lengths, so reducing **control divergence**.
# References
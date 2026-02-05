---
Data: 2026-02-04T18:18:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Interconnecton Networks]]"
Area: "[[Master's degree]]"
---
# Foster’s Parallel Algorithm Design Method (PCAM)

Conceptual framework for reasoning about parallelization of a given problem. Ian Foster proposed the **PCAM** approach:
- **Partitioning**: decompose the problem into a large amount of small (fine-grained) tasks that can be  executed in parallel.
- **Communication**: determine the required communication between tasks (dependencies)
- **Agglomeration**: combine identified tasks into larger (coarse-grained) tasks **to reduce communication by  improving data locality** (balance overhead vs. parallelism)
- **Mapping**:  assign the aggragated tasks to processes according to the network topology to minimize  ecommunication, enable concurrency, and balance **workload** 

![[Pasted image 20250527183819.png]]

###### Example: Jacobi Iteration
**[[Stencil]] code** applied on a 2-dimensional array. Used to solve 2D PDE. Update each value in the matrix with the average of its four neighbors

![[Pasted image 20250527184005.png]]

The update rule is applied iteratively until convergence. Boundary values (yellow part) remain constant at each iteration (fixed boundary conditions). At the end of each iteration, swap the updated array with the original one to avoid overwriting during the next iteration

Replaces all points of a given 2D matrix by the average of  the values around it in every iteration step until  convergence: 

```c
copy(buff, data, rows, cols);
for (int k=1; k<MaxIter; k++) { 
	for (int i=1; i<rows-1; i++) 
		for (int j=1; j<cols-1; j++) 
			buff[i*cols+j] = 0.25f * ( data[(i+1)*cols+j] + data[i*cols+j-1] 
			+ data[i*cols+j+1]+data[(i-1)*cols+j] ); 
	residual = R(buff, data); // e.g., L2-norm of the difference among cells
	if (residual < THRESHOLD) break;
	swap(data, buff, rows, cols); 
}
```

- The Initial matrix
![[Pasted image 20250527184039.png | 250]]

- The matrix after 1 iteration
![[Pasted image 20250527184113.png | 250]]

- The matrix after 25 and 75 iterations 
![[Pasted image 20250527184143.png | 500]]

There are two parallel schemes for Jacobi Iteration
- **Partitioning**: The smallest task is the computation of a single element of the Jacobi matrix
- **Communication**: Within an iteration all fine-grain tasks can be computed independently. Each task needs the data of four neighbors. At the end of each iteration, there is a synchronization barrier among  all p processors, and data is exchanged.
- **Agglomeration**: Two options proposed:
	1. by row (or by column);
	2. by using a square grid.
- **Mapping**: it follows the policy used for the agglomeration to map coarse-grain tasks to processors. By row, contiguous groups of rows are assigned to the p processors; or by square grids, rectangles of square grids are assigned to the p  processors organized in a $\sqrt{p} \times \sqrt{p}$ grid. 
	
![[Pasted image 20250527185546.png]]

Problem size (grid size) $n\times n; p$ processes running on $p$ nodes. Considering the **linear model** for the cost of communications between two processes:
$$
T_{comm}(n) = t_0 + n \cdot s
$$
- **Method 1**: each process own roughly $\frac{n}{p}$ rows to colums
$$
T_{comm}(n) \approx 2 \cdot (t_0 + n \cdot s)
$$
- **Method 2**: each process owns a $\frac{n}{\sqrt{p}} \times \frac{n}{\sqrt{p}}$ sub-block
$$
T_{comm}(n) \approx 4 \cdot \bigg(  t_0 + \bigg( \frac{n}{\sqrt{p}}\bigg) \cdot s\bigg)
$$
 Method 2 superior for large p since communication time decreases with p while it remains constant for Method 1.
# References
**Data time:** 18:40 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Parallelization methodology and metrics]]

**Area**: [[Master's degree]]
# Roofline model

[[Relative Efficiency]] tells us if our parallelization scales as expected. But are good is utilizing the given machine. The **Roofline model** is used to bound:
- **floating-point (FP)** performance
- the machine **peak** performance
- **Arithmetic intensity (AI)** of the problem. This means the number of FP operations executed per byte read from memory.

###### Example "daxpy"
Give an example of code:
```c
void daxpy(size_t n, double a, const double *x, double *y)
{
	size_t i;
	for (i=0; i<n; i++) {
		y[i] = a * x[i] + y[i];
	}
}
```
In this example we perform **2 FP operations**. Furthermore we read two doubles, totally we read 16bytes. The Arithemtic Intensity (AI) is 2/16 = 0,125. Is this case AI is constant and not change with the problem size.

###### Example "gemm"
```c
void gemm(size_t n, const double *A, const double *B, const double *C)
{
	size_t i, j, k;
	double sum;
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			sum = 0.0;
			for (k=0; k<n; k++) {
				sum = sum + A[i*n+k] * B[k*n+j];
			}
			C[i*n+j] = sum;
		}
	}
}
```
Each iterations of the innermost loop executes two FP operations, so we have $2n^3$ FP. We assume the two input matrices are transferred from the memory once, we have $16n²$ bytes. AI is $2n³/16n² = \frac{n}{8}$

###### Example
Let's suppose that in the considered system we have $R_{peak} = 10GFLOPS, B = 10 GB/s$. Let's consider also this loop:
```c
double s= <some-value>;
for (int i = 0; i < N; i++) {
	s = s + A[i] * A[i];
}
```

We have $I = \frac{2 \cdot N}{8 \cdot N} = \frac{1}{4} = 0.25$ FLOP/byte at the steady-state. $P = \min(10, 0.25 \cdot 10) = 2.5$ GFLOPS, which means bandwidth-bound computation.

![[Pasted image 20250524163918.png | 280]]

### Utilization 
The roofline is computed as:
$$AttainablePerf(AI) = \max\{PeakFP, PeakBW \cdot AI\}$$
Where ActualFP will be certainly lower than the upper bound given by roofline. Based on its value, we can understand if the problem is **memory bound** or **compute bound** and how far we are from the top performance that we can achieve on that machine having a problem with a given AI.

![[Pasted image 20250511200652.png | 550]]

For the read points there are a problem of memory bound, instead for yellow there are compute bound problem, in both case we can improve performance until the line above.
### Hierarchical Roofline
Multiple rooflines can be superimposed upon each other to represent different levels in the memory hierarchy. It helps in analyzing the application's **date locality** and **cache reuse pattern**, and understand how efficiently data is flowing.

![[Pasted image 20250511201424.png|500]]
# References
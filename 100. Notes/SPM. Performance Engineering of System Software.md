**Data time:** 13:57 - 12-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]]

**Area**: [[Master's degree]]
# SPM. Performance Engineering of System Software

This part is take form: https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/resources/mit6_172f18_lec1/

This is an example calls **Matrix Multiplication** (assume for simplicity that $n = 2^k$):

![[Pasted image 20250312141400.png | 350]]

The machine where we execute this example is the following:
![[Pasted image 20250312142138.png | 400]]
The theoretical performance of this machine is calls **peak**:
$$peak = (2.9 \times 10^9) \times 2 \times 9 \times 19 = 836 \:\:GFLOPS$$
```python
import sys, random
from time import *

n = 4096

A = [[random.random() for row in xrange(n) for col in xrange(n)]]
B = [[random.random() for row in xrange(n) for col in xrange(n)]]
C = [[random.random() for row in xrange(n) for col in xrange(n)]]

start = time()
for i in xrange(n):
	for j in xrange(n):
		for k in xrange(n):
			 C[i][j] += A[i][k] * B[i][j]
end = time()

print('%0.6f', (end - start))
```

With this code we have a running time of 21042 seconds that corresponds to 6 hours. With this time we can calculate that python gets $2^{37} / 21042 = 6.35$ MFOPS that is the 0.00075% of peak. ($2^{37} = 2 n^3$).

![[Pasted image 20250312143236.png | 500]]

If we use Java we have a speed up of 8.8 times faster than Python.

![[Pasted image 20250312143331.png | 500]]

With C language we have another speed up compared to Python and also Java.
- **Python** is interpreted
- **Java** is compiled to byte-code
- **C** is compiled directly to machine code

Furthermore the improvement from speed language we can change the code. The first improvement is change the order of loops to improve the cache.

![[Pasted image 20250312144218.png | 300]]

Each processor reads and writes main memory in contiguous blocks, called **cache lines** and previously accessed cache lines are stored in a smaller memory, called **cache**, much faster than normal memory.
- **Cache hits**: access to data in cache (fast)
- **Cache miss**: accesses to data not in cache (slow)

![[Pasted image 20250312144444.png | 400]]

In this matrix-multiplication code, matrices are laid out in memory in **row-major order**. We can also have column-major order (for example in fortan). In C we have this order for memory.

### Access Patter for Order i, j, k

![[Pasted image 20250312144817.png | 400]]

### Access Pattern for Order i, k, j

![[Pasted image 20250312144918.png | 400]]
### Access Pattern for Order j, k, i

![[Pasted image 20250312144943.png | 400]]

### Compiler Optimisation
We can also add a set of optimisation flags during compilation. For example **clang** provides a collection of optimisation switches. We can also add a set of optimisation flags during compilation. For example **clang** provides a collection of optimisation switches. 

![[Pasted image 20250312145153.png | 350]]

If we use all the optimisation we have seen we can optein the following speed up:

![[Pasted image 20250312145342.png | 400]]

In total we have a percent of peak of 0.301%, that is still to low. For this reason we can use other types of optimisation that include parallelism approach.

### Multicore Parallelism
In this example we use a MIT tools to parallelise code. The **cilk_for** loop allows all iterations of the loop to execute in parallel.

Parallel i we have a running time of 3.18s:
```c
cilk_for (int i = 0; i < n; ++i)
	for (int k = 0; k < n; ++k)
		for(int j = 0; j < n; ++j)
			C[i][j] += A[i][k] * B[k][j]
```

Parallel j we have a running time of 531.71s:
```c
for (int i = 0; i < n; ++i)
	for (int k = 0; k < n; ++k)
		cilk_for(int j = 0; j < n; ++j)
			C[i][j] += A[i][k] * B[k][j]
```


Parallel i and j we have a running time of 10.64s:
```c
cilk_for (int i = 0; i < n; ++i)
	for (int k = 0; k < n; ++k)
		cilk_for(int j = 0; j < n; ++j)
			C[i][j] += A[i][k] * B[k][j]
```

**Rule of Thumb**: parallelize outer loop rather than inner loops.

### Hardware Caches, Revisited
IDEA: Restructure the computation to reuse data in the cache as much as possible. This because cache misses are slow, and cache hits are fast for this we try to make the most of the cache by **reusing the data that's already there**.

##### Data Reuse: Loops
- **4096 * 1 = 4096** writes to C
- **2096 * 1 = 4096** reads from A
- **4096 * 4096 = 16.777.216** reads from B
Total: 16.785.408 memory access


##### Data Reuse: Blocks
If we use a 64 x 64 blocks of C we have:
- **64 * 64 = 4096** writes to C
- **64 * 4096 = 262.144** reads from A
- **4096 * 64 = 262.144** reads from B
Total: 528.384 memory access


# References
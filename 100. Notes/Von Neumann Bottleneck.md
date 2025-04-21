**Data time:** 12:13 - 16-04-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Shared Memory Systems]]

**Area**: [[Master's degree]]
# Von Neumann Bottleneck

The basic structure of the classical **Von Neumann Architectyre** is a [[Shared Memory Architectures]]. In Early computer systems timings for accessing main memory and for computations were reasonably well balanced.

![[Pasted image 20250416122249.png | 300]]

During the past few decades, computation speed grew much faster than main memory access speed, resulting in a significant performance gap. This was called **Von Neumann Bottlenck** and it's the discrepancy CPU compute speed and main memory (DRAM) speed.

###### Example of Von Neumann Bottleneck
Simplified model to establish an **upper bound on performance**, will never go faster than what the model predicts.

![[Pasted image 20250416122606.png | 350]]

- 1 CPU, with 8 cores @3GHz capable of 16 floating point operations per core per clock cycle (FLOP/cycle). The peak performance is:
$$R_{peak} = 3 \cdot 8 \cdot 16 = 384 \:\:GFLOPS$$
- The DRAM peak memory transfer rate is 51.2 GB/s


**Note**: $R_{peak}$ provides a **theoretical peak  performance** metric for a system, but real-world applications often archive only a fraction of this due to memory and architectural limitations. 

Let's consider the **dot product** kernel on the example platform (we use double 8bytes). If (size of two vectors) $n = 2^{30} \to 2 \cdot n$ floating points operations (2 GFLOP), and $2 \cdot n \cdot 8B = 16GB$ data transfered from memory, then:
$$t_{comp} = \frac{2 \:\:GFLOP}{384\:\:FLOPS} = 5.2ms \:\:\:\:\:\:\: t_{mem} = \frac{16\:\:GB}{51.2\:\:GB /s} = 312.5ms$$

If we **overlap computation data transfer**, a lower bound of execution time is:
$$t_{exec} \geq \max{(t_{comp},t_{mem})} = 312.5 ms$$
The achievable performance is: $\frac{2\:\:GFLOP}{312.5 ms} = 6.4\:GFLOPS$ (less then 2% of peak compute performance).
In this case, the dot product kernel, is a **memory bound** in the example platform, it's limited by the data transfer rate.
# References
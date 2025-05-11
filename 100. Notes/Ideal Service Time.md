**Data time:** 19:01 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Ideal Service Time

We define the **ideal service time** of process Q, denoted by $T_{id-Q}$, as the average time interval between the beginning of the processing of two consecutive stream items. It is composed of two components:
- **Average Calculation Time** to process a generic input $T_{calc-Q}$
- **[[Inter Time|Inter-Process]] Communication** $L_{com}$ to send the result onto the output stream

The complete formula is:
$$T_{id-Q} = T_{calc-Q} + L_{com}\:\:\:\:\:T_{calc-Q}= T_{calc-0} + T_{miss}$$
Where $T_{calc-0}$ is called **pure calculation time** and $T_{miss} = N_{miss} \cdot T_{line}$ is **cache misses** time.

- The pure calculation time depends on the characteristics of the processor’s micro-architecture
- The impact of cache misses depends on both micro-architectural features and algorithmic features of the problem
###### Cache Exploitation
Sequential programs might exhibit two properties regarding
cache exploitation
- **Locality**: if a program accesses one memory address, there is a good chance that it will also access other nearby addresses shortly
- **Reuse**: if a program accesses one memory address, there is a good chance that it will access the same address again

For example, in the [[Parallelization methodology and metrics|example above]], in the for loop we have locality. Theoretically there is a infinity loop and we repeat all many times for this reason we have also reuse on the locations of the array A.

Let us assume an **L1-L2 cache hierarchy** on the processore **on-demand caches** (no prefetching data)
- if **capacity(L1) > size(A)** we have $T_{miss} \sim0$

- if **capacity(L1) < size(A) && capacity(L1) > size(A)** we have $T_{miss} \sim L \cdot \frac{4}{\sigma} \cdot L_{L2-L1}(\sigma)$
$L_{L2-L1}(\sigma)$ is the time to transfer a cache line from L1 to L2. $L \cdot \frac{4}{\sigma}$ the size of array L times 4 bytes (integer are 4 bytes) divided by the size of a cache line ($\sigma$)

- if **capacity(L2) < size(A)**: $T_{miss} \sim L \cdot \frac{4}{\sigma} \cdot L_{M-L1}(\sigma)$
Its equal to case 2 but now we have the transfer time from L1 to main memory
# References
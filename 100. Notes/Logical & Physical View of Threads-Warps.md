**Data time:** 10:48 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Execution Model]]

**Area**: [[Master's degree]]
# Logical & Physical View of Threads-Warps

- **Logical perspective of a kerne**l: a grid can be 1D, 2D or 3D of blocks, each block can be 1D, 2D or 3D of threads
- **Hardware perspective of a kernel**: all threads are organized as a large 1D array with progressive identifiers

**Example**: a block of 128 threads will be divided into 4 warps (threads 0-31 belong to the first warp)

**Example**:
![[Pasted image 20250531104936.png|600]]

A kernel might have a huge amount of blocks (we can have $2^{31}-1$ blocks in the **x-dimension** and, $2^{16}-1$ in the y- and z- dimensions. Blocks of a grid (kernel) are distributed to the SM of the device. Blocks allocated on a SM are called **resident**. They are assigned to resources of the SM such as **registers** and **shared memory** depending on the requirements of their code. The number of blocks can be greater than the total number of resident blocks on the GPU (so, some are **non-resident**)

![[Pasted image 20250531105541.png]]

### Resident Warps
All warps of a resident block are themselves **resident**, i.e., ready to run on the SM. **Warp instructions** of resident blocks are scheduled onto the SM cores by running each instruction on a group of 32 cores. Some warps are currently **running** on the GPU cores while others are **eligible** to run in the next clock cycles. Both running and eligible warps are resident, and consume SM resources.

![[Pasted image 20250531105745.png | 550]]


### Resource Limits
The actual number of blocks and warps that can be resident on a given SM for a given kernel depends on:
- the amount of registers and shared memory used by the kernel
- the amount of registers and shared memory available on the SM

**Example** (old models, not updated): $Occupacy = \frac{\#resident\_warp}{\#max\_no\_resident_warps}$

![[Pasted image 20250531111159.png | 600]]

![[Pasted image 20250531111218.png | 600]]

Therefore, special care must be taken during the kernel development to properly tune the amount of resources used by **threads (registers)** and **blocks (shared memory)**. Otherwise, we might impair the actual parallelism exploited by the device since the actual number of resident blocks (and therefore warps) can be shrinked compared to the ideal case (**low occupancy**).

![[Pasted image 20250531111614.png]]

What you write in the kernel has a profound impact on the overall performance achieved
# References
**Data time:** 20:24 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Accesses to GEMM

Some GPU kernels are limited in performance by the **GMEM bandwidth**. Therefore, optimizing the exploitation of the GMEM bandwidth (i.e., several hundreds of GiB/s in modern devices) is pivotal.

**Goal**: the general goal is to **minimize** the number of transactions needed to serve the **number of GMEM accesses** required by our kernel.

It is important to remind that:
- Each thread of the same warp presents a distinct access the GMEM for a different addresss (in general), which can be served by **one or more memory transactions per warp**
- The number of required transactions per warp depends on the size of the read data per thread, and on **how addresses requested by threads are distributed**

For this reason, it is important to identify and understand different **memory access patterns to GMEM**, and which ones are ideal from the hardware viewpoint.

### GMEM Warp-based Transactions
GMEM accesses go through the L1 and L2 (or the L2 only in some cases). **Constant memory accesses** go through the read-only and L2 caches. **Local memory accesses** are routed through the L1 and L2. **Shared memory acccesses** do not go through any cache.

**Transactions** involving L1-L2 are of 128 bytes (i.e., the L1 cache line size), transactions involving the L2 only are of 32 bytes (useful to reduce over-fetching for scattered data)

![[Pasted image 20250601202858.png]]

- Each thread can request 1, 2, 4, 8 or 16 bytes with a single LOAD/STORE
- This translates in 32 up to 512 bytes per warp
- More transactions might be needed to serve all requests coming from the threads in a warp (ie 1, 2, 4)
- Minimizing such a number of transactions is pivotal

Combining multiple accesses by different threads of the same warp in a set of transactions depends on multiple factors:
- **Alignment**: the first address of a transaction must be a multiple of **128 bytes (L1)** or **32 bytes (L2)**
- **Coalescing**: the whole set of bytes read from the threads of a warp should represent a **contiguous data block** (i.e., no holes)

**LOAD Example** (each thread reads 4 bytes)

![[Pasted image 20250601203420.png | 550]]

##### Cached Loads (L1-trans. 128B)

![[Pasted image 20250601203709.png | 600]]

##### Cached Loads (L2-trans. 32B)
Sometimes, it might be useful to **disable the L1 cache** and go directly through the L2. To do that, we can compile with the flags: `-Xptxas -dlcm=cg`. In this case a **LOAD example** (each thread reads 4 bytes)

![[Pasted image 20250601203833.png | 600]]

##### GMEM Stores
The L1 cache is never used for memory writing operations (**STORE operations** are performed **directly to L2** before being transmitted to global memory). STOREs are performed on a **32-bytes granularity (segment)**. **Example**: threads write 4 bytes each, up to 4 segments can be involved in one transaction (if aligned and coalesced). One transaction might include 1, 2 or 4 segments

![[Pasted image 20250601203951.png | 600]]

# References
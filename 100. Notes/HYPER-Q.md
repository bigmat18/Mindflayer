**Data time:** 12:44 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Concurrency and Streams]]

**Area**: [[Master's degree]]
# HYPER-Q

On old **[[Fermi Architecture (2010)|FERMI]]** GPUs (2010), up to **16 kernels** can be run concurrently on the same device. Multiple host threads can issue GPU tasks on different CUDA streams. However, such activities are enqueued in a **single hardware queue** on the device.

![[Pasted image 20250602124149.png | 200]]

```C
for (int i=0 ; i<3 ; i++) {
	A<<<gdim, bdim, smem, streams[i]>>>();
	B<<<gdim, bdim, smem, streams[i]>>>();
	C<<<gdim, bdim, smem, streams[i]>>>();
}
```

- Using CUDA streams, we have declared the dependency chains
- GPU overlapping can occur only at the stream edges. So, only green tasks in the figure below can be run in parallel owing to the unique centralized hardware queue (others are serialized due to so-called **false dependencies**)

![[Pasted image 20250602124311.png | 400]]

Kepler introduces the **Grid Management Unit (GMU)**, which is equipped with **multiple hardware work queues** to eliminate or reduce false dependencies. With the GMU, streams can be kept as individual pipelines of work.

For the code above:
![[Pasted image 20250602124533.png | 200]]
- The **Hyper-Q** technology is incorporated within the GMU in such a way to handle multiple hardware queues
- On Kepler GPUs, we have up to **32** hardware queues
- The hardware allocates each CUDA stream to a queue. If more than 32 CUDA streams are used, more are mapped onto the same queue (leading again to **false dependencies**)

![[Pasted image 20250602124629.png]]

##### Example
Each kernel is launched as a single thread, which simply executes a loop for a defined amount of time and saves the total number of clock cycles in GMEM.
###### Profiling without HYPER-Q

```c
for (int i=0 ; i < nstreams ; i++) {
	kernel_A<<<1, 1, 0, streams[i]>>>(&d_a[2*i], time_clocks);
	total_clocks += time_clocks;
	kernel_B<<<1, 1, 0, streams[i]>>>(&d_a[2*i+1], time_clocks);
	total_clocks += time_clocks;
}
```

On devices without Hyper-Q, concurrency only between pairs of `kernel_B` from `stream[N]` and `kernel_A` from `stream[N+1]`

![[Pasted image 20250602125050.png]]

###### Profiling with HYPER-Q
With **Hyper-Q**, full concurrency between kernels and elimination of false dependencies between kernels of different CUDA streams.

![[Pasted image 20250602125223.png | 550]]


# References
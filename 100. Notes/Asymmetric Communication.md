**Data time:** 14:30 - 05-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Message Passing Interface (MPI)]]

**Area**: [[Master's degree]]
# Asymmetric Communication

Asymmetric communication is also called **collective communication**. Some examples are: one-to-many or one-to-all, many-to-one or all-to-one, all-to-all.

Higher efficiency than separate point-to-point communication. From a functional standpoint, collectives are not necessary. They can be simulated with point-to-point communications

MPI provides an **extensive set of collective operations**. All processes in a communicator must call the same collective operation.
### Barrier
```c
int MPI_Barrier(MPI_Comm comm)
```
Wait until all processes in the communicator reach the barrier

```c
int MPI_Ibarrier(MPI_Request* req)
```
Non-blocking version. All collectives have a non-blocking version

![[Pasted image 20250605180352.png | 250]]

Possible implementations: A process enters the arrival phase and does not leave until all others have arrived. Then they move to the departure phase.

![[Pasted image 20250605180438.png]]

### Broadcast
```c
int MPI_Bcast(void* buf, int count, MPI_Datatype dt, int source, MPI_Comm comm)
```
Sends the same message to all processes in the communicator (**one-to-all**)

![[Pasted image 20250605180648.png | 500]]

Each participant in a one-to-all broadcast calls the broadcast primitive (even though all but the root are receivers), **`source`** is the rank that originate the data (often the root, i.e., rank 0, of the group)

![[Pasted image 20250605180737.png | 400]]

### Reduction
```c
int MPI_Reduce(void* sndbuf, void* rcvbuf, int count, MPI_Datatype dt, MPI_Op op,
			   int target , MPI_Comm comm)
```
Performs a reduction and place the result in one `target` process (**all-to-one**)

![[Pasted image 20250605181621.png | 500]]

###### MPI_Op predefined Reduction operations

![[Pasted image 20250605181714.png | 300]]

To use `MPI_MINLOC`/`MPI_MAXLOC` in a reduce operation the datatype argument must be a pair (value and index). MPI provides the following predefined datatypes:

![[Pasted image 20250605181745.png | 350]]

###### Example of `MPI_MAXLOC`/`MPI_MINLOC`
![[Pasted image 20250605181832.png | 270]]

```c
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Seed the random number generator differently for each process
    srandom(rank + 1); // Adding 1 or some other value to rank can be good practice

    // This structure corresponds to the predefined MPI type MPI_2INT
    // MPI_2INT is a pair of (int, int) where reduction operations
    // like MPI_MAXLOC and MPI_MINLOC operate on the first int (value)
    // and use the second int (index/rank) as a tie-breaker or to identify the location.
    struct my2INT {
        int val; // The value to compare
        int idx; // The index (often the rank of the process that owns the value)
    } in, out;

    in.val = random() % 100; // Generate a random value between 0 and 99
    in.idx = rank;           // Store the rank of the current process

    std::printf("Process %d has value %d and index %d\n", rank, in.val,
                in.idx);

    // Perform a reduction operation.
    // MPI_Reduce combines data from all processes in the communicator
    // and places the result in the 'out' buffer on the root process.
    // &in: send buffer (data from this process)
    // &out: receive buffer (only significant on the root process)
    // 1: count of elements to send/receive (1 struct my2INT)
    // MPI_2INT: MPI datatype corresponding to our struct
    // MPI_MAXLOC: operation to find the maximum value (in.val) and the
    //             index (in.idx) of the process that holds that maximum value.
    //             If values are equal, the process with the smaller rank is chosen.
    // 0: rank of the root process that will receive the final result
    // MPI_COMM_WORLD: communicator
    MPI_Reduce(&in, &out, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    // After the MPI_Reduce, the 'out' struct on the root process (rank 0)
    // will contain the maximum value found among all processes and the
    // rank of the process that had that maximum value.
    if (rank == 0) { // Only the root process (rank 0) prints the result
        std::printf(
            "The root process (0) reports: Maximum value is %d, found at index (rank) %d\n",
            out.val, out.idx);
    }

    MPI_Finalize();
    return 0;
}
```

###### Example of `MPI_REDUCE`
Trapezoid rule for numerical integration using `MPI_Reduce`() (Example already implemented using
[[Symmetric Communication|point-to-point communications]])

```c
int main(int argc, char *argv[]) {
    const double a = 0.0;
    const double b = 1.0;
    const int n = 10000000;
    double partial_result;
    double result = 0.0;
    int myrank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    partial_result = trap(myrank, size, a, b, n);
    MPI_Reduce(&partial_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (!myrank) {
        std::printf("Area: %.18f\n", result);
    }

    MPI_Finalize();
    return 0;
}
```

### [[Scatter]]
```c
int MPI_Scatter(void* sndbuf, int sndcount, MPI_Datatype snddt,
				void* rcvbuf, int rcvcount, MPI_Datatype rcvdt, int source , MPI_Comm comm)
```
Sends p-1 data blocks from the **source** process to all other processes in the communicator (**one-to-all**)

![[Pasted image 20250605182242.png | 500]]
### [[Gather]]
```c
int MPI_Gather(void* sndbuf, int sndcount, MPI_Datatype snddt,
			   void* rcvbuf, int rcvcount, MPI_Datatype rcvdt, int target , MPI_Comm comm)
```
The **target** receives p-1 data blocks from all processes in the communicator (**all-to-one**)

###### Example Map: Vector Sum
We want to execute a parallel map computation for the sum of two vectors of N elements
![[Pasted image 20250605182504.png | 450]]

With MPI using p processes: the root process **partitions** (**Scatter** operation) the A and B vectors to all p
processes, each computing a local sum (i.e., a local C vector of about $\left\lfloor  \frac{n}{p} \right\rfloor$ elements). Then, the root
process collects all local vectors into the final result vector (**Gather** operation). In the first version $n\mod p = 0$

```c
int main(int argc, char *argv[]) {
    int n = 5000; // default if no argument provided
    int myrank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (2 == argc) {
        n = std::stoi(argv[1]);
    }

    if (!myrank && (n % size)) {
        std::fprintf(
            stderr,
            "ERROR: the vector lenght (%d) must be multiple of the communicator size(%d)\n",
            n, size);

        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    std::vector<double> A, B, C; 
    if (!myrank) { // root allocates the vectors
        A.resize(n);
        B.resize(n);
        C.resize(n);
        for (int i = 0; i < n; ++i) {
            A[i] = static_cast<double>(i);
            B[i] = static_cast<double>(n - 1 - i);
        }
    }

    /* All nodes (including the master) allocate the local vectors */
    int localn = n / size;
    std::vector<double> localA(localn);
    std::vector<double> localB(localn);
    std::vector<double> localC(localn);

    // Scatter vector A
    MPI_Scatter(A.data(),    // sndbuf (significant only on root)
                localn,      // sndcount; the partition size sent to each process
                MPI_DOUBLE,  // snddatatype
                localA.data(), // rcvbuf
                localn,      // rcvcount (elements received by each process)
                MPI_DOUBLE,  // rcvdatatype
                0,           // root
                MPI_COMM_WORLD);

    // Scatter vector B
    MPI_Scatter(B.data(), localn, MPI_DOUBLE, localB.data(), localn,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // All processes compute the localC vector
    for (int i = 0; i < localn; i++) {
        localC[i] = localA[i] + localB[i];
    }

    // Gather results from all processes
    MPI_Gather(localC.data(), // sendbuf
               localn,        // sendcount
               MPI_DOUBLE,    // senddatatype
               C.data(),      // recvbuf (significant only on root)
               localn, // recvcount; elements received from each process
               MPI_DOUBLE, // recvdatatype
               0,          // root
               MPI_COMM_WORLD);

	// check
    if (!myrank) {
        for (int i = 0; i < n; i++) {
            if (std::fabs(C[i] - (static_cast<double>(n - 1))) > 1e-6) {
                std::fprintf(stderr,
                             "Test FAILED: C[%d]=%f, expected %f\n", i,
                             C[i], static_cast<double>(n - 1));
                MPI_Abort(MPI_COMM_WORLD, 0);
            }
        }
        printf("Result OK\n");
    }

    MPI_Finalize();

    return 0;
}
```

We can do a second version with $n\mod p \neq 0$ 

```c
int MPI_Scatterv(void* sndbuf, int sndcounts[], int displs[], MPI_Datatype snddt,
				 void* rcvbuf, int rcvcount, MPI_Datatype rcvdt, int source , MPI_Comm comm)
```

```c
int MPI_Gatherv(void* sndbuf, int sndcounts[], MPI_Datatype snddt,
				void* rcvbuf, int rcvcounts[], int displs[], MPI_Datatype rcvdt, 
				int target , MPI_Comm comm)
```
Gaps are allowed between distinct partitions in the source data; partitions may have different message sizes, and the distribution order can be any:

![[Pasted image 20250605183002.png | 550]]

**NOTE**: the entries in the displacement array (**displs**) are measured in extents of the datatype argument, not in bytes.

### All directives
##### `Allgather`/`Allgatherv`
```c
int MPI_Allgather(void* sndbuf, int sndcount, MPI_Datatype snddt,
				  void* rcvbuf, int rcvcount, MPI_Datatype rcvdt, MPI_Comm comm)
int MPI_Allgatherv(void* sndbuf, int sndcount, MPI_Datatype snddt,
				void* rcvbuf, int rcvcount, int displs[], MPI_Datatype rcvdt, MPI_Comm comm)
```

![[Pasted image 20250605183125.png | 550]]

#####  `Allreduce`
```c
int MPI_Allreduce(void* sndbuf, void* rcvbuf, int count, MPI_Datatype dt,
				  MPI_Op op, MPI_Comm comm)
```

![[Pasted image 20250605183235.png | 550]]
###### Example: Power Iteration
**Power Iteration** is one of the simplest numerical algorithms for computing a single eigenvalue-
eigenvector pair , specifically the pair associated with the eigenvalue of largest magnitude (dominant
eigenpair, i.e., $\lambda_{max}, x_{max}$)

![[Pasted image 20250605183508.png | 350]]

The algorithm is sketched in the box above. The parallelization is straightforward: Iterative Map-Reduce computation:
- Partitioning A across processors and broadcasting the initial $x_0$
- Each processor compute 𝐴𝑥 producing a chunk of 𝑦
- `Gatherall` to collect all local chunks into the full y = 𝐴𝑥
- Normalize 𝑦 to produce $x_{k+1}$
- Compute $\lambda$ through the Rayleigh quotient using the normalized vector
- Check convergence $||x_{k+1} - x_{k}|| < \epsilon$

##### `Alltoall`/`Alltoallv`
```c
int MPI_Alltoall(void* sndbuf, int sndcount, MPI_Datatype snddt,
				void* rcvbuf, int rcvcount, MPI_Datatype rcvdt, MPI_Comm comm)
int MPI_Alltoallv(void* sndbuf, int sndcount, int sdispls[], MPI_Datatype snddt,
		void* rcvbuf, int rcvcount, int rdispls[], MPI_Datatype rcvdt, MPI_Comm comm)
```

![[Pasted image 20250605183329.png | 550]]
Each process performs a scatter operation (analogous to a matrix transpose)

### Parallel prefix operations
```c++
int MPI_Scan(void* sndbuf, void* rcvbuf, int count, MPI_Datatype dt,
			 MPI_Op op, MPI_Comm comm)
```
Inclusive scan operation: process 𝑖 compute $result = op(v_0, v_i)$
```c
int MPI_Exscan(void* sndbuf, void* rcvbuf, int count, MPI_Datatype dt,
			   MPI_Op op, MPI_Comm comm)
```
Exclusive scan operation: process 𝑖 compute $result = op(v_0, v_{i-1})$, $v_0$ is the identity element of the op

![[Pasted image 20250605183928.png | 550]]

##### MPI_Scan (inclusive)

![[Pasted image 20250605183958.png | 600]]

```c
int main(int argc, char *argv[]) {
    int myrank;
    int localn = 3; // Number of elements in the local vector for each process

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Create a local vector for each process
    std::vector<int> localData(localn);

    // Initialize localData: values depend on the rank and index
    // P0: [0, 1, 2]
    // P1: [3, 4, 5]
    // P2: [6, 7, 8]
    // etc.
    for (int i = 0; i < localn; ++i) {
        localData[i] = i + myrank * localn;
    }

    // Create a vector to store the result of MPI_Scan
    std::vector<int> scan(localn);

    // Perform the MPI_Scan operation.
    // MPI_Scan computes an inclusive prefix reduction (scan).
    // For each process j, the result in scan.data() is the reduction
    // (MPI_SUM in this case) of localData.data() from all processes
    // with rank 0 up to and including j.
    // The operation is element-wise if 'count' (localn) > 1.
    MPI_Scan(localData.data(), // Send buffer: input data from this process
             scan.data(),      // Receive buffer: output of the scan for this process
             localn,           // Count: number of elements in send/receive buffers
             MPI_INT,          // Datatype of elements
             MPI_SUM,          // Operation: sum the elements
             MPI_COMM_WORLD);  // Communicator

    // Print the results of the scan operation for each process
    for (int i = 0; i < localn; ++i) {
        std::printf("rank=%d scan[%d]=%d\n", myrank, i, scan[i]);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
```

# References
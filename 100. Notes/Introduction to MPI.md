**Data time:** 13:33 - 05-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Message Passing Interface (MPI)]]

**Area**: [[Master's degree]]
# Introduction to MPI

$p$ processes, each with its private address space. All data must be explicitly **partitioned** and **distributed**. All interactions among processes are two-sided:
- The process that has the data sends it (e.g., `MPI_send`)
- The process that needs the data receives it (e.g., `MPI_receive`)

![[Pasted image 20250605133733.png | 200]]

**Strengths** are:
- Relatively simple performance model
- Offers high performance by co-locating data with computation
- General model (portability) that can be used in all systems

**Challenges**: explicit coordination can increase programming complexity 

Two primary mechanisms are needed:
1. A method of creating separate processes for execution on different computers:
	- **MPMD**: Multiple-Program Multiple-Data
	- **SPMD**: Single-Program Multiple-Data
2. A method of sending and receiving messages: Send and receive can be **synchronous** or **asynchronous**

### MPMD Model
Separate programs for each executable. Processes cooperate typically through a connection-oriented library (e.g., POSIX socket API).

![[Pasted image 20250605134052.png | 300]]

### SPMD Model
**Different logical flows merged into one program**. Control statements select different parts for each processor to execute. All executables started together. SPMD is the default execution model in **MPI** (MPI also supports the MPMD model).

![[Pasted image 20250605134154.png | 300]]

### Message Passing in MPI
MPI is a message-passing interface specification. It is a library interface/specificantion and not a language. It has multiple implementations: OpenMPI, Intel MPI, MPICH, MVAPICH ecc. MPI goals are **portability**, **ease-of-use**, and **efficiency**. MPI-1 was released in 1994. MPI-4.1 in 2023. 

MPI has language bindings:
- C, Fortran (officially supported)
- C++ (deprecated as of MPI-2.2 and removed in MPI-3.0, still provided by several implementations). You must compile MPI with `–with-cxx-bindings` to enable C++ bindings
- Other bindings: C#, Python, Rust, Go, ….

**No shared variables** (MPI-3 introduced Shared-Memory Windows For processes mapped on the same node). Communication and Synchronization operations are **MPI library functions**:
- **Communication**: Point-to-Point, Collectives
- **Synchronization**: Explicit barrier, Implicit synchronization bound to communication collective execution
- **Other functions** for querying the MPI library runtime:
	- How many processes are taking part in the computation?
	- Which rank do I have?
	- Is that specific communication completed?

Simple to get started. Simple programs use only six library routines:
- **`int MPI_Init(int* argc, char** argv)`**: initializes MPI
	- No MPI calls before this call
	- Parse the command-line striping off MPI arguments
	- Initialize the MPI environment
- **`int MPI_Finalize()`**: terminates MPI
	- No MPI calls after this call
	- Performs clean-up to terminate the MPI environment
- **`MPI_Comm_size`**: determines the number of processes in the initial group (`MPI_COMM_WORLD`)
- **`MPI_Comm_rank`**: determines the ID of the calling process in the group
- **`MPI_Send/MPI_Isend`**: sends a message (blocking/non-blocking send)
- **`MPI_Recv/MPI_Irecv`**: receives a message (blocking/non-blocking receive)

The same program is executed by all P processes (**SPMD model**). Each process chooses a different
execution path depending on its ID (called **rank** in MPI). The ID is unique for each process in 0…P-1

```c
…
MPI_Init(…); // no MPI call before this point
myFunction1(); // executed by all processes
switch(myID) {
	case 0: foo1(); break; // executed by process 0 only
	case 1: foo2(); break; // executed by process 1 only
	default: foo3(); break; // executed by all other processes
}
myFunction2(); // executed by all processes
MPI_Finalize(); // no MPI call after this point
…
```
The possible **Return codes** are `MPI_SUCCESS` or `MPI_ERROR`.

###### Example: Hello World
We use the **communicator** `MPI_COMM_WORLD`: contains all application processes. The **Rank**: `myId` is the process identifier within the `MPI_COMM_WORLD` group containing `numP` processes.

```C++
int main (int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int numP; 
    MPI_Comm_size(MPI_COMM_WORLD, &numP);

    // Get processor name
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    MPI_Get_processor_name(processor_name, &namelen);

    // Get the ID of the process (rank)
    int myId;
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    // Every process prints Hello
    std::printf("From process %d out of %d, running on node %s: Hello, world!\n",
                myId, numP, processor_name);

    // Terminate MPI
    MPI_Finalize();
    return 0;
}
```

**Compilation**:
```
// compiling on the master node
mpicxx -Wall -O3 hello_world.cpp -o hello_world
// compiling on a cluster node using SLURM
srun -n 1 make hello_world
```
**Execution**:
```
salloc -N 4 // suppose SLURM gives us node[01-04]
mpirun -n 4 ./hello_world // directly using mpirun
srun --mpi=pmix -n 4 ./hello_word // using SLURM
// 2 processes on host node01, 1 on node03 and node04
mpirun -n 4 --host node01,node01,node3,node4 ./hello_world
```

![[Pasted image 20250605140936.png | 500]]

### Sending and Receiving Messages
There are the following interfaces:
```c
int MPI_Send(void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm)

int MPI_Recv (void* buf, int count, MPI_Datatype dt, int source, int tag,
			  MPI_Comm comm, MPI_Status* status)
```

Source and destination are the IDs (ranks) of the processes in the communicator **comm**
- **Receiver** source wildcard `MPI_ANY_SOURCE`. Any process in comm can be a source

The message TAG is an integer value, 0 ≤ tag < `MPI_TAG_UB`
- **Receiver** TAG wildcard `MPI_ANY_TAG`

The receiver has no partial reads and message `size ≤ buffer length` specified, otherwise `MPI_ERR_TRUNCATE` error.

- **`MPI_STATUS`**
	- Stores information about an `MPI_Recv` operation
	- It contains the source (`MPI_SOURCE`), tag (`MPI_TAG`), and error (`MPI_ERROR`) of the communication. Therefore, a process can check the actual source and tag of a message received with `MPI_ANY_SOURCE` and/or `MPI_ANY_TAG`
	- t is possible to use `MPI_STATUS_IGNORE` if no information is needed
- **`int MPI_Get_count(MPI_Status* status, MPI_Datatype dt, int* count)`**
	- Returns in count of how many elements of a given MPI datatype (dt) have been received
	- `MPI_Recv` may complete even if fewer than count elements have been received if the matching `MPI_Send` sent fewer elements

These are the primitive data types. Custom data types can be constructed by using specific MPI functions.
![[Pasted image 20250605142455.png | 400]]

###### Example of Send/Receive
```c
int main(int argc, char *argv[]) {
    char message[20];
    int myrank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) { /* process zero */
        strcpy(message, "Hello, there");
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 1, 99, MPI_COMM_WORLD);
    } else if (myrank == 1) { /* process one */
        MPI_Recv(message, 20, MPI_CHAR, 0, 99, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);
        printf("received: %s (MPI_CHAR received %d)\n", message, count);
    }

    MPI_Finalize();
    return 0;
}
```

#### MPI Tags
Used to differentiate between different types of messages being sent. The TAG is used to let the receiving process identify the message. The message tag is carried within the message.

If special type matching is not required, a wildcard message tag (`MPI_ANY_TAG`) can be used so that the receive operation will match with any send operation.
- If the receiver gets a message with a different TAG than that expected by the `MPI_Recv()` call, the message is kept on hold and will be matched by a future `MPI_Recv()` with a matching TAG

### Communication Taxonomy
###### [[Symmetric Communication|Symmetric or point-to-point or 1:1]]
###### [[Asymmetric Communication|Asymmetric or collective]]


# References
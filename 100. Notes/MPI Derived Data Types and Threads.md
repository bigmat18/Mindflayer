---
Data: 2026-01-03T15:28:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Message Passing Interface (MPI)]]"
Area: "[[Master's degree]]"
---
# MPI derivated data types and threads

## Derived Data Types
MPI allows users to build new, **user-defined datatypes based on primitive MPI datatypes**. Derived datatypes are useful in situations where:
- the data to send is non-contiguous data (e.g., sparse vector elements)
- the data is contiguous in memory but made of mixed types (e.g., a struct containing an integer and an array of floats)
	- Such kind of data can be sent using `MPI_CHAR/BYTE`. However, using a specific type improves program readability

Such kind of data can be sent using MPI_CHAR/BYTE. However, using a specific type improves program readability. A general datatype is an opaque object that specifies two things:
- A sequence of basic datatypes
- A sequence of integer (byte) displacements

Order of items need not coincide with their order in memory. An item may appear more than once

#### Building a datatype
The primary steps are:
1. Construct the datatype using a constructor, for example `MPI_Type_vector()`, The new datatype has type `MPI_Datatype`
2. Allocate the datatype using `MPI_Type_commit()`
3. Once the datatype is not used anymore, free it `MPI_Type_free()`

Construction and allocation is mandatory, the release of the datatype is optional but recommended
```c++
MPI_Datatype new_type; // declare the new_type datatype
…
MPI_Type_vector(…, &new_type); // construct new_type
MPI_Type_commit(&new_type); // allocate new_type
…
MPI_Type_free(&new_type); // free new_type
```
#### Datatype constructors
The most used datatypes constructors are:
- `int MPI_Type_indexed(int count, int blocklen[], int displacements[], MPI_Datatype oldtype, MPI_Datatype* newtype)`

##### Contiguous datatype
```
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype* newtype)
```

The **newtype** is the datatype obtained by concatenating **count** copies of **oldtype**
![[Pasted image 20260103153445.png | 300]]

##### Vector datatype
```c++
int MPI_Type_vector(int count, int blocklen, int stride, MPI_Datatype oldtype, MPI_Datatype* newtype)
```

Create a datatype from elements of an existing data type:
- **count** is the number of blocks
- **blocklen** is the number of elements in each block
- **stride** is the number of elements between the start of each block
	- to use bytes instead of the number of elements, the function `MPI_Type_hvector()` can be used

![[Pasted image 20260103153725.png | 550]]

##### Structure datatype
```c++
int MPI_Type_create_struct(int nblocks, int blocklen[], MPI_Aint displacements[], MPI_Datatype oldtype[], MPI_Datatype* newtype)
```

Create a datatype from a set of datatypes. Each block is a collection of data of the same type:
- **nblocks** is the number of blocks
- **blocklen** is an array of integers setting the size of each block
- **displacements** is an array setting the relative offset for each block (expressed in bytes!)
	- Set displacements manually is unsafe and not portable, use **MPI_Get_address()** 

```c++
struct Particle {
 float x;
 float y;
 int type;
} p;

int blocklen[]= {2, 1}; // 2,1 -> two float, one int
MPI_Datatype oldtypes[] = {MPI_FLOAT, MPI_INT};
MPI_Aint_displ xaddr, typeaddr, displs[2];
MPI_Get_address(&p.x, &xaddr); MPI_Get_address(&p.type, &typeaddr);
displ[0] = 0; displ[1] = typeaddr – xaddr;
MPI_Type_create_struct(2, blocklen, displs, oldtypes, &new_type);
MPI_Type_commit(&new_type);
```

###### Example: Jacobi iterations
![[Pasted image 20260103154302.png | 600]]

**Proc1** needs the last row from **Proc0** and the first row from **Proc2** to compute the stencil on the elements of the first and last rows of its partition. **Proc0** needs the first row from **Proc1**. **Proc2** needs the last row of **Proc1**.

#### Overlapping computation & communication
- **no overlap** (best case with blocking communications) $T_C = T_{calc} + T_{comm}$
![[Pasted image 20260103155957.png | 300]]

- **overlap** (best case with non-blocking communications) $T_C = \max(T_{calc}, T_{comm})$
![[Pasted image 20260103160019.png]]

Performance of the Jacobi parallelization with blocking and non blocking communications on system with 64 cores. Matrix dimensions are 4096x4096 and error threshold is 0.1

![[Pasted image 20260103160147.png]]

#### Complex Communicators: SUMMA
**S**calable **U**niversal **M**atrix **M**ultiplication (**SUMMA**). The matrices A, B, and C are distributed among processes without redundancy. Not all data to compute the partial product are stored in  the local memory of processes 
$$
C01 = A00 \times B01 + A01 \times B11 + A02 \times B21
$$
Rows and Columns group communicators are used to obtain all data needed locally (through broadcast phases)

![[Pasted image 20260103160754.png | 300]]

Processes organized in a 3 × 3 grid: 3 broadcast phases (in general $p$ phases for a grid $p\times p$) 
![[Pasted image 20260103160847.png | 650]]

Horizontal broadcast for A, vertical broadcast for B. Splitting `MPI_COMM_WORLD`:
```c++
// Create the communicators by splitting MPI_COMM_WORLD
MPI_Comm rowComm;
MPI_Comm colComm; 
// color key 
MPI_Comm_split(MPI_COMM_WORLD, myId/gridDim, myId%gridDim, &rowComm)
MPI_Comm_split(MPI_COMM_WORLD, myId%gridDim, myId/gridDim, &colComm)
```

![[Pasted image 20260103160943.png | 400]]

## Threads and MPI
MPI describes parallelism between processes with separate address spaces. MPI does not support threads by default:
- e.g. OpenMPI needs to be compiled with `--enable-mpi-threads`
- If no thread support, only one thread per process executes

Threads in MPI are not addressable:
- It is not possible to send a message to a specific thread of a process

To init MPI to support threads `MPI_Init_thread` **must be used** instead of `MPI_Init`. The MPI process can use threads for computation and/or communication
- e.g., MPI+OpenMP, MPI+C++ threads, MPI+FastFlow, MPI+CUDA, etc
- The paradigm MPI + “X” is known as **hybrid programming**
- NOTE: X could also be MPI Shared Memory (introduced in MPI-3, for true on-node shared-memory regions

```c++
 int MPI_Init_thread(int* argc, char*** argv, int required, int* provided)
```

The argument required is used to request the desired level of thread support. It can be:
- **MPI_THREAD_SINGLE**, only one thread calling MPI functions will execute. NOTE: all other threads of the process must sleep during MPI function calls.
- **MPI_THREAD_FUNNELED,** only the thead that called MPI_Init_thread may make MPI calls, e.g., if OpenMP is used, only the Master thread executes MPI function calls.
- **MPI_THREAD_SERIALIZED**, only one thread will make MPI function calls at one time. In this case, multiple threads may execute MPI function calls with the restriction that calls are serialized
- **MPI_THREAD_MULTIPLE**, multiple threads may call MPI functions with no restrictions

MPI_THREAD_SINGLE < MPI_THREAD_FUNNELED < MPI_THREAD_SERIALIZED < MPI_THREAD_MULTIPLE

The MPI runtime fills the provided argument with the actual thread support offered by the implementation.  NOTE: could be less than required! `MPI_Init` is equivalent to `MPI_Init_thread(…, MPI_THREAD_SINGLE, …)`

When multiple threads make MPI calls concurrently, the outcome will be as if the calls are executed sequentially in some order.
- User must ensure that collective operations are correctly ordered among threads (cannot call broadcast in one thread and scatter in another thread on the same communicator)
- User must ensure data race free MPI  program

With `MPI_THREAD_MULTIPLE`, the MPI library must allow concurrent calls from multiple threads and guarantee that one thread’s blocking call cannot starve progress in another thread. In the example,  no matter which thread runs first, the two sends and two receives will always match and complete.

![[Pasted image 20260103162007.png | 400]]

This does not come for free. The implementation must protect certain code and data structures with mutexes and this usually introduces overhead.

###### Example MPI-[[Introduction to OpenMP|OpenMP]] 
Pay attention to process-node bindings and OpenMP threads affinity. Command to run:
```
mpirun -x OMP_NUM_THREADS=16 -x OMP_DISPLAY_AFFINITY=true --bynode --bind-to none 
 --report-bindings -n 8 ./trapezoid_mpi+omp 200000000
```

NOTE: env variables can be passed to the application when launching it with srun through the --export command line option. For example:
```
srun --mpi=pmix --export=“ALL,OMP_NUM_THREADS=16” --bind-to none ./trapezoid_mpi+omp 2000000000
```
# References
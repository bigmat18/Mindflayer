**Data time:** 13:02 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]]

**Area**: [[Master's degree]]
# Level-based view for parallel computing

## Vertical structuring
The functional of a system can be organised in a hierarchy of **interpretations layers** (or **virtual machines**).
Each level provides a set of functionalities for upper levels ad hides the implementation of the lower levels.

![[Screenshot 2024-09-22 at 15.15.33.png | 150]]
###### Virtual machines
The hierarchy is structured with a language-based approach, where each level $MV_i$ are interpreted by programs written at level $MV_j$ with $j < i$.

###### Runtime system (RTS)
If COM is a command of language $L_i$ of $MV_i$ the interpreter, or **run-time support**, of COM will be denoted by RTS(COM)

![[Screenshot 2024-09-22 at 15.17.11.png | 250]]

### Interpreters
An interpreter is a program that translate the input program **one statement at a time** and replace each instance of a command with the same implementation. The **optimizations** are possibile but they are more difficult because are runtime.

![[Screenshot 2024-09-22 at 15.22.16.png | 300]]

### Compilers
The compiler analyzes the input program **statically** and creates a traslated version in the target language $L_j$ with $j < i$ in a single step. More **optimizations** are possibile. In compilers exist multiple versions of RTS(COM), it chooses the best depending on how COM is used in the **surrounding context** (it's called **static analysis**)

![[Screenshot 2024-09-22 at 15.24.59.png | 300]]

### Interpreters vs Compilers
Interpretation and Compilation are the two basic approaches to implement a hierarchy of levels.

![[Pasted image 20250511131314.png | 400]]

###### Example 
This is a possibile optimization example performed by **compiler**:
![[Pasted image 20250511131430.png | 600]]

- **Example 1**: there is 3 read (4 bytes each) operation per iterations (A[i], B[i], X[i]) and 1 write operations (X[i])
- **Example 2**: we use a temporary variable for x is initialized and allocated in a register and only at the end of the for loop the value is written in memory

This allow to save 2 memory access each iterations and **alleviate memory contention**.

## Level-based View

![[Pasted image 20250511132440.png | 500]]

#### Application Level
Different **high-level formalism** to express parallelism and parallel processing at the application level.

![[Pasted image 20250511133351.png | 550]]
In this two examples we use two high level formalism for parallelism, **OpenMP** and **SkePU**. With this two library we don't create threads, divide things or anything else, these operation are doing below.
#### Operating System Level
This level represents the **runtime support** of processes and threads. It implements:
- **Process** and **threads** implementation
- Inter-process Communication mechanism (**IPC**)
- Thread **synchronization**
- Distributed-memory **communication primitives**
- **Inter-process communication** runtime

![[Pasted image 20250511133142.png | 200]]

Such functionalities are often implemented in kernel mode, although this yields some extra overhead. In some cases, not in the figure above (e.g., RDMA verbs), communication primitives are in user space only.

Same example of the high-level parallel programs written in **OpenMP** and **SkePU** developed now with raw **pthreads**:
```c
struct task {
	int *v1;
	int *v2;
	int *v3;
	int size;
};

//Thread code
void *thread_func(void *ptr)
{
	pthread_barrier_wait(&barrier);
	struct task *t = (struct task *) ptr;
	for (int i=0; i<t->size; i++) {
		(t->v3)[i] = (t->v1)[i] + (t->v2)[i];
	}
}

int main(int argc, char **argv)
{
	…
	//Create array of threads with size p (a parameter)
	pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * p);

	// Create and fill task array
	struct task *t = (struct task *) malloc(sizeof(struct task) * p);
	size_t offset = L/p;
	for (int i=0; i<p; i++) {
		t[i].v1 = v1 + i*offset;
		t[i].v2 = v2 + i*offset;
		t[i].v3 = v3 + i*offset;
		t[i].size = offset;

		// Pass task filled to a create thread
		pthread_create(&(threads[i]), NULL, thread_func, (void *) &(t[i]));
	}
	//Wait all threafs termitation
	for (int i=0; i<p; i++) {
		pthread_join(threads[i], NULL);
	}
	…
}
```

#### Assembler Level
The assembler language is a humanly-understandable representation of the **machine code** (Architecture Level). It will be optioned after compiling and it has a **imperative structure**.
#### Firmware Level
Firmware level provides interpretation of the executable version of assembler instructions. We model the firmware level as a collection of interconnected **firmware units**. Each unit is **autonomous**, ie, with its self-control capabilities and executing a fixed **micro-program**. This level work ad physical machine level. 

![[Pasted image 20250511135301.png | 600]]

The processor is a potentially highly-parallel component in charge of itnerpreting assembler instructiions. For example:
1. **P** asks the **MMU** to fetch the instructions whose logical address is in the program counter (**PC**)
2. The operating code of the 32-bit instruction is used to execute the right micro-program subpart
3. At the end of the micro-program P checks the present of **interrupts**, if not present update PC and go to 1
# References
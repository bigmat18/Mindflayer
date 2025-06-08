**Data time:** 11:46 - 05-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Parallel Programming with OpenMP]]

**Area**: [[Master's degree]]
# OpenMP task-parallelism

Even if loops are the main sources of parallelism, not all programs have easily parallelizable loops:
- Loops in which the iteration count is not known in advance
- Loops containing complex dependencies (e.g., LU, Cholesky algorithms)
- Divide and Conquer algorithms
- Data streaming computations (i.e., producer-consumer pattern)

In OpenMP a **`task`** is an **independent unit of work** composed of:
- a function code to execute
- (references to) needed data and in general, its data environment (shared and private)
- control variables

```c++
#pragma omp parallel
{
	#pragma omp single
	{
		for(ListNode *n=List->head; n, n=n->next) {
			#pragma omp task
			{
				do_something(n);
			}
		}
	}
}
```

The execution of a task by a thread of the team can begin immediately or **can be deferred**. For example, a task can be inserted in a task queue to be executed by any thread in the team. 

By default, a task is **«tied»** to the thread that begins executing it. For tied task, once a thread starts executing the task, the thread must complete it. The thread cannot suspend the task and resume it later on a different thread.

Creating tasks with the **`task`** directive is more flexible than sections
- Sections are defined statically, while tasks can be dynamically created
- Starting from `OpenMP 4`, it is possible to specify `depend(in/out/inout)` clauses on tasks to orchestrate task dependencies

### Execution Model
1. When a team thread encounters a **`task`** directive, it may choose to execute the task immediately or defer its execution
2. If the task execution is deferred, **the task is placed in a task queue associated with the current parallel region**
3. All threads of the team, will take tasks out of the task pool until the pool is empty
4. The code associated with the task construct will be executed only once
5. A **tied** task is executed by the same thread from beginning to end
6. An **untied** task **can be rescheduled**, it can start on one thread, get suspended, and resume on another
```
#pragma omp task [clause list]
```
Some clauses that can be used in `[clause list]`:
- **`if (scalar expression)`**: determines whether the task is deferred (if true) or executed immediately (if false)
- **`final (scalar expression)`**: if the expression evaluates to true the task will not have descendants
- **`untied`**: no scheduling restrictions for the task. A task, by default, is tied to the first thread that executes it.
- **`private (variable list)`**: specifies variable local to the child task
- **`firstprivate (variable list)`**: similar to `private`, the variables are initialized to variable value before the parent directive (copy)
- **`shared (variable list)`**: specifies that variables are shared with the parent task
- **`default (data scoping specifier)`**: default data handling specifier may be `shared` or `none`
- **`depend(kind: variable list)`**: specifies dependencies between tasks. The kind can be in, out, or inout, defining that the task reads, writes or reads and writes the variables

###### Example
![[Pasted image 20250605123337.png | 600]]

### Data Scoping for tasks
Data scooping for tasks is tricky if the scope is not explicitly specified. If no default clause is specified, then:
- Static and global variables are **`shared`**
- Local variables are **`private`**
- Variables for orphaned tasks (those executing after the spawning thread exits its region) are **[[OpenMP loop-parallelism|firstprivate]]**. This is because the execution of a task can be deferred, and when the task will be executed, variables may have gone out of scope.
- Variables for non-orphaned tasks are **`firstprivate`** by default unless the **`shared`** clause is specified in the enclosing context
```c
int main() {
	int x =1;
	int y = 2;
	#pragma omp parallel private(y)
	{
		int z = 2;
		#pragma omp task
		{
			int w = 3;
			// x shared, y and z firstprivate, and w private
			// NOTE: y is not initialized!
			F(x,y,z,w);
		}
	}
}
```
Variables from enclosing context are **`firstprivate`** by default in tasks.

### Wait for the completion of tasks
The thread **`barrier`** waits for all threads to arrive, and each thread also waits for the completion of its own child tasks. This is also true for implicit barriers at the end of a **parallel** region. However, since child tasks are executed separately from the generating tasks, it is possible that a child task gets executed after the generating task has finished.

The **`taskwait`** directive can be used to wait for the completion of child tasks defined in the current task (direct child tasks)
- It waits for the generated tasks, but not for all descendant tasks
- The **`taskwait`** suspends the parent task until all its child tasks have finished
```
#pragma omp taskwait
```

The **`taskgroup`** directive defines a region in which any tasks created become part of that group. **At the end of the taskgroup region, there is an implicit barrier**. The **`taskgroup`** waits for all tasks generated within its structured block (including nested tasks).
```
#pragma omp taskgroup
```

### Task Generation
The typical pattern for task generation is:
```c
#pragma omp parallel
{
	…
	#pragma omp single
	{
		for(int i=0; i < NTASKS; ++i)
			#pragma omp task
			{
				processTask(…);
			}
	} // single
	…
} // parallel
```
If **`NTASKS`** is very large, the implementation can stop generating new tasks and switch all threads in the team to execute the already generated tasks. If the thread that generated the tasks is executing a long task, the other threads might not have anything to do. 

One option is to make the task generation an untied task, thus any other thread is eligible to resume the task generating the loop:
```c
…
#pragma omp single
{
	#pragma omp task untied
	for(int i=0; i < NTASKS; ++i)
		#pragma omp task
		{
…
```

###### Example: Fibonacci
It computes the n-th Fibonacci number using the exponential algorithm. The task is the fib(n) function. `n1` and `n2` must be **`shared`**, otherwise the default would be **`firstprivate`**.

```c++
long fib(long n) {
    long n1, n2;
    if (n < 2) return 1;

#pragma omp task shared(n1)
    n1 = fib(n - 1);
#pragma omp task shared(n2)
    n2 = fib(n - 2);
    // Wait for the two tasks to complete
#pragma omp taskwait
    return n1 + n2;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::printf("Use %s n #threads\n", argv[0]);
        return -1;
    }
    long n = std::stol(argv[1]);
    long nth = std::stol(argv[2]);

    long result;
    // creating a pool of nth threads
#pragma omp parallel num_threads(nth)
    {
        // only the Master thread starts the recursion
#pragma omp master
        result = fib(n);
    }
    std::printf("fib(%ld) = %ld\n", n, result);
}
```

The **`taskwait`** at line 13, suspends the parent task until both the two children tasks complete. If we had not used **`taskwait`**, `n1` and `n2` would have got lost. To stop recurring the clause **`final(n <= THRESHOLD)`** could be used at line 8 and 10.

### Task Scheduling
Task scheduling refers to the mechanism by which the OpenMP runtime system decides which tasks created by the program are executed and when they are executed by the available threads. 

Clauses such as the following influence the scheduling behavior:
- **`if`**: to decide whether a construct should be executed as a task or inline
- **`final`**: to mark a task as non-deferrable
- **`priority`**: to influence the scheduling order
- **`untied`**: to allow tasks to be resumed on a different thread

OpenMP defines the following task scheduling points:
- When a task is created (`#pragma omp task`)
- When encountering `taskwait`, or `taskyield`
- At explicit or implicit barriers
- When a task completes execution
- When exiting a **`taskgroup`** region

**Tied tasks** (only the thread to which the task is tied may execute the task) is a task can be suspended at specific scheduling points (creation, completion, `taskwait`, `barrier`, …). If a task is not suspended at a barrier, the thread executing the task can only switch to a descendant of any task tied to the thread.

**Untied tasks** (the task can be executed by different threads). There are no scheduling restrictions. The tasks can be suspended at any point, and the thread can switch to any task.
### Task Loop
The **`taskloop`** construct is designed for parallelizing loops using tasks:
- Automatically generates tasks for loop iterations, enabling dynamic scheduling
- Useful when dynamic scheduling is needed or iteration count is not predictable

Decompose a loop into chunks and create a task for each chunk. The overhead with respect to `#pragma omp parallel for` is generally higher due to task creation
```
#pragma omp taskloop [clause list]
```

Some clauses in `[clause list]`:
- **`grainsize(grain-size):`** chunks have at least a grain-size iterations
- **`num_tasks(num-tasks)`**: create num-tasks chunks, each must have at least one iteration
- They are mutually exclusive

If neither `grainsize` nor `num_tasks` is used, the number of chunks and iterations per chunk is implementation dependent.

```c++
#pragma omp parallel
{
	#pragma omp single
	{
		#pragma omp taskloop
		for(int i=0; i < N; ++i)
			do_something(i);
	}
}
```
# References
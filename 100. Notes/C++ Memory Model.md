---
Data: 2026-02-02T21:30:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Lock-Free Programming in C++]]"
Area: "[[Master's degree]]"
---
# C++ Memory Model

The compiler may reorder instruction for performance reasons independently from the low-level
micro architectures. It is generally accepted that a compiler can reorder ordinary reads from and writes to memory almost arbitrarily, provided the reordering cannot change the observed **single-threaded** execution of the code. 

**Modern C++** (from C++11) and also the Java language, define a memory model that guarantees SC for DRF programs. 
- The compiler will insert proper synchronization to cope with the HW memory model.
- No guarantees if the program contains data races. Unsynchronized programs contain data races, i.e., the output of the program depends on the relative speed of processors  executing threads (**non-deterministic program results**) 

C++11 introduced, at the language level, atomic operations and fences to specify how memory ordering
works for both atomic and non atomic operations. This helps programmers control visibility and ordering of memory accesses across threads, making the program portable across different micro architectures.

C++ provides the user with six memory ordering options for atomic types:
1. `std::memory_order_relaxed` 
2. `std::memory_order_consume` 
3. `std::memory_order_seq_cst`
4. `std::memory_order_acquire`
5. `std::memory_order_release`
6. `std::memory_order_acq_rel`

**Load**, **Store**, and **Read-Modify-Write (RMW)** operations accept explicit ordering values. The default memory ordering is the stricter one: [[Sequential Consistency (SC)]] (`std::memory_order_seq_cst`)
- **Store** operations can be relaxed, release, or `seq_cst`
- **Load** operations can be relaxed, (consume), acquire, or `seq_cst`
- **RMW** operations can be relaxed, (consume), acquire, release, `acq_rel` or `seq_cst`

Each ordering imposes different constraints on how operations can be reordered around that atomic operation

**Note** for the consume: `std::memory_order_consume`, is unimplemented by most compilers the C++17 standard advises against using it because of the difficulty in implementing it correctly and consistently

Types of memory ordering:
- [[Sequential Consistency (SC)]] loads/stores atomic operations executed in the order specified by the program
- **Acquire**, only valid for atomic **load** operations. It prevents all loads/stores in the current thread from being moved to a position **before** the current atomic load operation
- **Release**, only valid for atomic **store** operations. It prevents all loads/stores in the current thread from being moved to a position **after** the current atomic store operation
- **Relaxed**, no restrictions in memory reordering of surrounding load/store operations. Only the atomicity of  the actual operation is guaranteed
- **Acquire-Release,** only valid for atomic exchange, compare-exchange, and fetch_* operations (i.e. RMW operations). It is a combination of the Acquire and Release modes, simultaneosly loads the current value in  Acquire mode and stores a new value in the atomic variable in Release mode. It ensures no subsequent operations can move before it, and no prior operations can move after it
### SC Total Ordering

The **SC** semantics requires a single total ordering over all atomic operations with ordering `std::memory_order_seq_cs`

![[Pasted image 20260203031645.png]]

The assert at line 37 can never fire, because either the store to x  (line 9) or the store to y (line12) must happen before at least one  of the loads on x or y (we don’t know the order)

If the load at line 16 returns false, it means that the store to x at line 9 must happen before the store to y at line 12 and thus the  load at line 21 must return true
- The other way around is also valid, if the load at line 21 returns false, the  load at line 16 must return true

SC is the most expensive memory ordering, it requires global synchronization between all threads (enforced by the C++  compiler).

### Relaxed Ordering
Under std::memory_order_relaxed, each atomic operation is guaranteed to be atomic, but no additional ordering constraints are imposed.
- With relaxed ordering there is no synchronization among atomic operations

```c++
std::atomic<bool> x,y;

std::atomic<int> z;

void write_x_then_y() {
	x.store(true,std::memory_order_relaxed);
	y.store(true,std::memory_order_relaxed);
}



  
void read_y_then_x() {
	while(!y.load(std::memory_order_relaxed));
	if(x.load(std::memory_order_relaxed))
		++z;
}

int main() {
	x=false;
	y=false;
	z=0;
	std::thread a(write_x_then_y);
	std::thread b(read_y_then_x);
	a.join();
	b.join();
	assert(z.load()!=0); // it can fire!
}
```

In this new code, the assert at line 27 **can** fire
- The load at line 15 can return false even if the load at line 14 returns true
- This means one thread could see the store to y happen before the store to x, while the other thread sees the store to x but not 

### Acquire and Release Ordering
With acquire-release ordering the synchronization is **pairwise** between the thread that does the **release** and the thread that does the **acquire**. Different threads can see different ordering.
- A release operation synchronizes-with an acquire operation that reads the value written.

```c++


std::atomic<bool> x,y;

std::atomic<int> z;


void write_x() {
	x.store(true,std::memory_order_release);
}
void write_y() {
	y.store(true,std::memory_order_release);
}
void read_x_then_y() {
	while(!x.load(std::memory_order_acquire));
	if(y.load(std::memory_order_acquire))
		++z;
}

void read_y_then_x() {
	while(!y.load(std::memory_order_acquire));
	if(x.load(std::memory_order_acquire))
		++z;
}

int main() {
	x=false;
	y=false;
	z=0;

	std::thread a(write_x);
	std::thread b(write_y);
	std::thread c(read_x_then_y);
	std::thread d(read_y_then_x);
	a.join();
	b.join();
	c.join();
	d.join();
	assert(z.load()!=0); // it can fire!
}
```

In this new code, the assert at line 37 **can** fire because the load at line 15 synchronizes only with the store at line 9, and the load at line 20 synchronizes only with the store at line 12. The ordering from the release to the acquire in each case has no effect on the operations in the other threads

### Relaxing SC with Acquire and Release Ordering
In code aside, we obtained the same effect of SC but with less costly synchronizations. 

![[Pasted image 20260203032655.png]]

The load from y, at line 13 synchronizes with the store at line 10.The store to x at line 9 happens before the store to y (i.e., such store cannot be moved after the store to y) and thus happens before the load from y at line 13, which happens before the load from x at line 14 (the load from x cannot be moved before the load from y). 
- **to provide any synchronization, acquire and release operations must be paired up** (in our example is fundamental that the load at line 13 is made in a loop, if not, they are not paired up)

### [[Safety Nets (fences)|Fences]]
Fences are operations that affect the ordering of other atomic operations **without modifying any data**
- Fences are typically used for ordering `memory_order_relaxed` atomic operations.

C++11 provides `std::atomic_thread_fence` (memory order) 
- **Release fence**: prevents store to move after the fence 
- **Acuire fence**: prevents load to move before the fence 
- **Acq_rel fence**: combines the previous two, it is a full barrier

Fences with atomic operations can also be used to order nonatomic operations like in the code snippet here

![[Pasted image 20260203034341.png]]

The store at line 10 happens before the load at line 17 because of the fences. In fact, the store at line 10 cannot bypass the release fence,  which synchronizes with the acquire fence at line 16, and the load at  line 15 cannot bypass the acquire fence.

### [[Basic Spin-Lock|Spin-Lock]]
Simple implementation of spin lock using C++20 features. It uses an `atomic_flag` (a boolean flag, operations on this type are required to be lock-free)
- There are no operations other than **test_and_set** and **clear**.
- C++20 provided also **test, wait, notify_one/all**

![[Pasted image 20260203035047.png]]

It tries to acquire the lock at line 11. If it fails, the lock method spins using the test (line 12) until it sees a change, and then it retries to acquire the lock. The **clear** method (line 16) sets the `atomic_flag` to false

### [[Barriers|Barrier]] and Spin-Barrier
The `std::barrier` is a synchronization primitive introduced in C++20 that allows a fixed number of threads to wait until all have reached a certain point in the code.

The barrier is reusable, so it can be used in iterative or phased algorithms where you need multiple barrier points across iterations.

When you expect that the waiting time at the barrier to be very short, a busy-waiting implementation (spin) may 
have less overhead that a blocking barrier because it avoids the overhead of context switches
- Blocking barriers usually put threads to sleep (using OS-level synchronization)

If the wait times can be longer or are unpredictable, using a blocking barrier (such as `std::barrier`) is generally 
preferable because it lets the OS suspend waiting threads, freeing up CPU resources for other work.

### The ABA problem
Atomics-based data structures might be faster to use than locks-based data structures. However, atomics-based code features the so-called **ABA problem**. 

The ABA problem occurs when a location's value changes from A to B and then back to A. A thread that reads A later cannot detect that an intermediate change occurred. **This is a typical problem of pointer-based lock-free data structure**

Detailed steps leading to the **ABA problem**:
1. Thread 1 reads on atomic variable, x, and finds it has value A.
2. Thread 1 performs some operation based on this value, such as dereferencing it (if it's a pointer) or  doing a lookup, or something.
3. Thread 1 is stalled by the operation system.
4. Another thread performs some operations on x that change its value to B.
5. A thread then changes the data associated with the value A such that the value held by thread 1  is no longer valid. This may be as drastic as freeing the pointed-to memory or changing an associated value
6. A thread then changes x back to A based on this new data. If this is a pointer, it may be a new object that happens to share the same address as the old one.
7. Thread 1 resumes and perfomrs a compare/exchange on x, comparing against A. The compare/exchange succeeds (because the value is indeed A), but this is the wrong A value. The data originally read at step 2 no longer valid, but thread 1 has no way of telling and will corrupt the data structure

##### ABA Example
![[Pasted image 20260203040659.png]]

```c++
// typical pseudocode of a pop operation of a pointerbased lock-free stack
Node* pop() {
 Node* current = head;
 while (current) {
 if (head == current && 
 CAS(&head, current, current->next)) break;
 current = head;
 }
 return current;
}

```

1. Thread 1 tires to pop the head node, but it is suspected just before executions the CAS (just after reading next read)
2. Thread 2 starts and pops the head node (1)
3. Thread 2 pops the new head node (2)
4. Thread 2 pushes the new node (containing 4), reusing the node that contained the initial head node (ie, 1)
![[Pasted image 20260203040932.png]]

5. Thread 1 resumes, and its CAS succeds because the current pointer metches ((raw pointers looks the same even though the content has changed), thus removing the actual head node (4) and setting the head to a node that is no longer in the list (potentially deleted) 

![[Pasted image 20260203040959.png]]

##### Solution to ABA
The common approach to avoid the ABA problem is to add an extra tag (version number) to the data. **The tag is incremented every time the pointer is updated**. The CAS operations works on both the pointer and the tag as a single atomic operation.

**This way the CAS will fail even if the address is the same**
- If the pointer was changed twice (A→B → A), the tag would have changed

It requires CAS2 (double-word CAS, not available on all architectures)
- Additionally, the tag number may need to be very large

Another option is to implement deferred reclamation
- Prevent node reuse while pending requests exist
- **Hazard pointers** are often used for safe reclamation of memory

**Hazard pointers** address the problem of safely freeing or reusing nodes that might still be  accessed by other threads.
- Each thread before reading or modifying a shared node, “publicly announce” it is using that node using a per-thread hazard pointer list
- If a pointer appears in any hazard pointer list, it cannot be reclaimed
- Once the thread is done with the node, it removes the pointer from the list. If no other thread has that pointer in its hazard list the node can be safely freed. 


# References
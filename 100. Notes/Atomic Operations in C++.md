---
Data: 2026-02-02T20:24:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Lock-Free Programming in C++]]"
Area: "[[Master's degree]]"
---
# Atomic Operations in C++

C++11 introduces **atomic data types** that can be safely manipulated in a concurrent context without  acquiring locks (i.e., `std::mutex`). Operations on atomic data types are executed entirely or not al all (i.e., **atomic operations are indivisible**). In many scenarios, atomic operations c**an be faster than using mutexes**, particularly under high concurrency

C++ atomics enable lock-free (non-blocking) programming, which avoid:
- Thread suspension/resume overhead
- **Deadlock** conditions
- **Priority inversion** issues
	- a thread with high priority is waiting to acquire a lock that is currently held by a low-priority thread

Lock-free programming avoids certain synchronization issues (e.g., deadlock, priority inversion) but introduces new challenges:
- In general, lock-free programming is more complex and introduce problems related to **starvation** and **livelock**
	- **Livelock**: threads continuously interfere with each other in a way there is no real progress
	- Careful design is necessary to avoid subtle data races, ensure correctness and progress

Three main progress guarantees classes:
- **Wait-freedom** (strongest guarantee): Every thread completes its operation **in a finite number of steps**, independently of other threads. No starvation for any thread
- **Lock-freedom**: Guarantees that at least one thread makes progress. Possible starvation for some threads
- **Obstruction-freedom**: Guarantees progress only if a thread runs without contention

C++ atomic operations typically provide lock-free guarantees (`std::atomic::is_lock_free()` to check)

###### Example: Atomic counting
C++ atomic operations typically provide lock-free guarantees (std::atomic::is_lock_free() to check). concurrent increment of a variable using both mutexes and atomics (spmcluster front-end node).

```c++
std::mutex mutex;
std::vector<std::thread> threads;

const uint64_t num_threads = 10;
const uint64_t num_iters = 100'000'000;

auto lock_count =
[&] (uint64_t& counter, const auto& id) {
	for (uint64_t i = id; i < num_iters; i += num_threads) {
	std::lock_guard<std::mutex> lock_guard(mutex);
	counter++;
	}
};

auto atomic_count =
[&] (std::atomic<uint64_t>& counter, const auto& id) {
	for (uint64_t i = id; i < num_iters; i += num_threads) {
	//counter++;
	//counter.fetch_add(1);
	std::atomic_fetch_add(&counter,1);
	}
};

TIMERSTART(mutex_multithreaded)
uint64_t lock_counter = 0;
threads.clear();

for (uint64_t id = 0; id < num_threads; id++)
	threads.emplace_back(lock_count, std::ref(lock_counter), id);

for (auto& thread : threads)
	thread.join();

TIMERSTOP(mutex_multithreaded)

TIMERSTART(atomic_multithreaded)

std::atomic<uint64_t> atomic_counter(0);

threads.clear();

for (uint64_t id = 0; id < num_threads; id++)
	threads.emplace_back(atomic_count, std::ref(atomic_counter), id);

for (auto& thread : threads)
	thread.join();

TIMERSTOP(atomic_multithreaded)
std::cout << lock_counter << " " << atomic_counter << std::endl;
```

In this example, the use of atomics is approximately 6 times more efficient than the traditional approach using locks:
![[Pasted image 20260202203319.png | 350]]

### `std:atomic<T>`

`std::atomic` is neither copyable nor movable. T must be a trivially copiable type, Operations (i.e., the methods of the std::atomic class):
- **load/store**: to get and set the content of a std::atomic
- **exchange**: atomically replace the value and return the old  value of the variable
- **compare and exchange:** it does an atomic exchange only if the value is equal to the provided expected value
- **fetch operation**s: perform a read-modify-write in one  atomic step
- **wait/notify** (from C++20) behavior similar to condition variables

![[Pasted image 20260202203519.png]]

**Example:** we could have  implemented the dynamic scheduling policy using  an atomic variable instead of a mutex-protected  plain variable

C++11 provides built-in atomic support for 8, 16, 32, or 64 bits wide integers and pointers. However, we can wrap structs and objects of  different length with `std::atomic<T>`. The  compiler realizes their concurrent manipulation with locks
- The code you write remains correct even if the underlying HW does not natively support the corresponding data type
- The member function `is_lock_free()` can be used to check whether the object manipulation is done with atomics or mutexes

![[Pasted image 20260202204325.png | 500]]

### Compare-And-Swap (CAS) 
Every C++ atomic data type features a CAS (Compare-and-Swap) operation for implementing
arbitrary atomic assignments. Two methods: `compare_exchange_strong` and `compare_exchange_weak`
- `compare_exchange_weak` is more efficient but might suffer from spurious fails (it may return false even if the  comparison yields true)
![[Pasted image 20260202204511.png | 500]]

Semantics:
1. Compare the value **expected** with the value stored in the atomic
2. If yes, then it sets the **atomic** to the value **desired**, otherwise writes the actual value stored in atomic into expected
3. Return true if the swap operation in step 2 was successful, false otherwise

CAS operations should always be performed in loops.

###### Example: Parallel Max-Reduction using CAS
Compute the maximum of a sequence of 64-bit integers. `false_max` (lines 14-19) occasionally computes the 
incorrect results since the condition tested in the if and  the assignment (line 17) are two independent operations
- Two or more threads might have read the same value before executing the assignment in a random order

![[Pasted image 20260202205130.png]]

`correct_max`: uses a CAS in which the read and write operations are atomically executed

![[Pasted image 20260202205256.png]]


# References
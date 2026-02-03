---
Data: 2026-02-02T20:09:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Workload Balancing in C++]]"
Area: "[[Master's degree]]"
---
# Implementing a ThreadPool in C++

Instead of creating threads on-demand for each task, we can maintain a pre-spawned set of worker threads that are reused to process all the submitted tasks. There is no out-of-the-box implementation of a Thread Pool (TP) in the C++ standard library. **The idea is to submit tasks to the thread pool's queue, which returns a std::future**. The worker threads execute the tasks and fulfill the associated promises, allowing the results to be retrieved asynchronously
```c++
ThreadPool TP(8); // 8 Workers in the pool
// function executed by the generic Worker
auto square = [](uint64_t x) { return x*x; }; 
std::vector<std::future<uint64_t>> futures;
for (uint64_t x = 0; x < N; ++x) 
	futures.emplace_back( TP.enqueue(square, x) );
```
- The Workers threads compete to access the shared concurrent task queue
- Each Worker can execute any submitted task (i.e., Workers are not specialized, but generic executors)
- The synchronization is managed through mutex and condition variable, following the Producer-Consumer synchronization pattern
- In the provided implementation (threadPool.hpp) the task queue is unbounded
	- What happens if **producers generate tasks much faster than Workers can process them**? The tasks queue can grow excessively, leading to high memory usage or even memory exhaustion
	- To avoid this issue we can: **Increase the number of Worker threads** (if we have enough available cores) to match the producer(s) rate or **Impose a limit to the task queue** size, blocking or throttling producers when the queue is full 

### Selecting the Partition/Task size
Choosing the task size (or partition size in static distributions) is critical for achieving good parallel  performance. The right size balances workload distribution against overhead introduced by task scheduling and synchronization.
- **Too small**: tasks incur high overhead due to frequentsynchronization and scheduling
- **Too large**: minimize overhead but risks poor load balancing leading to idle threads
- **Optimal task size** is a trade-off between minimizing overhead and maximizing workload balancing

![[Pasted image 20260202201810.png | 400]]


# References
**Data time:** 17:32 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Concurrency in C++]]

**Area**: [[Master's degree]]
# Multiple-Reader Single-Writer Pattern

Classical concurrency pattern involving two types of threads: **readers** and **writers**:
- A reader may access the data concurrently with other readers
- A writer needs exclusive access to the shared data
- How does this differ from the Producer-Consumer pattern? In [[Producer-Consumer Pattern]] both producer and consumer modify the data structure, in reader and writer one read and one write.

Using a plain **`std::mutex`** to protect the accesses to the critical section **prevents** the concurrent access of multiple readers. To implement the pattern, the C++17 standard provides **`std::shared_mutex`** and **`std::shared_timed_mutex`**.

No guarantees about fairness:
- A continuous stream of readers can potentially starve a writer
- If priority or fairness between readers and writers is required to avoid potential starvation, it must be explicitly implemented by the programmer

```c++
int main() {
	int shared_counter = 0;
	std::shared_mutex smutex; // it supports both exclusive write and non-exclusive read
	
	// reader's function
	auto reader = [&](int count, int id) {
		for(int i=0;i<count; ++i) {
			// shared_lock for non-exclusive access
			std::shared_lock<std::shared_mutex> lock(smutex); 
			std::printf("Reader%d has read %d\n", id, shared_counter);
		}
	};
	
	// writer's function
	auto writer = [&](int count, int id) {
		for(int i=0;i<count; ++i) {
			// unique lock to gain exclusive access
			std::unique_lock<std::shared_mutex> lock(smutex); 
			++shared_counter;
			std::printf("Writer%d has written %d\n", id, shared_counter);
		}
	};
	
	// Create multiple reader threads
	std::vector<std::thread> readers;
	for (int i = 0; i < 5; ++i) {
		readers.emplace_back(reader, 4, i);
	}
	
	// Create multiple writer threads
	std::vector<std::thread> writers;
	for (int i = 0; i < 3; ++i) {
		readers.emplace_back(writer, 3, i);
	}
	
	// Wait for all threads to finish.
	for (auto& reader : readers) reader.join();
	for (auto& writer : writers) writer.join();
	
	return 0;
}
```
# References
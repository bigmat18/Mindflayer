**Data time:** 17:32 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Concurrency in C++]]

**Area**: [[Master's degree]]
# Producer-Consumer Pattern

Classical concurrency pattern involving two types of threads: **producers**, and **consumers**:
- **Producers**: generate data items and add them to a shared data structure (usually a queue that can have bounded or unbounded capacity)
- **Consumers**: retrieve data items from the shared data structure (usually one at a time) and consume them doing some work
Let’s see an implementation in which the buffer is unbounded in size, implemented through **`std::deque`** container, and the classical implementation with mutex and condition variables.
##### Example: Alarm Clock
In the code snippet a thread executes the student function, which checks, in a passive-wait loop, a global variable (`time_for_breakfast`). The master thread, sets the global variable after a while and notifies the student thread (line 38).

Remember to use wait on a CV either in a while loop or, alternatively, using the wait method with a predicate:

![[Pasted image 20250602214757.png | 400]]
###### Wrong Implementation
```c++
int main() {
    std::mutex mutex;
    std::condition_variable cv;
    bool time_for_breakfast = false; // globally shared state
    // to be called by thread
    auto student = [&] ( ) -> void {
        {   // this is the scope of the lock
            std::unique_lock<std::mutex> unique_lock(mutex);

            // check the globally shared state
            while (!time_for_breakfast) {
				std::this_thread::sleep_for(1s); // <=== isert to see error				
                // lock is released during wait
                cv.wait(unique_lock);
			}

        }
        std::cout << "Time to make coffee!" << std::endl;
    };

    // create the waiting thread and wait for 2s
    std::thread my_thread(student);
	std::this_thread::sleep_for(0.5s); // <=== insert to see error	

	time_for_breakfast = true;

    cv.notify_one();
    // wait until breakfast is finished
    my_thread.join();
}
```
This is an incorrect implementation of the alarm clock example. The sleep_for calls at lines 20 and 31 have been inserted purposely to make it stall. What’s the problem? Why it stalls? **Race condition** because in the main we dont use a lock.
###### Right Implementation
```c++
int main() {
    std::mutex mutex;
    std::condition_variable cv;
    bool time_for_breakfast = false; // globally shared state
    // to be called by thread
    auto student = [&] ( ) -> void {
        {   // this is the scope of the lock
            std::unique_lock<std::mutex> unique_lock(mutex);

            cv.wait(unique_lock,
                    [&](){ return time_for_breakfast; });			
        } // lock is finally released
        std::cout << "Time to make coffee!" << std::endl;
    };

    // create the waiting thread and wait for 2s
    std::thread my_thread(student);
    std::this_thread::sleep_for(2s);

    { // prepare the alarm clock
        std::lock_guard<std::mutex> lock_guard(mutex);
        time_for_breakfast = true;		
	} // here the lock is released

	// ring the alarm clock
	cv.notify_one();
	
    // wait until breakfast is finished
    my_thread.join();
}
```
In this implementation we use two lock, one in main and one in thread, in the thread we use 
```c++
cv.wait(unique_lock,[&](){ return time_for_breakfast; })
```
the conditional variable check the predicate every time, if it is false wait, otherwise continute.

# References
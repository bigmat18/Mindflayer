**Data time:** 17:32 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Concurrency in C++]]

**Area**: [[Master's degree]]
# Mutexes and Condition Variables

### Mutex Variables
A mutex is a synchronization mechanism that guarantees mutual exclusion execution of a critical section. Its usage restricts the execution of a critical section to a single thread at a time. A thread locking a mutex prevents other threads from acquiring the mutex. The other threads **wait** for its release (**passive waiting**)

![[Pasted image 20250602213338.png]]

##### `std::lock_guard`
Is a RAII wrapper around std::mutex that acquires ownership of the mutex upon construction and
releases it upon destruction:
- Once constructed, it cannot be explicitly released until it goes out of scope
- It can only manage a single mutex
##### `std::unique_lock`
Is more flexible and supports lock/unlock if needed:
- It can be moved to transfer ownership of the lock
- Often used with condition variables
##### `std::scoped_lock`
Manages multiple mutexes simultaneously, acquiring/releasing all of them simultaneously when entering/exiting the scope.
- It provides an unlock() method to explicitly release the locks before the end of the scope
- Helps avoid deadlock when you need to acquire multiple locks together (using a lock ordering strategy)
- Pay attention to the fact that it accepts 0 mutexes; it simply does nothing

### Condition Variables 
A CV enables one or more threads **to wait (passively) for an event inside a critical section**. Conceptually, a CV is associated with an **event or condition**. When a thread has determined that the condition is satisfied, it can **notify** one or more of the threads waiting on the CV to wake them up.

**Pay attention to spurious wake-ups!** The condition must be checked in a loop (typically a while loop) or the wait with the predicate version should be used.
##### Workflow of the signaling thread
1. The signaling thread has to acquire a mutex using either `mutex.lock()` or through a scoped wrapper such as `std::lock_guard` or `std::unique_lock.`
2. While holding the lock, the shared state is modified.
3. The lock is released either explicitly with `mutex.unlock()`, or implicitly by leaving the scope of `std::lock_guard` or `std::unique_lock`.
4. The actual signaling of the condition variable cv is performed using `cv.notify_one()`, or `cv.notify_all()`
Note that 3 and 4 can be swapped! Slower but usually safer!

##### Workflow of the waiting thread(s)
1. Acquire a **`std::unique_lock`** using the same mutex as in the signaling phase. Note that std::lock_guard cannot be used here.
2. While being locked call either `cv.wait()`, `cv.wait_for(),` or `wait_until()`. The lock is released automatically (**and atomically**).
3. In case (i) cv is notified, (ii) the timeout of `cv.wait()` or `cv.wait_for()` expires, or (iii) a spurious wake-up occurs, then the thread is awakened, and the lock is reacquired. At this point, we must check whether the globally shared state (the condition) indicates proceeding or waiting (i.e., sleeping) again.

### One-shot notifications using futures
CVs are helpful in case many threads perform non-trivial, or repetitive synchronization patterns. Futures and promises can be used for the **so-called one-shot synchronization** scenario.

Synchronization between the issuing thread and the waiting threads is achieved since the access to a future’s values via `f.get()` blocks until the fulfilling of the corresponding promise with `p.set_value()`

The so-called **shared futures** can be used to **broadcast** a specific value to more than one thread.
```c++
int main() {
    // create pair (future, promise)
    std::promise<void> promise;
    auto shared_future = promise.get_future().share();

    // to be called by thread
    auto students = [&] (int myid) -> void {
        // blocks until fulfilling promise
        shared_future.get();
        std::cout << "Student [" << myid <<
			"] Time to make coffee!" << std::endl;
    };

    // create the waiting thread and wait for 2s
    std::thread my_thread0(students, 0);
    std::thread my_thread1(students, 1);

	std::cout << "thread created, waiting for 2s\n";
    std::this_thread::sleep_for(2s);
    promise.set_value();

    // wait until breakfast is finished
    my_thread0.join();
    my_thread1.join();
}
```
# References
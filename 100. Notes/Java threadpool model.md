---
Created: 2024-09-25T00:31:00
tags:
  - "#note"
  - youngling
Links: "[[Networks and Laboratory III]]"
Area:
---
# Java threadpool model

## Why threadpool?

- Frequent activation/eliminations of threads is very expensive operation because we need to interact with JVM and SO
- Large number of task (for example a task for each client on server in a [[Introduction to application layer|client-server]] application)

To mitigate this problem we can use a threadpool where we use same thread to execute many tasks.

## How generally threadpool work?

- We use a queue of tasks that wait execution (use FIFO policy)
- We have a pool of thread to execute tasks
- If a thread is free the system assign a task to a thread
- If all thread are busy
	- or task wait a free thread
	- or we create a new thread

![[Screenshot 2024-09-25 at 00.36.51.png | 400]]

## Java implementation

- In java a task is a class that implements Runnable
- We can create a ThreadPool with `java.util.concurrent` package 
	- define type of queue
	- define parameter for threadpool
- We use the `Executors` class to generate object `ExecutorService` like ThreadPoolExecutor of ScheduledThreadPoolExecutor (this is a [[Factory]] class)

```java
public interface Executor { 
	public void execute (Runnable task) 
} 

public interface ExecutorService extends Executor {.. }
```

```java
public class Task implements Runnable { 
	
	public void run() { 
		// Some code
    }
}
```

### FixedThreadPool

```java
import java.util.concurrent.Executors; 
import java.util.concurrent.ExecutorService; 

public class ExampleFixed{ 
	public static void main(String[] args) { 
		// create the pool 
		ExecutorService service = Executors.newFixedThreadPool(10); 
		//submit the task for execution 
		for (int i =0; i<100; i++) { 
			service.execute(new Task(i)) 
		} 
		System.out.println("Thread Name:"+ Thread.currentThread().getName()); 
	} 
}
```

- A threadpool with default behavior
- It create a fixed number of thread (n is defined when initialize the threadpool)
- When a task T is submited
	- if all thread are busy it will put ina queue
	- If least one thread are free it are used for task
- Use a **LinkedBlockingQueue**
### CachedThreadPool

```java
ExecutorService service = Executors.newCachedThreadPool();
```

- if all threads are busy a new thread is created
- If a thread is free it is used
- If a thread are free per 60 second, it's terminate

![[Screenshot 2024-09-25 at 00.54.17.png | 400]]
### ThreadPoolExecutor
It's the general constructor to create a custom threadpool.

```java
public class ThreadPoolExecutor implements ExecutorService {
	public ThreadPoolExecutor (int CorePoolSize, 
							   int MaximumPoolSize, 
							   long keepAliveTime, 
							   TimeUnit unit, 
							   BlockingQueue workqueue,
							   RejectedExecutionHandler handler)
}
```

- **CorePoolSize**, **MaximumPoolSize**, **KeepAliveTime** control the handle of thread pool. 
- **Workqueue** is the data structure to save the eventually tasks are awaiting the execution

![[Screenshot 2024-09-25 at 01.01.08.png | 400]]

- **Core** is the minimum number of threads that must remain active
	- with `PrestartAllCoreThreads()` all core threads are active at creation
	- **"on demand"** core threads are active at submission
- **KeepAliveTime** for threads not in core is the maximum time that a thread can be active without task (is in milliseconds)

#### Rejection handler
When a task is rejected a **rejection policy** is activated. There are many types of rejection policy to choose during creation:
- **AbortPolicy** run a `RejectionExcutionException`
- **DiscardPolicy, DiscardOldestPolicy, CallerRunsPolicy**  other default policy 
- It's possible create a custom Rejection handler with `RejectExecutionHandler` interface and the `rejectedExecution` method.

```java
import java.util.concurrent.*; 
public class RejectedException { 
	public static void main (String[] args ) {
		ExecutorService service = new ThreadPoolExecutor(10, 12, 120, TimeUnit.SECONDS, 
														 new ArrayBlockingQueue(3)); 
		for (int i=0; i<20; i++) {
			try { 
				service.execute(new Task(i)); 
			} catch (RejectedExecutionException e) {
				System.out.println("task rejected"+e.getMessage());
			} 
		}
	}
}
```
#### Execution lifecycle
- the JVM terminate all threads execution when all no demon threads stop their execution
	- **gradual termination** (`shutdown()`) any tasks are accepted when this method is invoked and all tasks submitted unfinished remain running (included those in the queue)
	- **instant termination** (`shutdownNow()`) any tasks are accepted and the tasks not started are deleted

#### Scheduled executor service
There is the possibility to schedule a task after some time or periodically
- `schedule(Runnable command, long delay, TimeUnit unit)` 
- `scheduleAtFixedRate(Runnable command, long initialDelay, long delay, TimeUnit unit)`
- `scheduleWithFixedDelay(Runnable command, long initialDelay, long delay, TimeUnit unit)`

![[Screenshot 2024-09-25 at 23.20.41.png | 400]]
### Java Blocking queue
There is a several problems to share resource between threads, without any control can be obtain error or inconsistency. To resolve this problems there are many class thread safe like the blocking queue.

- **BlockingQueue** is a JAVA interface that represent a queue
- It's implement a synchronisation 

###### ArrayBlockingQueue
- Fixed size defined in initialization
- Save object in an Array
- Only insert and remove are possible
- Use one lock
###### LinkedBlockingQueue
- Can be limited or unlimited
- Use a LinkedList 
- Possible insert and get correctly

![[Screenshot 2024-09-25 at 23.23.14.png | 400]]

# References
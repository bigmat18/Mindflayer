---
Created: 2024-09-24T23:57:00
tags:
  - "#note"
  - "#padawan"
Links: "[[Networks and Laboratory III]]"
Area:
---
# Java threads

## What is a thread ?

In a process, a thread is a execution flow inside a process (it's a light weight process)
- Threads are less expensive than process
- They can be executed in a single core or in multicore.

![[Screenshot 2024-09-25 at 00.00.47.png | 300]]

## Why multithreading ?

- **Best resource usage:** when a thread is suspended other threads are sent to execution
- **Best performance for computationally intese app**: We can split application in multiple sub-task and execute them in parallel.

But alto they introduced **problems**: hard debugging, deadlock, synchronisation problems


## How use Threads in Java

- In java, when we run an app the JVM run a main thread
- Afterwards, we can create other sub-threads

### Extends Threads

```java
public class ExtendingThread { 
	public static class MyThread extends Thread { 
		public void run() { 
			System.out.println("MyThread running"); 
			System.out.println("MyThread finished"); 
		} 
	} 
	
	public static void main (String [] args) { 
		MyThread myThread = new MyThread(); 
		myThread.start();
	} 
}

Output:
MyThread running
MyThread finished
```

- Overriding of run()
- We create a new class that extends the standard Thread class
- This class memorize a reference to Runnable object, possibly it's passed like paramiter.
- When we call run() if it's redefined invoke the most specific, else it execute the standard run() method in Thread class

#### Start vs run
If we call `thread.run()` instead of `thread.start()` No threads are activated and the run() method is executed in like a normal method in the flow of main thread.

### Implements Runnable

```java
public class ThreadRunnable { 
	public class MyRunnable implements Runnable { 
		public void run() { 
			System.out.println("MyRunnable running"); 
			System.out.println("MyRunnable finished"); 
		} 
	} 
	
	public static void main(String [] args) { 
		Thread thread = new Thread (new MyRunnable()); 
		thread.start(); 
	}
}

Output:
MyRunnable running
MyRunnable finished
```

- It's in `java.language` and contain only the signature
- An instance of Runnable it's a task
	- A fragment of code can be execute on a thread
	- The task creation don't imply the creation of a thread that execute it.

### Stop a thread
- A JAVA program stop when all non demon thread are terminated
- If a thread use `System.exit()`all thread are stopped.

## Callable interface
- It allows to define a tasks that can be return a value and raise exceptions
- The result is a object that implements the `Future` interface
- This interface contain only the `call()` method, it can be return a value

```java
import java.util.concurrent.Callable; 
public class Calculator implements Callable<Integer> { 
	private int a; 
	private int b; 
	
	public Calculator(int a, int b) { 
		this.a = a; 
		this.b = b; 
	} 
	
	public Integer call() throws Exception { 
		Thread.sleep((long)(Math.random() * 15000)); 
		return a + b; 
	} 
}

public class Adder { 
	public static void main(String[] args) throws ExecutionException,InterruptedException{ 
		// Create thread pool using Executor Framework 
		ExecutorService executor = Executors.newFixedThreadPool(5); 
		List<Future<Integer>> list = new ArrayList>(); 
		
		for (int i = 1; i < 11; i=i+2) { 
			// Create new Calculator object 
			Calculator c = new Calculator(i, i + 1); 
			list.add(executor.submit(c));} 
			int s=0; 
		}
		for (Future f : list) { 
			try { 
				System.out.println(f.get()); 
				s=s+f.get(); 
			} catch (Exception e) {
			
			};
		} 
		System.out.println("la somma e'"+s); executor.shutdown(); 
	}
}
```

#### Future interface
```java
public interface Future { 
	V get( ) throws...; 
	
	V get (long timeout, TimeUnit) throws...; 
	
	void cancel (boolean mayInterrupt); 
	
	boolean isCancelled( ); boolean isDone( ); 
}
```

- the `get()` method is blocked until the thread produce the right value 
- `get(long timeout, TimeUnit)` define a maximum waiting time
# References
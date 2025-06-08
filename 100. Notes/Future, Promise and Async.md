**Data time:** 16:57 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Concurrency in C++]]

**Area**: [[Master's degree]]
# Future, Promise and Async

C++ provides a mechanism for return-value-passing specifically designed to fit the characteristics of **asynchronous execution**. The so-called **promises** that are **fulfilled in the future**.

A promise and a future implement a "**single-assignment channel**" (a one-shot communication mechanism) between two entities, usually two threads. `s = (p, f)` where `p` is a writable view of the state `s` (the promise), f (the future), is a readable view of the state s that can be accessed after being signaled (**only once!**) by the promise.

The causal dependency between the **promise p** and the **future f** implements a **synchronization mechanism** between two threads.

The state s = (p, f) represents the logical channel. It is defined by a promise `std::promise<T> p` for a specific data type T, which has associated the `std::future<T> f =p.get_future()`
- `std::promise<T>` p represents the write-end of the communication
- `std::future<T> f` represents the read-end of the communication

`p` is passed as **rvalue reference** in the signature of the called function via `std::promise<T> && p`; p has to be moved using `std::move()` from the master thread to the spawned thread since std::promise is not copiable.

![[Pasted image 20250602170616.png]]

In the code snippet, we show how to pass a single promise to the thread to get the result
- When using a lambda to call the thread function, pay attention that the promise cannot be passed by reference, it must be moved
- The lambda must be mutable to make the captured entities non-const attributes of the closure object

```c++
template <typename value_t>
void fibo(value_t n,
          std::promise<value_t>&& result) { // <- pass promise
  value_t a_0 = 0, a_1 = 1;
  for (uint64_t index = 0; index < n; index++) {
    const value_t tmp = a_0;
    a_0 = a_1;
    a_1 += tmp;
  }
  result.set_value(a_0); // fulfill the promise
}

// this runs in the master thread
int main(int argc, char* argv[]) {
  const uint64_t num_threads = 32;

  std::vector<std::thread> threads;

  // allocate num_threads many result values
  // Each future will correspond to the result
  // computed by one of the worker threads.
  std::vector<std::future<uint64_t>> results;

  for (uint64_t id = 0; id < num_threads; id++) {
    // define a promise and store the associated future
    std::promise<uint64_t> promise;

	
    results.emplace_back(promise.get_future());

    threads.emplace_back(
        // specify template parameters and arguments
        // fibo<uint64_t>, id, std::move(promise)

        // using lambda and automatic type deduction
        // mutable avoids p (and also id) to be const
        [id, p{std::move(promise)}]() mutable { fibo(id, std::move(p)); }

    );
  }
  // read the futures resulting in synchronization of threads
  // up to the point where promises are fulfilled
  for (auto& result : results)
	  // If the worker thread associated with this 'future' has already called
	  // 'promise.set_value()', 'result.get()' immediately returns the value.
	  //      - If the value has not been set yet, 'result.get()' **blocks**
	  //        the execution of the main (master) thread and waits until
	  //        'promise.set_value()' is called by the worker thread.
	  //      - Once the value is obtained, it's printed.
    std::cout << result.get() << std::endl;
  // join the threads
  for (auto& thread : threads)
    thread.join();
}
```

### Task Objects
The “future” C++ header provides the `std::package_task` function that allows you to conveniently construct task objects. A task object is a **callable object** with the associated corresponding **future** object handling the return value. 

A callable object is any entity that can be invoked using the function **operator()**. It includes `std::function` wrapper, functors, lambda expression, std::bind expression. A callable includes callable objects, regular functions, and function pointers.

**Example**: Assume we want to create a task from this function.
![[Pasted image 20250602173503.png]]

We can create a specific task that calls the fibo function using `std::packaged_task`. **Note**: std::packaged_task is a move-only type:

```c++
// traditional signature of fibo
uint64_t fibo(uint64_t n) {
  uint64_t a_0 = 0, a_1 = 1;
  for (uint64_t index = 0; index < n; index++) {
    const uint64_t tmp = a_0;
    a_0 = a_1;
    a_1 += tmp;
  }
  return a_0;
}

// this runs in the master thread
int main(int argc, char* argv[]) {
  const uint64_t num_threads = 32;
  std::vector<std::thread> threads;

  // allocate num_threads many result values
  std::vector<std::future<uint64_t>> results;

  // create tasks, store futures and spawn threads
  for (uint64_t id = 0; id < num_threads; id++) {
    // create one task
    // La firma del template di packaged_task è: ReturnType(Arg1Type, Arg2Type, ...)
    std::packaged_task<uint64_t(uint64_t)> task(fibo);
    // store the future
    results.emplace_back(task.get_future());
    // spawn a thread to execute the task
    // std::move(task) è necessario perché packaged_task non è copiabile.
    // 'id' viene passato come argomento alla funzione 'fibo' quando il task viene eseguito.
    threads.emplace_back(std::move(task), id);
  }

  // Leggi i future, causando la sincronizzazione dei thread
  // fino al punto in cui i task sono completati e i future sono pronti.
  for (auto& result : results) std::cout << result.get() << std::endl;
  // Join dei thread
  for (auto& thread : threads) thread.join();

  return 0; // Buona pratica aggiungere un return in main
}
```

The problem with the described approach is that the function signature is hard-coded in the template parameter of the `std::package_task`. It would be better if all tasks exhibit the same signature, e.g., `void task(void)`, **independently of the callable object to be invoked**

This can be achieved by using a task factory function template (`make_task`). Purpose:
- Custom scheduling of tasks with a dedicated thread pool
- Allows to decouple the creation of the task from its execution
- More control over error handling and cancellation

```c++
template<
	typename Func,    // <-- tipo della funzione/chiamabile originale
	typename ... Args,// <-- tipi degli argomenti da "pre-caricare"
	typename Rtrn=typename std::result_of<Func(Args...)>::type
	> 				// ^-- type of the return value func(args)
auto make_task(
	  Func && func,      // Riferimento universale alla funzione
	  Args && ...args)   // Riferimenti universali agli argomenti
	  -> std::packaged_task<Rtrn(void)> { // Tipo di ritorno esplicito

	// Fondamentalmente, costruisce un oggetto chiamabile 'aux'
	// (una funzione ausiliaria aux(void)) senza argomenti
	// che quando chiamata esegue func(arg0, arg1, ...)
	auto aux = std::bind(std::forward<Func>(func),
						 std::forward<Args>(args)...);

	// Crea un task che avvolge la funzione ausiliaria:
	// task() esegue aux(void), che a sua volta esegue func(arg0, arg1, ...)
	// La firma del task è Rtrn(void) -> ritorna Rtrn, non prende argomenti
	auto task = std::packaged_task<Rtrn(void)>(aux);

	// Il valore di ritorno di aux(void) (e quindi di func(args...))
	// viene assegnato a un oggetto future accessibile tramite task.get_future()
	return task;
}
```

### Asynchronous
The “future” C++ header provides an out-of-the-box **`std::async`** function for creating task objects. **Higher-level interface for launching tasks asynchronously**. It executes a task object asynchronously using either a separate thread or the calling thread and returns a future.
```c++
auto future = std::async(fibo, id);
```
However, there are pitfalls to pay attention to:
- A call to std::async does not necessarily imply that a new thread is spawned
	- The calling thread might execute the task without spawning a new thread!
	- Default behavior depends on the implementation; thus, remember to use **`std::launch::async`**
- The execution of the task could be **deferred forever** if the future is not accessed (i.e., **`future.get()/.wait()`**)
```c++
	auto future = std::async(std::launch::async, fibo, id);
```
- The execution of distinct tasks could be serialized. If the destructor of the future is called just after the async.

The code snippet of the Fibonacci example is rewritten using the `std::async` construct. Pay attention to code like the following one:
```c++
for (uint64_t id = 0; id < num_threads; id++){
	auto future = std::async(std::launch::async, fibo, id);
} // <- here, the destructor of the future is called
```
- The destructor of the future can block until the async task completes (any exceptions are lost)
- The future must be stored outside the loop body

The use of std::async is not advisable for methods without a return value.

```c++
uint64_t fibo(uint64_t n) {
  uint64_t a_0 = 0, a_1 = 1;
  for (uint64_t index = 0; index < n; index++) {
    const uint64_t tmp = a_0;
    a_0 = a_1;
    a_1 += tmp;
  }
  // std::cout << "thread " << n << " has completed its task\n";
  return a_0;
}

int main(int argc, char* argv[]) {
  const uint64_t num_threads = 32;

  std::vector<std::future<uint64_t>> results;
  // for each thread
  for (uint64_t id = 0; id < num_threads; id++) {
    // directly emplace the future
    results.emplace_back(
        std::async(std::launch::async, fibo, id)
        // std::async(std::launch::deferred, fibo, id)
    );
  }
  // std::this_thread::sleep_for(std::chrono::seconds(2)); // If using 2s

  for (auto& result : results) {
    std::cout << result.get() << std::endl;
  }
  // There's no need for an explicit loop to join threads
  // when using std::async with the std::launch::async policy.
  // The future's destructor (if it's the last reference to the shared state)
  // will block until the asynchronous task is complete.
  // However, the .get() call above already ensures synchronization.

  return 0;
}
```


# References
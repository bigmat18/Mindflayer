**Data time:** 16:27 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Concurrency in C++]]

**Area**: [[Master's degree]]
# C++ Threads

The master thread can spawn threads, and each thread can spawn threads as well. The number of spawned threads should be roughly the amount of cores (i.e., pay attention to **oversubscription**). Threads share process resources (each thread has a separate stack).  A  thread can be **joined** or **detached** once:
- A detached thread cannot be joined
- Joined or detached threads cannot be reused
- All threads must be joined or detached within the scope of their declaration

![[Pasted image 20250602163006.png | 300]]

### Spawning and Joining C++ Threads
We need to store the thread handles explicitly to be able to access them during the join phase. In the code snippet, we use a `std::vector` container and the method `emplace_back`. Alternatively, we could have **moved** the thread object implicitly using the vector member function `push_back`.
```c++
treads.push_back(std::thread(say_hello, id))
```

The type **std::thread** is **move-only** (i.e., not copyable), copy constructor and copy assignment operator deleted (i.e., =delete).

```c++
void say_hello(uint64_t id) {
 std::cout << "Hello from " << id;
}

// this runs in the master thread
int main(int argc, char * argv[]) {
	const uint64_t num_threads = 32;
	std::vector<std::thread> threads;
	
	// allocate num_threads many result values
	std::vector<uint64_t> results(num_threads, 0);
	
	for (uint64_t id = 0; id < num_threads; id++) {
		auto& th = threads.emplace_back(
			say_hello, id
		);
		// separate th from parent thread, it became non-joinable, th
		// can be reasagned to other thread, the responsabilitiy to cleanup
		// is transfer to OS
		// th.detach()
	}
	
	// join the threads
	for (auto& thread: threads)
		thread.join();
	
	// print the result
	for (const auto& result: results)
		std::cout << result << std::endl;

}
```

Let’s consider a simple scalar function that iteratively computes the n-th Fibonacci’s number
$$
a_n = a_{n-1} + a_{n-2} \:\:\: with \:\: a_0= 0, a_1=1
$$

In the code snippet we show two different ways to pass arguments to the thread functions; the second argument is passed by reference to store the result. Potential pitfalls:
- The memory passed via a reference (or pointer) has to be persistent during the execution of threads
- The memory used for variables or objects defined within the loop body where threads are spawned is destroyed when leaving the scope (the destructor is called)

```c++
template <typename value_t>
void fibo(value_t n,
          value_t &result) { // <- here we pass the reference
  value_t a_0 = 0, a_1 = 1;
  for (uint64_t index = 0; index < n; index++) {
    const value_t tmp = a_0;
    a_0 = a_1;
    a_1 += tmp;
  }
  result = a_0;
}

// this runs in the master thread
int main(int argc, char *argv[]) {
  const uint64_t num_threads = 32;
  std::vector<std::thread> threads;

  // allocate num_threads many result values
  std::vector<uint64_t> results(num_threads, 0);

  for (uint64_t id = 0; id < num_threads; id++) {
    threads.emplace_back(
        // Version 1: specify template parameters and arguments
        //       we have to explicitly state the type in
        //       the fibo function tesmplate because it is not
        //       automatically deduced
        // fibo<uint64_t>, id, std::ref(results[id])
        

        // Version 2: using lambda and automatic type deduction
        //       pay attention to id, must be passed by value!
        //[id, &results]() { fibo(id, results[id]); }
        [id, &r = results[id]]() { fibo(id, r); }

    );
  }

  // join the threads
  for (auto &thread : threads)
    thread.join();
  // print the result
  for (const auto &result : results)
    std::cout << result << std::endl;

  return 0; // Aggiunto return 0 per coerenza
}
```

**Note**: in **Version 1**, when we pass `result[id]` we use `std::ref`, it return a `std::reference_wrapper<T>` che è un oggetto speciale che "avvolge" un riferimento, ma è esso stesso copiabile. 
- l costruttore di `std::thread` ha bisogno di conservare questi argomenti per poterli usare quando il nuovo thread inizierà effettivamente l'esecuzione della funzione `fibo`. Per fare ciò in modo sicuro li **copia** (o li sposta, se applicabile) in una sua **area di memoria interna**.
- Una copia dell'oggetto `std::reference_wrapper<uint64_t>` viene memorizzata. Chiamiamo questa copia `internal_ref_wrap`. Questa i`nternal_ref_wrap` punta ancora all'elemento originale `results[id]` nel vettore results del thread principale.
- La classe `std::reference_wrapper<T>` è progettata con un operatore di conversione speciale. Un operatore di conversione permette a un oggetto di una classe di essere usato come se fosse di un altro tipo in certi contesti. La chiamata assomiglia a questo (concettualmente):
```
fibo(copied_id, internal_ref_wrap);
```
- Il compilatore vede il secondo parametro di `fibo(value_t& result)` si aspetta un `uint64_t&`. Ma gli stiamo passando `internal_ref_wrap`
- Qui si attiva l'operatore di conversione, il compilatore cerca un modo per convertire il std::reference_wrapper in un `uint64_t&`. Trova l'operatore operator `uint64_t& () cons`t definito dentro `std::reference_wrapper`
- Quindi, la funzione `fibo` riceve effettivamente un `uint64_t&` che punta all'elemento corretto del vettore results

What happens if we use the following?
```c++
[&]() { fibo(id, results[id]); }
```
- `id` **verrebbe catturato per riferimento**. Poiché il loop che crea i thread modifica `id` ad ogni iterazione, tutti i thread (o la maggior parte di essi, a seconda dello scheduling) finirebbero per vedere il valore finale di `id` (cioè `num_threads`) o comunque valori imprevisti, invece del valore che id aveva al momento della creazione del thread. 


# References
[std:reference_wrapper C++ guide](https://en.cppreference.com/w/cpp/utility/functional/reference_wrapper.html)
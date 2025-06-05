**Data time:** 15:57 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Concurrency in C++]]

**Area**: [[Master's degree]]
# C++ Essentials for Concurrency

### Automatic type deduction
The compiler inspects the right-hand side (RHS) of the assignment and deduces the type. Useful for:
- Complex types, e.g., iterator types from STL containers or types from lambda expressions
- To make code cleaner and more maintainable
- To work with templates and type traits

```c++
auto x = 10; // deduced as int
auto d = 3.14; // deduced as double
auto str = “Hello”; // deduced as const char*
const int c = 1024; // just a constant number
auto x = c; // deduced as int (not const int)
const auto y = c; // deduced as const int
auto& r = c; // deduced as const int&
auto list{1,2,3}; // deduced as std::initializer_list<int>
```

In Functions and lambdas:
```c
auto add(int a, int b) {
	return a+b; // deduced as int
}
// generic lambda, it works for any type as long as the
// types passed support the multiplication operator (*)
auto F = [](auto a, auto b) {
	return a * b;
};
```

Can be also **template argument deduction** were the compiler determines the types from the arguments used in a function or class template instantiation. 
```c
template <typename T, typename U>
auto multiply(T x, U y) {
	return x * y;
}
multiply(2, 3); // int return type
multiply(2, 3.14); // double return type
multiply(2, 3.14f); // float return type
```
##### Forwarding reference (auto&&)
Special kind of reference that can bind to both **lvalues** and **rvalues** enabling to write code that accepts both `T&` and `T&&` (i.e., lvalue and rvalue references). A **rvalue** is a temporary object that does not persist beyond the expression where it is used. However, named rvalues do exist, such as `int&&` `r=getSomeValue()`

`std::forward` enables perfect forwarding by preserving the value category (lvalue or rvalue) of the reference. Note that `std::string&& s` is not a forwarding reference but denotes a rvalue reference (Here the type is explicit and not deduced)

```c++
std::vector<std::string> V;
void insert(const std::string& s) {
	V.push_back(s); // calls the copy constructor
}

void insert(std::string&& s) {
	V.emplace_back(std::move(s)); // calls the move constructor
}

template <typename T>
void insertValue(T&& value) {
	insert(std::forward<T>(value)); // perfect forwarding
}
```

### Move Semantics
Move semantics improves performance and resource management, **enabling resources to be transferred (moved) rather than copied**. Moving objects only makes sense if the object type owns a resource such as std::string and std::vector (containers in general).

`std::move` is a utility function that casts its argument to a `rvalue reference`. It indicates to the compiler that the object can be moved from the current object ( `std:move`  does not move by itself)
- A **rvalue reference** is declared with && and it is a temporary or short-lived object not tied to a named variable
- Move semantics relies on rvalue references to implement and use move constructors and move assignment operators

![[Pasted image 20250602161457.png]]
After moving, an object is in a valid but unspecified state (you should not rely on old content)

Move semantics requires move constructors and move assignment operators, such functions may be either user-defined or compiler-generated.

**Rule of Five**: if your class explicitly allocates and deallocates resources, or if you define any of the following, all of them must be defined:
1. Copy constructor
2. Copy assignment operator
3. Move constructor
4. Move assignment operator
5. Destructor

Standard containers (e.g., `std:vector`) support move semantics. `noexcept` tells the compiler that a function does not throw exception. C++ containers prefer move constructor to be marked `noexcept` to enable move semantics. If not used it may fall back to copying.

```c
class MyClass {
public:
	// constructor
	MyClass(std::string str): data(std::move(str)) {}
	// copy constructor
	MyClass(const MyClass&)= default;
	// move constructor
	MyClass(MyClass&& other) noexcept
	: data(std::move(other.data)) { }
	// copy assignment operator
	MyClass& operator=(const MyClass&)= default;
	// move assignment operator
	MyClass& operator=(MyClass&& other) noexcept {
		if (this != &other) data = std::move(other.data);
		return *this;
	}
	std::string data;
};
```

### Function Objects and Lambdas
Functors are more flexible than functions as they carry arbitrary state information. The compiler often inlines functor calls, since the functor’s **operator()** definition is entirely visible. Functors used in generic programming and STL algorithms.

```c++
template <typename T>
struct MaxTracker {
	MaxTracker(const T& v) : curr_max(v) {}
	inline bool operator()(const T& v) {
		if (v> curr_max) { curr_max=v; return true; }
		return false;
	}
	T curr_max;
};

std::vector<float> V = {1.0, 2.0, 3.0, 4.0};
MaxTracker tracker(V[0]);
std::for_each(V.begin(), V.end(), tracker);
```

**Lambdas** are an **inline**, **anonymous** way to define function objects. Syntax:
```
[capture](parameters) mutable → return_type { body }
```
- `[capture]` specifies which variables from the surrounding scope to capture, and how (by value or by reference)
	- `[&]` captures all variables **by reference**
	- `[=]` captures all variables **by value**
- `mutable` (optional) allows modification of variables captured by value
- `return_type` is optional because it is often deduced automatically

```c++
auto print= [](const auto& e) { std::cout << e << “\n”; };
std::vector<float> V={1.0, 2.0, 3.0, 4.0};
for(const auto& e: V) {
	print(e);
}
```

**Lambda functions have unique types**. It is difficult to store lambdas in containers and pass them around. `std::function` can be used to hold functors constructed by lambdas
```
std::function< return-type (param1, param2, …) >
```
Lambda functions with the same number of parameters and the same return type can be held by `std::function`. It can store: function pointers, lambdas, functors, std::bind expressiong, member function pointer, std::package_task objects.

```c++
void print(const std::string& s) { std::cout << s; }
std::vector<std::function<void (std::string&)>> F={
	print, [](std:.string& s){ std::cout << s; } 
};
```

Useful for event handlers, threading API and asynchronous execution. Small performance loss of std::function than lambdas. More overhead but more flexibility
- Usually std::function uses heap-allocated memory to store the captured variables
- More boilerplate due to type erasure. If type is unnecessary, prefer template function or auto instead of `std::function`
# References
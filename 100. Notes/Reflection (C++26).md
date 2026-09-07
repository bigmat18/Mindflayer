---
Data: 2026-08-10T19:46:00
Tags:
  - note
  - padawan
  - "#article"
Connection:
  - "[[C++ Features]]"
Area:
---
# Reflection (C++26)

C++ has a lot of compile-time information that are used every time the compiler checks types or resolves overloads. The problem was that **none of that information was available to your program**

The limitation wasn't in the compiler but it was that there wasn't a standard way for your code to access it.

Some famousness examples:
- Convert enum to a string
- Serialise struct to JSON
- Generate a GUI based on class members

This issue will be fixed in C++ 26 with the reflaction allowing you to access at this information and use it to generate more code via `constexpr` and `consteval`

### Basic Core Features
The metal model looks like this:
![[Pasted image 20260810201145.png]]
From the source code we can extract the meta information to generate new code in compile time. To to that we introduce a new operatore.
##### Reflection operator `^^`
If you apply this operatore to something in your program (a type, a variable a member a class ecc..) it will get back a value of type `std::meta::info`. This is the core info value.

Smallest possible example:
```c++
#include <meta>  
  
int global;  
  
// The compiler must evaluate these at compile time  
consteval std::meta::info type_int   = ^^int;  
consteval std::meta::info var_global = ^^global;
```

- `^^int` and `^^global` means "give me the reflection info for the type that specific type.
- The result type is always `**std::meta::info**`.

You can also use `constexpr`:
```c++
#include <meta>  
  
struct S {};  
  
constexpr auto s_type = ^^S; // OK: constexpr object of type std::meta::info
```

> [!NOTE]
> you never construct `std::meta::info` by hand. You always get it from `^^` (or from library functions that return it).
> 

The `std::meta::info` is an **opaque** value type that identifies some program entity.  **Opaque** means that you cannot see its internals, you don't know whether it holds a pointer, an index or something else.

Furthermore consider that `std::meta::info` should be view as a stable ID that the compiler gives you for a specific type (or class, enum ecc..)

This assumtion can be demostrated with the following example:
```c++
#include <meta>  
#include <iostream>  
  
struct S {};  
  
int main() {  
    S a{};  
    S b{};  
  
    constexpr std::meta::info ra = ^^a;  
    constexpr std::meta::info rb = ^^b;  
    constexpr std::meta::info ra2 = ra; // Copying the handle to 'a'  
  
    // 'a' and 'b' have the same type, but are different entities  
    constexpr bool same_vars = (ra == rb);  // false  
    constexpr bool diff_vars = (ra != rb);  // true  
  
    // 'ra' and 'ra2' both point to the exact same entity 'a'  
    constexpr bool same_handles = (ra == ra2); // true  
  
    std::cout << std::boolalpha   
              << "same_vars: " << same_vars << '\n'  
              << "same_handles: " << same_handles << '\n';  
}
```

#### Splicing operator `[: ... :]`
To use the `^^` operator and its result is very useful another new operator introduced by c++26 that is the **splicing**. Below an example:

```c++
#include <meta>  
  
int main() {  
    constexpr auto r = ^^int;  
    typename[:r:] x = 42;        // becomes: int x = 42;  
    typename[:^^char:] c = '*';  // becomes: char c = '*';  
}
```

In this example:
1. `^^int` produces a `**std::meta::info**` that describes the type `int`
2.  `[:r:]` takes that info and injects the described type at that point in the code.

> [!NOTE]
> `typename` is required here because, syntactically, `[:r:]` is a dependent construct that yields a type

### Asking the Compiler Questions

### Compile-Time Type Synthesis


# References
- [Paper about Reflection](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r9.html)
- [Medium Guide](https://towardsdev.com/cpp26-static-reflection-guide-part-1-0a4f21ff781d)
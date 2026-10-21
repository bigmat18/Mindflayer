---
Data: 2026-10-07T00:39:00
Tags:
  - note
  - youngling
Connection:
Area: "[[Resources Collections]]"
---
# C++ `declval` vs `decltype`

### `decltype`
**decltype** is used when you want the compier to deduce a type based on an expression or a variable. For an expression it deduces the type without evaluating it by analyzing the expression's value category and type. We can have both:
- `decltype(entity)`
- `decltype(expression)`

This entity/expression can be **parenthesized** or **unparenthesized**:
- When you apply decltype to an **unparenthesized** entity or expression `decltype(parameter)`, decltype returns the exact type of the variable as it was declared.
```cpp
int x = 5;
int& y = x;

decltype(x) a; // `a` is of type `int`
decltype(y) b = a; // `b` is of type `int&
```

- But when decltype is applied to a **parenthesized** entity or expression `decltype((parameter))`, it follows a different set of rules to deduce the type.
	- If the expression is an **lvalue**, decltype returns `T&`, where `T` is the underlying type of the expression.

	- If the expression is a **prvalue** (pure rvalue, a subset of rvalues, Temporary values with no specific memory address. I like to call them the real rvalues), decltype returns the underlying type `T` without any reference qualifiers.
	
	- If the expression is an **xvalue** (eXpiring value, object that is about to be moved or is near the end of its lifetime), decltype returns `T&&`, where `T` is the underlying type of the expression.
	
```cpp
int a = 10;                  // `a` is an lvalue
int b = a + 20;              // `a + 20` is a prvalue (temporary result)
int&& c = std::move(a);      // `std::move(a)` is an xvalue

int x = 5;

decltype(x + 0) d; // `d` is of type `int` (prvalue)
decltype((x)) e = d; // `e` is of type `int&` (lvalue)
decltype(std::move(x)) f = 10; // `f` is of type `int&&` (xvalue)
```

Usually `decltype` is used paired with `auto` for **type deduction**, they serve differents purpose and have different behaviors

### `declval`
Different from the above feature`std::declval` is a utility that creates a "fake" instance of a type by returning an rvalue reference to that type. However this fake instance does not actually exist in memory. It's just a conceptual tool that the compiler uses to deduce types.

This means that `std::declval` can only be used in **unevaluated contexts**. If you try to evaluate an expression involving `std::declval`, the code would attempt to access a non-exists object, leading to undefined behavior or a compiler error.

Examples of unevaluated contexts include:
- `decltype:` Used to deduce the type of an expression without evaluating the expression.
- `sizeof:` Determines the size of a type or expression without evaluating the expression.
- `noexcept:` Checks whether an expression can throw an exception without evaluating the expression.

```cpp
#include <utility>

int main() {
    auto obj = std::declval<int>();  // Error: `std::declval` cannot be used here
}
```

Using `std::declval` with `decltype`:

```cpp
#include <utility>
struct MyClass {
    int foo() const { return 42; }
};

// Here, `std::declval<MyClass>()` is used in an unevaluated context
// to deduce the return type of `foo()` without creating an actual `MyClass` object.

decltype(std::declval<MyClass>().foo()) x;  // Deduces `x` as `int`
```
# References
- https://jenniferchukwu.com/posts/decltype
- https://en.cppreference.com/cpp/language/decltype
- https://en.cppreference.com/cpp/utility/declval
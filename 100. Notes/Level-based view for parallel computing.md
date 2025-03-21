---
Created: "{{time}} - {{date}}"
tags:
  - "#note"
Links: "[[High performance computing]]"
Area: "[[Master's degree]]"
---
# Level-based view for parallel computing

## Vertical structuring
The functional of a system can be organised in a hierarchy of **interpretations layers** (or **virtual machines**).
Each level provides a set of functionalities for upper levels ad hides the implementation of the lower levels.

![[Screenshot 2024-09-22 at 15.15.33.png | 150]]
###### Virtual machines
The hierarchy is structured with a language-based approach, where each level $MV_i$ are interpreted by programs written at level $MV_j$ with $j < i$.

###### Runtime system (RTS)
If COM is a command of language $L_i$ of $MV_i$ the interpreter, or **run-time support**, of COM will be denoted by RTS(COM)

![[Screenshot 2024-09-22 at 15.17.11.png | 250]]

### Interpreters
An interpreter is a program that translate the input program **one statement at a time** and replace each instance of a command with the same implementation. The **optimizations** are possibile but they are more difficult because are runtime.

![[Screenshot 2024-09-22 at 15.22.16.png | 300]]

### Compilers
The compiler analyzes the input program **statically** and creates a traslated version in the target language $L_j$ with $j < i$ in a single step. More **optimizations** are possibile. In compilers exist multiple versions of RTS(COM), it chooses the best depending on how COM is used in the **surrounding context** (it's called **static analysis**)

![[Screenshot 2024-09-22 at 15.24.59.png | 300]]
# References
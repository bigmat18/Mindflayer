---
Data: 2025-11-14T14:40:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Introduction to Machine Learning]]"
Area: "[[Master's degree]]"
---
# Learning Algorithms

A learning algorithm is based on [[Data in ML|Data]], [[Tasks in ML|Task]] and [[Models in ML|Model]]. We use a **heuristic** that means search through the hypothesis space **H** of the **best hypothesis**
- ie the best approximation to the (unknown) target function
- typically searching for the $h$ with the minimum error
- e.g. free parameters of the model are fitted to the task at hand
- examples: best $w$ in linear models, best rules for symbolic models, ...

**H** may not coincide with the set of all possible functions and the search can not be exhaustive: we need to make assumption that we call **inductive bias**

![[Screenshot 2025-11-14 at 12.12.24.png | 500]]

# References
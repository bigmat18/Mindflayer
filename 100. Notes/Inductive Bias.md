---
Data: 2025-11-14T14:39:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Introduction to Machine Learning]]"
Area: "[[Master's degree]]"
---
# Inductive Bias

In order to set up a model and a learning algorithm we can make assumptions (about the nature of the target function) concerning either
- **Language Bias (inductive bias):** Constraints in the model (in the hypothesis space H, due to the set of hypothesis that we can express of consider). It's about the _scope_ of what can be learned.
- **Searching Bias:** Constraints or preferences in learning algorithm/search strategy. It's about the _strategy_ for finding the best model within the scope.
- We can also use **both**

We will see that such assumptions are strictly need to obtain an useful model for the ML aims, ie a model with generalisation capabilities.

We start to discuss it within examples in discrete hypotheses space (rules), **learning a concept** (a boolean function)
- e.g. $x$ is a 'cat' if $h_{cat}(x) = 1$, otherwise is $0$ for $x$ in 'animals'
###### Example 1
![[Screenshot 2025-11-14 at 12.21.21.png | 400]]
This is an **ill posed** (inverse) problem: we may violate either existence, **uniqueness**, stability of the solution of solutions.

There are $2^{16} = 65536$ possible **boolean functions** over four input features. We can non figure out which one is correct until we have seen every possible input-output pair. After 7 examples we still have $2^9$ possibilities.

In the general case, in this discrete hypothesis space H: $|H| = 2^{\text{\#input-instances}} = 2^{2^n}$ 
for binary inputs/outputs, n = input dimension

I.e. a rote learner: Store/memorize examples, classify $x$ if and only if matches a previously observed example (else "no answer")
- No inductive bias we hace no generalization

###### Example 2
As second example of discrete H, we can image to learn a discrete function with discrete inputs assuming **conjunctive rules** (propositions with AND among literals, a language bias).

I.e. using a language bias work a restricted hypothesis space. 
Example: $h_1 = l_2, \:\:\: h_2 = (l_1 \text{ and } l_2), \:\:\: h_3 = true, \:\:\: h_4 = not(l_1) \text{ and }l_2$  ...
- rules such as if $l_2(=true)$ then $h(x) = true$ else $h(x)=false$ of equivalently if $(x_2 = 1)$ then $h(x)=1$ else $h(x) = 0$

With $n$ binary inputs we had $|H| = 2^{\text{\#input-instances}} = 2^{2^n}$
With only conjunctive rules: \#semantically distinct hypotheses (conjunctions):
$3^n$ (for each of the n positions we can have $l_i, not(l_i)$, don’t care) + 1 (+1 because all h with $(l_i \text{ AND } not(l_i)$) are equivalent to ”false”)

###### Find the version space
Given the def: a hypothesis $h$ is **consistent** with TR, if $h(x)=d(x)$ for each training example $<x, d(x)>$ in TR.

- It is possible to perform a complete search (finding the set of all h consistent with the TR set) in an efficient way in this reduced space (of conjunctive rules)by cleverer algorithms.
	- Instead of searching enumerating all the possible combination of literals ie, every $h$ in H
- We are only interested to say that these algorithms find the VS
- Call the $version space$, $VS_{H, TR}$ with respect to hypothesis space H, ad training set TR, the subset of hypotheses from H consistent with all training examples

#### Unbiased Learner
Hence, this conjuctive assumption for H lead to an efficient solution in finding a VS. However, using only conjuctive rules may be **too restrictive**. If the target concept is not in H, it cannot be reppresend in H.
- e.g. if $(x_1 =1)$ or $(x_2 = 1)$ then $h(x) = 1$ else $h(x) = 0$

**Idea**: Choose H that expresses every teachable concept (among propositions), that means $H$ is the set of all possible subsets of $X$ (instance or input space): the power set $P(X)$

E.g. $n=10$ binary inputs $|X|=x^{10}=1024$, $|P(X)| = 2^{1024} \sim 10^{308}$ distinct concepts (much more than the num of atoms in universe)
- H = disjunctions, conjunctions, negations
- H surely contains the target concept

Recall that the **version space** $VS_{H,TR}$ with respect to hypothesis space H, and training set TR, is the subset of hypotheses from H consistent with all trining examples.

**Property**: An unbiased learner is unable to generalize (on new instances)
**Proof**: Each unobserved instance will be classified 1 (positive) by precisely half the hypothesis in VS and 0 (or negative) by the other half (rejection: no answer is made by the VS for new input instances). 
Indeed: $\forall h$ consistent with $x_i(test)$, $\exists h'$ identical to $h$ except $h'(x_i) <> h(x_i)$, $h \in VS \to h' \in VS$  (because they are identical on TR)

###### Futility of Bias-Free Learning
A learner that makes no prior assumptions regarding the identity of the target function/concept has no rational for classifying any unseen instances. (Rescriction, preference) bias not only assumed for efficiency, it is needed for the generalization capability.
- However, it does no tell us (quantify) which one is the best solution for generalization yet

**Trivial example** (TR=training set, TS=test set):

![[Screenshot 2025-11-14 at 14.36.06.png | 500]]
In other words, in order to learn the target concept, one would have to present every single instance in X as a training example (lookup table)
###### Inductive systems and equivalent deductive systems
![[Screenshot 2025-11-14 at 14.37.46.png | 600]]

Why the **search bias** can be preferred over the **language bias**?
- In ML typically use **flexible** approaches (expressive hypothesis spaces, universal capability of the models, e.g. Neural Networks, DT)
- avoiding the language bias, hence without excluding a priori the unknown target function,
- retaining an inductive bias but focusing on the search bias (which is ruled by the learning algorithm). In practice using an incomplete search strategy

**Conclusions**:
- Learning without bias cannot extract any regularities from data (lookup-table: no generalization capabilities)
- Every state-of-the-art ML approach shows an inductive bias
- Issue: characterize the bias for different models/learning approaches
# References
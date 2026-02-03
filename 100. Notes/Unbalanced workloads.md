---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Unbalanced workloads

Some data structurs are irregular. A Triangulated Irregular Network (TIN) consists of a collection of non-overlapping triangles and is inherently irregular. Straightforward static partitioning may lead to severe workload unbalance between Workers.
- Graph/Geometry-based partitioning can help distribute the workload more evenly (domain specific algorithms)

In many such cases, dynamic task distribution policies are needed to adapt to changing workloads. In the static data distribution, the data assignment is predetermined at the program start.
- No extra overhead at runtime for assigning tasks to threads

Static approaches work well if the computation per data element does not vary too much and we know in advance the number of tasks to compute
- The computation of the Mandelbrot set is an example of unbulanced workload

One option is to use very small chunk sizes of tasks and block-cyclic assignment strategies:
- However, this strategy does not work well in all cases. When tasks are heavily skewed, only a few more tasks per thread may produce  load imbalance. In the worst-case scenario, all heavy tasks can be assigned to the same thread
- Additionally, static assignments cannot be used when the number of tasks is not known statically
- For unbalanced workloads, it is better to use **dynamic assignment strategies**

###### Example: All-pairs distance Matrix
We have a matrix  $D_{ij}$ of shape $m\times n$ where $i$ denotes one of the $m$ vectors (for examples, a vector is all pixels of a given image) and $j$ enumerates all $n$ elements of the vectors. We want to compute the distance (or more generally the similarity) $d()$ between all pairs in $D$
$$
\Delta_{ij} = d(x^{(i)}, x^{(i')}) \forall i, i' \in \{0, \dots m-1\}
$$
- The distance/similarity measure d(∙,∙) might be a traditional metric such as Euclidean distance or any symmetric binary function that assign a notion of similarity to pair of instances
- We have to calculate $m²$ distance/similarity scores between vectors. Assuming that  the computation of a single core value takes $O(n)$,we have $O(m²n)$ operations to compute the ditance matrix $\Delta$
- Howeever the matrix $\Delta$ is symmetric $\Delta_{ij} = d(x^{(u)}, x^{(i')}) = d(x^{(i')}, x^{(i)} )$ for all $i, i' \in \{0, \dots, m-1\}$  thus we have to calculate the lower triangular part of the matrix ∆ and then copy the elements in the corresponding position in the upper triangular part.

Let us consider as D the MNIST dataset. It consists of 70,000 handwritten digits stored as gray-scale images of shape 28 × 28. 60,000 images coming from the training dataset and 10,000 coming from the test dataset

For our purpose, we interpret each of the m=70,000 images as plain vectors each with n=784 points. All m images are stored in the matrix $D_{ij}$. Additionally, for the sake of simplicity, d(∙,∙) is the Euclidean distance.

![[Pasted image 20260202200548.png | 250]]

Due to the symmetry of $\Delta_{ij}$ we only have to compute i+1 entries in row i. In $\Delta_{ij}$ we have the distance between the i-th element and the j-th element, and $\Delta_{ij} = \Delta_{ji}$

A naive parallelization of the problem uses a static block-cyclic distribution of row elements of the 
matrix ∆ to threads for some value of the chunk c. For example, considering a 12 × 12 all-pairs distance matrix and c=2, we have:

![[Pasted image 20260202200709.png | 500]]

- The workload of the threads is **unbalance**
- It can be better with c=1 but still it remains unbalanced (thread0 22 tasks, thread1 26 tasks, thread2 30 task)
- We need a different strategy that assigns elements to compute to threads dynamically 

Results obtained running the all_pair.cpp implementation for the MNIST dataset on the cluster front-end node
- Sequential time: 704s
- Block-cyclic distribution varying c and using 40 threads.
- Dynamic distribution varying c and using 40 threads.

![[Pasted image 20260202200834.png | 550]]

![[Pasted image 20260202200911.png]]


# References
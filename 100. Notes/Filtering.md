---
Data: 2025-12-16T17:57:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Filtering
Typical node filtering/sampling is based on:
- **the node degree**: low degree nodes are less important than high degree nodes
- **the node role in the graph**: nodes that keep the graph connected are more important than redundant nodes
- **random choice (simple random sampling)**: every data point has the same probability of being selected

### Coreness
Rather than deleting nodes of low degree, nodes are recursively deleted, each deleted node decreses the degree of its adjacent nodes.

The **k-core** of a graph is obtained by recursively delete all nodes of degree less than k
- the k-core of a graph is the maximal induced subgraph such that each node has degree at least k

###### Example
![[Pasted image 20251216180253.png | 400]]

The **1-core** of G is obtained by removing isolated nodes
![[Pasted image 20251216180326.png | 400]]

The **2-core** of G is obtained by recursively removing nodes of degree 1.
![[Pasted image 20251216180354.png | 400]]

The **3-core** of G is obtained by recursively removing degree 1 and 2 nodes
![[Pasted image 20251216180421.png | 400]]

The **4-core** of G is obtained by recursively removing degree 1, 2, and 3 nodes
![[Pasted image 20251216180450.png | 400]]
##### Properties of k-cores
If, for some $k \geq 1$, a graph G has the k-core $G_k$ then:
- the k-core $G_k$ is unique
- graph G has also a (k-1)-core $G_{k-1}$
- the k-core is contained into the (k-1)-core $G_k \subset G_{k-1}$

### Betweenness
Let $\delta_{st}(v)$  denote the fraction of the shortest paths between nodes $s$ and $t$ that pass thorough $v$
$$
\delta_{st}(v) = \sigma_{st}(v) / \sigma_{st}
$$
where:
- $\sigma_{st}(v)$ is the number of shortest paths from s to t passing through v
- $\sigma_{st}(v)$ is the total number of shortest paths from s to t

The **betweenness** of v is defined as the sum of all such values for all other nodes s and t

###### Degree vs betweenness

![[Pasted image 20251216181648.png | 500]]


# References
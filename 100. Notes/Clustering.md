---
Data: 2025-12-16T18:17:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Clustering

One way of obtaining a high-level view of a graph is that of clustering its nodes, each cluster represents to the user the nodes it contains. Clusters may be recursively defined this means a cluster may contain other clusters.

![[Pasted image 20251216181813.png | 450]]

![[Pasted image 20251216181838.png | 450]]

![[Pasted image 20251216181903.png | 450]]

### Flat clustered graph

![[Pasted image 20251216181947.png | 450]]

### Visualizations of clustered graphs
#### 2D visualizations
- the underlying network uses the standard node-link representation
- clusters are represented by regions bounded by simple curves
	- each region includes all and only the nodes belonging to the corresponding cluster
	- region inclusion represents inclusion of clusters

###### Proper inclusion tree
![[Pasted image 20251216182226.png | 450]]

###### View of a clustered graph
![[Pasted image 20251216182300.png | 450]]

###### Multilevel drawings
- Each level is a planar drawing of a view
- The inclusion tree is represented by inclusion edges

![[Pasted image 20251216182333.png | 300]]

##### 2D visualizations of c-graphs

###### Clustered graph planarity
###### Clustered graph planarization
###### Clustered matrix
Rows and columns can be recursively contracted and expanded
![[Pasted image 20251216182936.png | 450]]
###### Hybrid visualizations
- NodeTrix combines node-link and matrix-based representations
- Clusters are manually defined
	- no automatic clustering
	- no automatic ordering for rows-columns

![[Pasted image 20251216183022.png | 500]]

#### 3D visualizations
- Clusters are surfaces
	- the position of the nodes is known in advance
	- details come in place when the user focuses on the clusters

![[Pasted image 20251216182113.png | 500]]

### Clustering Quality
Where the clusters come from?
- Manually defined clusters: the users themselves cluster the objects
- Extrinsic classification of nodes: takes advantage of meta-information
- Intrinsic classification of nodes: only based on the structure of the graph
	- property-bases clustering
	- cut-based clustering

**Graph clustering indexes**

**Coverage**: how the computed clusters covers edges of the whole graph

**Performance**: counts the number of “correctly interpreted pairs of nodes” by the clustering
- intra-cluster edges + non-connected pairs in different clusters over all pairs of nodes

### K-way partitioning
Partition the network into k clusters in such a way to minimize the number of inter-cluster edges
- need to know the number of clusters in advance
- NP-hard problem
- sometimes you require the clusters to be balanced

Alternatively: recursively partition the graph into two clusters until some ending condition is reached

### K-core components
The k-core components are the connected components of the k-core, for example these are the 3-core components:

![[Pasted image 20251216183351.png | 400]]

### The (X,Y)-clustering model
The goal is to guarantee:
- desired properties for the clusters
- desired properties for the graph of clusters

Let X and Y be two classes of graphs, G is an (X,Y)-graph if it admits a clustering such that
- the graph of clusters belongs to X
- each cluster induces a subgraph that belongs to Y

###### Example
Let X be the class of cycles and let Y be the class of $K_4$

![[Pasted image 20251216183600.png | 400]]
###### Interesting classes for X and Y
- Y is some class of highly connected graphs: cliques, subgraphs with high-degree vertices, ...
- X is some class of sparse graphs that guarantees readability: planar graphs, cycles, trees, paths, ...

**Note**: If you require that every node belongs to some cluster you have a stricter model
###### The (X,Y)-clustering problem
Given a graph G and two desired classes X and Y, is G an (X,Y)-graph? This problem is NP-hard in general, deciding whether G is a (planar, k-clique)-graph for desired $k\geq 5$ is NP-hard

This result motivates us to look for some relaxation of cliques
# References
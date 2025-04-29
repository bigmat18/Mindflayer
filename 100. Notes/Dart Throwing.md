**Data time:** 17:00 - 27-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Sampling]]

**Area**: [[Master's degree]]
# Dart Throwing

```c
n_miss = 0
do {
	x_cand = RandomPoint()
	if not x_cand ⊆ covered(X) // no disk in X contains x_cand
		X = X + x_cand // hit
	else 
		n_miss = n_miss + 1 // miss	
} while (n_miss / (n_miss + #X) < threshold)
```

1. We take a random point
2. check the point not containt other point in a ray r
3. if is not contains we add x_cand
4. else we add a miss
5. if we have too many miss we quit

**Pros**:
- Works on any domain provided a metric
**Cons**:
- Very slow convercente rate, $O(n²)$ asymptotic complexity
- Maximality not guaranteed in given time (likability of hit tend to 0)

#### Improve Efficiency in PDS Algorithms
There are tow basic operations that determinate convergence speed of a PDS algorithms:
1. Choosing a simple location with unbiased probability
2. Testing if a new location is not already covered

To apply these operations we can use a set of tips, some are the following:
- [[Scalloping]]
- [[Hierarchical Dart Throwing (HDT)]]
# References
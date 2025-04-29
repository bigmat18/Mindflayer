**Data time:** 16:29 - 27-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Sampling]]

**Area**: [[Master's degree]]
# Poisson Disk Sampling (PDS)

Poisson Disk Samplig are a class of sampling technique. A Poisson Disk Sampling is a sampling $X = \{(x_i, r_i) : i = 1, \dots, n\}$ such that:

1. **Minimal distance**: $\forall (x_i, x_j) \in ||x_i - x_j|| > \min(r_i, r_j)$
If I put a disk of ray $r$ all the disks do not touch each other.

2. **Unbiased sampling**: The probability of a region to be covered is proportional to its size.
It's difficult to verify, we don't favor a region over another.

3. **Maximal sampling property**: $\Omega \subseteq \bigcup disk(x_i, r_i)$
Mi domain is the union of all the disks that we have placed. This means we don't add any other points.

These property doen't say anything about regularity. About that we can ask is any PDS a «good» sampling? The answer is No, a maximal PDS sampling (for a given radius) is obtained placing samples at the centers of cells of a hexagonal lattice.

![[Pasted image 20250427164333.png | 150]]
##### Non uniform sampling
We can obtained a non uniform sampling, changing the radius of PDS.

![[Pasted image 20250427191658.png | 150]]

##### Subsampling Point clouds
In many real word sampling we optein a points scene where the nearest areas contains more points than distant areas, if we applay sampling tecniques here the probability to take point in nearest areas is higher, we can fix this issues with subsampling.

![[Pasted image 20250427192012.png | 400]]

##### Edge preserving sapling
Sometimes we want sampling a surface favor a specific domain, for example the edges of a mesh. For first thing we put a sample on the corners, after we do the sample for the rest of surface using a similar approach to sub-sampling.

![[Pasted image 20250428163658.png | 320]]

### [[Dart Throwing]]

### [[Scalloping]]

### [[Hierarchical Dart Throwing (HDT)]]

# References
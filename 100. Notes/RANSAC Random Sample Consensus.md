**Data time:** 16:04 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]]

**Area**: [[Master's degree]]
# RANSAC Random Sample Consensus

This is a statistc instrument used in many topic of Computer Science and it is a statistical method that is used to **estimate parameters** of a mathematical model from a set of observed data that contains outliers. It is an **iterative method** (can be interpreted as ma outlier detection method).

In Geometry Processing is used for:
- **[[Surface Recostruction]]**: in particular to find good estimates for image registration of [[Range Maps]]
![[Pasted image 20250509164824.png | 250]]

- **Fragment assembly**: putting together a separate set of elements.
![[Pasted image 20250509164842.png | 250]]

- **Object completion**
![[Pasted image 20250509164900.png | 250]]

- **Protein docking**
![[Pasted image 20250509164915.png | 250]]

### Principal Component Analysis (PCA) 
##### Rigid Matching problem
Rigid matching is a problem in computer vision, pattern recognition and computer graphics communities. It is the process of finding the transformation that maps one rigid object onto another. 

![[Pasted image 20250509165032.png | 300]]

Use PCA to place models into a canonical coordinate frame. We build the matrix of eigenvalue and eigenvectors of a object covariance matrix, intuitively they get the axes of **ellipsoid of inertia** that represent the object.

The **ellipsoid of inertia** is the ellipsoid that has the same mass, center of gravity and moments of inertia (they say how mass is distributed in the object) of the object.

![[Pasted image 20250509165741.png | 400]]

To calculate the ellipsoid of inertia we need da collection of points $\{p_i\}$ form the co-variance matrix
$$c = \frac{1}{N} \sum^N_{i=1}p_i \:\:\:\:\:C= \frac{1}{N}\sum^N_{i=1}p_ip_i^T - cc^T$$
Compute eigenvectors of matrix C.

#### Issues with PCA
The main issues are:
- Principal axes are not oriented.

![[Pasted image 20250509170637.png | 250]]

- Axes are unstable when principal are similar
![[Pasted image 20250509170706.png | 200]]

- Partial similarity

![[Pasted image 20250509170743.png | 350]]


### RANSAC: Basis
RANSAC is the main method to found what fit and discard during the fitting operation. Random Sample Consensus has for core idea is found model parameters to fit two things to do this we hypothesized match can be described by parameters (ess. translation, affine).

In synthesis we match enough features to determine a hypothesis:
- See if it is good
- And repeat until the final results

##### Example
A famous example is the grouping points into Lines. We start with the following dataset:

![[Pasted image 20250509172117.png| 200]]

The classic approach with least square create a line moved for all noise points, the result line isn't good for the major of points. With **RANSAC**:
- we select a random subset of the original data. Call this subset the hypothetical inliers.
- A model is fitted to the set of hypothetical inliers
- All other data are than tested against the fitted model
- The estimated model is reasonably good if enough points have been classified as part of the consensus set
- Afterwards the model may be improved by re-estimating it using all members of the consensus set.

![[Pasted image 20250509174847.png | 200]] ![[Pasted image 20250509174857.png | 200]]

###### Complexity
How many samples do we need to take? 
- p is fraction of points on the line
- n points needed to define hypothesis (2 for lines)
- k number of trials

The probability that after N trials I have the correct solution is: $1 - (1 - p^n)^N$ similar to [[Variabili Aleatorie Notevoli|binomial]].

### More complex fitting
How many point-pairs specify a rigid transform in R2 or R3? We need additional constraints? 

- For **2D** to found translation we need 2 pairs of points, 4 points in total, 2 on one surface and 2 to the other and we need a minimum of coherence (distance among points must be the same)
![[Pasted image 20250509175946.png | 150]]
- For **3D** we need 3 pairs of points, total 6 points. Beyond the distance we need to know they are quite distance.
![[Pasted image 20250509180516.png | 150]]

### Algorithm
The full RANSAC algorithm we need to follow the following steps:

1. Sample tree (two) pairs, check distance constraints 
2. Fit a rigid transform
3. Check how many point pairs agree. If above threshold terminates, otherwise goes to step 1.

![[Pasted image 20250509180854.png | 400]]

This algorithm work well but in 3D the probability begin to decrease a lot why we need 3 points. To fix that we add other measures, it is usually call **feature point detection**
1. Use feature descriptors, they store local information about surface 
2. Denote a larger success rate p
3. Probability a descriptor identifies the correct match
4. Try only candidated made by pairs od samples with similar descriptor 

A basic analysis is:
- The probability of having a valid triplet $p³$
- The probability of having a valid triplet in N trials is $1-(1 - p³)^N$

#### Ransac with Normals
One of feature that can be added is the [[Normals on 3D Models|normal]] and ask how many surfel (position + normal) correspondences specify a rigid transform:
![[Pasted image 20250509182219.png ]]

We need only 2 points with normal to define a transformation from a surface to other, reduce the number of trials from $O(m³)$ to $O(m²)$, the success rate became $1 - (1 - p²)^N$

#### Ransac with Curvature
Other features that can we add is the [[Curvature in 3D models|principal curvature]]. We that one point is enough

![[Pasted image 20250509182913.png | 400]]

Further reduce the number of trials from $O(m²)$ to $O(m)$, with the success rate at $1 -(1-p)^N$. The problem with this feature is that **estimate curvature is harder**.
# References
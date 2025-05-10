**Data time:** 18:33 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]]

**Area**: [[Master's degree]]
# ICP Iterative Closest Point

This algorithm is used after [[RANSAC Random Sample Consensus]] to refine the alignment. It optimize the transformation that map one surface to other. A lot of variants have been proposed for the original algorithm, but the core idea is simple. It works with point cloud and polygonal surfaces.

Just consider for the moment the problem in 2D, let M be a model point set and S be a scene point set, we assume: 
- $n_m = n_s$ with $n \geq 2$
- each point $S_i$ correspond to $M_i$

What do we need to align M to S? We need to compute a **transformation** (represented in mathematics by a matrix). To do that we start from one surface, watch the closest points on the other surface and we search the transformation the align the point of fist surface on the second surface, we need to minimize this transformation.

![[Pasted image 20250510013717.png | 250]]

The idea of this method is iterative and has the following steps:
1. We choose a pair of points and we search the closest points
![[Pasted image 20250510013838.png | 150]]

2. Search a transformation for first couple of points
![[Pasted image 20250510014001.png | 150]]

3. Iterate for each pair of points
![[Pasted image 20250510014027.png | 200]]

If correct correspondences are know, can find correct relative rotation/translation. On average this algorithm converge on 10 iterations.

![[Pasted image 20250510014123.png | 350]]

Throw the iterations the **mean square error** decrease. The MSE is defined in the following way:
$$MSE = \frac{1}{N_S}\sum_{i=1}^{N_S}||\hat{Y} - Y||²$$
where $\hat{Y}$ is a predication value and $Y$ the measured value. Imagine you are the weather forecast reporter and every day you predict the temperature for the next day, MSE will give you an average extimation on how much you predictions were wrong.

### ICP Algorithm
![[Pasted image 20250510015040.png | 500]]

```c
function ICP(Scene,Model)
begin
E` <- + ∞;
(Rot,Trans) <- In Initialize-Alignment(Scene,Model);
repeat
E <- E`;
	Aligned-Scene <- Apply-Alignment(Scene,Rot,Trans);
	Pairs <- Return-Closest-Pairs(Aligned-Scene,Model);
	(Rot,Trans,E`) <- Update-Alignment(Scene,Model,Pairs,Rot,Trans);
Until |E`- E| < Threshold
return (Rot,Trans);
end
```

### ICP Variants
Variant on the following stages of ICP have been proposed:
1. **Selecting samples points**
	- Use all available points
	- Uniform sub-sampling
	- Random sampling in each iteration
	- Ensure that samples have [[Normals on 3D Models|normal]] distributed as uniformly as possible

	![[Pasted image 20250510015713.png | 350]]

we choose points pairs using a distribution that reflect the normals on the surface distribution, the probability that a point can be choose depend by normal direction

2. **Matching to points in the other mesh**
	- Closest point in the other mesh
		![[Pasted image 20250510021546.png | 150]]
		
	- Normal shooting
		![[Pasted image 20250510021626.png | 150]]
		
	- Reverse calibration
		![[Pasted image 20250510021655.png | 150]]
		Project the sample point onto the destination mesh , from the point of view of the destination mesh’s camera.
		
	- Restricting matches to compatible points (color, intensity, normals, curvature ...)

3. **Weighting the correspondences** 
	- Assigning lower weights to pairs with greater point-to-point distance
	$$Wheight = 1 - \frac{Dist(p_1, p_2)}{Dist_{max}}$$
	- Weighting based on compatibility of normalized normals
	$$Weight = n_1 \cdot n_2$$

4. **Rejecting certain (outlier) point pairs.**
	- Corresponding points with point distance higher than a given threshold
		![[Pasted image 20250510023144.png | 120]]
		
	- Rejection of worst n% pairs based on some metric
	- Pairs containing points on end vertices
		![[Pasted image 20250510023259.png | 150]]
		
	- Rejection of pairs that are not consistent with their neighboring pairs $(p_1, p_2), (p_2, q_2)$ are inconsistent if and only if
		$$|Dist(p_1, p_2) - Dist(q_1, q_2)| > threshold$$

		![[Pasted image 20250510023218.png | 120]]


This algorithm work well with a good initial solution which I have the guaranteed to converge to a local minimum.
# References
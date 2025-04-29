**Data time:** 15:36 - 27-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Sampling]]

**Area**: [[Master's degree]]
# Introduction to Sampling

In general this is the process to sample a domain, these techniques are used for example, point based rendering, photo-realistic rendering, Gaussian splatting.

Anther import case of sampling is in remeshing. There are many technique in remeshing that use sampling to found a set of points on a surface, these points then are triangulated. 

![[Pasted image 20250427161143.png | 400]]

Very useful in case of image/video stippling where for example we approssimate a grey scale with a sample process. One of this algorithms is the **Bilateral blue noise sampling**.

### Jittering

![[Pasted image 20250427162217.png | 400]]

- If we get **random** points in a domain with an [[Calcolo combinatorio|uniform distribution]] the domain not uniformly sampled.
- If we use a perfectly **uniform** sampling the domain uniformly but not in a random way.
- The best approach is the **jittered** sampling. It is uniform and random. Trading aliasing for noise.

```js
S JitteredSampoling() {
	for each cell in GRID
		S = S + RandomPointInTheCell()
	return S
}
```

![[Pasted image 20250427162519.png | 550]]


# Reference
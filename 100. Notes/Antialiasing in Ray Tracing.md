---
Data: 2026-10-23T19:32:00
Tags:
  - note
  - youngling
Connection:
  - "[[Ray Tracing Walktrough]]"
Area:
  - "[[Books.base]]"
---
# Anti-aliasing in Ray Tracing

If you zoom into the rendered images so far, you might notice the harsh “stair step” nature of edges in our rendered images. This stair-stepping is commonly referred to as “**aliasing**”, or “**jaggies**”.

When a real camera takes a picture there isn't any aliasing effect, because we can consider that a true images of the world is continuous. Put another way, the world (and any true image of it) has effectively infinite resolution. **We can get the same effect by averaging a bunch of samples for each pixel.** 

With a single ray through the center of each pixel, we are performing what is commonly called **point sampling**


> [!error] The Problem
> The problem with point sampling can be illustrated by rendering a small checkerboard far away. If this checkerboard consists of an 8×8 grid of black and white tiles, but only four rays hit it, **then all four rays might intersect only white tiles**, or only **black**, or **some odd combination**.

In the real world, when we perceive a checkerboard far away with our eyes, we perceive it as a gray color, instead of sharp points of black and white. That's because our eyes are naturally doing what we want our ray tracer to do: **integrate the (continuous function of) light falling on a particular (discrete) region of our rendered image.**

![[Pasted image 20261023214327.png|271]]

We'll adopt the simplest model: sampling the square region centered at the pixel that extends halfway to each of the four neighboring pixels. This is not the optimal approach, but it is the most straight-forward
# References
- [_A Pixel is Not a Little Square_](https://www.researchgate.net/publication/244986797)
---
Data: 2026-10-07T16:54:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computer Vision]]"
Area: "[[Master's Degree.base]]"
---
# Pinhole Camera

Every point on a wall is illuminated by light arriving from multiple directions. The light reflected from the wall integrates light from every reflecting surface in the room, so the reflected intensities are an average of light intensities from many different directions and many different sources.

Mathematically, **integrating** the equation for Lambertian reflections over all possible incoming light directions $p$:
$$
\ell_{\texttt{out}} = \int_{\mathbf{p}}  a \ell_{\texttt{in}}(\mathbf{p}) \cos(\mathbf{n} \cdot \mathbf{p}) \mbox{d} \mathbf{p}
$$
This integral **averages away** all the spatial structure of the scene: the wall just looks like a uniformly lit surface, with no image of the scene visible on it.

The intensity $l_{out}$ thus tells us very little about the incoming light intensity $l_{in}(p)$ from any given direction $p$. To learn about $l_{in}(p)$ **we need to form an image.** The role of a camera is to organize those rays, to convert the cacophony of light rays going everywhere to a set of measurements of intensities coming from different directions in space.

The simplest camera is a **pinhole camera**: A pinhole camera requires a light-tight enclosure, a small hole that lets light pass, and a projection surface where one senses or views the illumination intensity as a function of position.

![[Pasted image 20261007171446.png|510]]

Now for any given point on the projection surface, the light that falls there comes from only **from one direction**. This creates an image of what’s in the world on the projection plane. This breaks the integral above into
a one-to-one (point-to-point) correspondence between scene points and image points.

Notice that the image is:
- inverted compared to the scene (upside down, left-right flipped)
- dimmer than the scene (because only a tiny fraction of the light passes through the hole)
- blurrier than the scene (because the hole is not an ideal point, and diffraction spreads the rays)

In the end the **Pinhome Camera Model Setup** consists of:
- **opaque plane** (the barrier) with a single small hole (the pinhole / camera center / optical center)
- an imaging plane behind the barrier, where the image forms


**Problems** with a real-world pinhome cameras:
- **Blur**: a real pinhole has finite size, so rays from a single scene point pass through different parts of the hole and arrive at different image points, blurring the image
- **Light Gathering**: a small pinhole admits little light, requiring long exposure times

**Solution**: a lens can have a large aperture (more light) while still focusing rays from each scene point to (approximately) a single image point

![[Pasted image 20261007172612.png|349]]

# References
- https://visionbook.mit.edu/imaging.html#sec:pinhole_camera_formation:~:text=The%20Pinhole%20Camera%20and%20Image%20Formation,-Naively
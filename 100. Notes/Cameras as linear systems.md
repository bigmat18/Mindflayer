---
Data: 2026-10-21T16:44:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computer Vision]]"
  - "[[Image Formation]]"
Area: "[[Master's Degree.base]]"
---
# Cameras as Linear Systems

So far, “camera” has meant an ideal pinhole or lens focusing rays onto a sensor. But **any** device that turns a scene into measurements can be described the same way: light is linear, so the scene to sensor map is always some linear map $l_s = Al_w$
- This abstraction lets us reason about any generic imaging setup
- blurry real pinholes
- corner cameras

**Objective**: once we know $A$, recovering the scene $l_w$ from measurements $l_s$ becomes a general problem (inverting, or regularising a linear system)

### Flatland
For the simplicity of visualisation, let’s consider one dimensional (1D) imaging systems. As shown in the sensor is 1D and the scene lives in flatland.
- a vector $l_w \in \mathbb{R}^N$ of scene light intensities (one entry per scene "position" or direction)
- The sensor records a vector $l_s \in \mathbb{R}^M$ of measurements

![[Pasted image 20261021170351.png|409]]

### Cameras as Linear Systems
Because the light is **linear** the relationship between scene and sensor is a **linear map**:
$$
\boldsymbol\ell_{\texttt{s}} = \mathbf{A} \boldsymbol\ell_{\texttt{w}}
\tag{7.1}
$$
- $A$ represents the camera matrix (imaging operator)
- Each row of 𝐴 describes how one sensor measurement is a (weighted) combination of scene intensities, i.e. the camera’s response/sensitivity pattern 

For an **ideal pinhole camera**, each sensor pixel receives light from exactly one scene direction, using the virtual image plane convention, so the mapping is direct (no flip), we have 𝐴 = 𝐼, the identity matrix.

# References
- https://visionbook.mit.edu/camera_as_linsys.html
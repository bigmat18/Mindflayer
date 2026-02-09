---
Data: 2025-12-04T18:54:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Color in Perception
Colour vision can be considered as largely superfluous in modern life yet color is extremely useful in data visualization:
- showing patterns
- labeling
- highlighting

> “Think of color as an attribute of an object rather than its primary characteristic”
### Trichromancy Theory
We have three distinct color receptors on the retina, the cones, which are active at normal light levels. The influence of rods on color perception can be ignored.

![[Pasted image 20260209184836.png | 350]]

Cones are sensitive to **different wavelengths** (short, medium, long). Hence, they absorb light around the spectrum of blue, green, and red. 

The theory says that we perceive color as a three channel system. All color spaces, even if designed to different purposes, are three dimensional.

![[Pasted image 20260209184934.png | 400]]

Since only three different receptors are involved in color vision, **it is possible to match a patch of color light using a mixture of three color lights**, called primaries.

![[Pasted image 20260209185013.png | 450]]

Given a standard set of primaries, one can use a transformation to create the same color on different output devices.

##### CIE RGB Color Matching Function
The RGB color space were created by the CIE (Commission internationale de l'éclairage) in 1931.
![[Pasted image 20260209185150.png | 350]]
##### CIE XYZ Color Space
This is A transformed version of the CIE RGB color matching functions such that:
- Y corresponds to the perceived luminance in well-lit conditions. **How much light looks bright.**
- Z is close to the short cone response. **Very similar to blue cone (the short)**
- X is a mix such that the values become non-negatives. **Linear combination between R,G,B**

For a fixed value of Y, the plane XZ contains all the possible chromaticities at that luminance.
![[Pasted image 20260209185417.png | 450]]
##### CIE xyY Color Space
This is a normlized versione of color space (CIE XYZ). The normalization remvoe the "scale" .(brighness). Given x, y and Y we can came back to CIE XYZ.
![[Pasted image 20260209185553.png | 450]]

with this coordinates we can have the **Gamut** diagram. The colors that a device (a printer, a
monitor) can reproduce
![[Pasted image 20260209185636.png]]

Bach color is a polidromo inside this graph where each vertex are the primary on the display
![[Pasted image 20260209185759.png | 350]]

### Opponent Process Theory
In the late 19th century, German psychologist Hering proposed the theory (later supported by experimental evidence) that cones combines their stimulus forming three pairs of colours that compete together to form the final one. These pairs, called opponent pairs, are:
- black-white
- yellow-green
- yellow-blue

![[Pasted image 20260209190648.png | 500]]

Evidence that supports the theory:
- Naming
- Unique hues
- Neurophysiology

Properties of opponent color channels:
- Spatial resolution
- Shape perception
- Color contrast

Look at this picture for at least 30 (or 60) seconds, and look/focus at the little white dot that is in the middle. Then, switch to a white slide: what you see on the white background is the flag of the United States.
![[Pasted image 20260209191838.png | 400]]

That is because when you are staring at these colors, you are **exciting the same source of these colors**, and when you switch to the white background, since these sensors have been excited for too long, **they inhibit those colors and the only colors**.

##### Color Theories
Trichromacy Theory and Opponent Color Theory work at different levels.
- **Trichromacy Theory** explains what happen at level of **photoreceptors**.
- **Opponent Color Theory** explains what happen at **neural level**.


### Color spaces
##### RGB Color Space
Based on three color channeles that combine toghether can produce any colors.

![[Pasted image 20260209192206.png | 350]]
##### HSV/HSL color space
A cylindrical color space
- H(hue) = tonalità
- S (saturation) = saturazione
- V (value) = brillantezza

![[Pasted image 20260209192320.png | 350]]

Color specification is more natural with HSV/HSL than with RGB. But they are not perceptually uniform spaces (distances calculated in the color space do not correspond to perceptual distances)

![[Pasted image 20260209192358.png | 350]]
##### CIE Lab color space
Perceptually uniform color spaces (remember opponent process theory?). 3 channels:
- Light (brighness)
- Red-green
- Yellow-blue

![[Pasted image 20260209192500.png | 250]]
###### Color differences
The Euclidean distance is meaningful from a perceptual point of view for CIE Lab color space. Delta E (1976) is defined as:
$$
\Delta E^*_{ab} = \sqrt{(L_2^* - L_1^*)² + (a_2^* - a_1^*)² + (b_2^* - b_1^*)²}
$$
where first component is light differences, the second is gree-red axes, thirs blue-yellow axes
- ΔE < 1 : not perceptible, 1 < ΔE < 2 close observation needed to perceive the difference, 2 < ΔE < 10 different but similar color
- Note: Delta E is not perceptually uniform as originally intended, hence superseded by 1994 and 2000 specifications

Having a color space in which equal perceptual distances are equal distances is useful t**o specify color tolerances, color codes, pseudocoloring** (using sequences of colors to represent data values, possibly with perceptually-equal steps)

![[Pasted image 20260209192752.png | 350]]

Though, uniform color spaces only provide a rough first approximation of how color differences will be perceived An important **influencing factor is size** (we are much more sensitive to differences between large patches) 

**Tip**: Use saturated colors when coding small symbols o thin lines, and less saturated colors for large areas

### Color and visualization
##### Luminance and visualization
The **red-green** and **yellow-blue** chromatic channels are each capable of carrying only about 1/3 of the amount of detail carried by the black and white channel (Mullen, 1985).

![[Pasted image 20260209192852.png | 500]]

Purely chromatic differences are not enough to display fine details. Ensure **adequate luminance contrast with the background** (also if colors with different chromaticity are used)

![[Pasted image 20260209192929.png | 450]]
A contrast boundary can improve the readability of colored symbols
##### Saturation and visualization
Use saturated colors for coding small symbols/fine details, and less saturated colors for coding large areas
![[Pasted image 20260209193016.png | 500]]
##### Color for labeling
Post and Greene (1986) carried out an experiment on the naming of colors (210 different colors were shown on a black background in a darkened room)

![[Pasted image 20260209193036.png | 300]]

Only eight colors plus white are consistently named. Though not generally applicable, this
suggests that only few colors can be used as category labels

**12 colors recommended by Colin Ware in its book**: red, green, yellow, blue, black, white, pink, cyan, gray, orange, brown, purple.

Widely agreed-upon category names, and reasonably far apart in color space.

##### Color and semantics
Pay attention to **color convention** and **semantic associations** (red for hot/bad/danger, blue for cold, green for life/go, etc.) as conventions are not universal. The semantic association with gray is that of belonging to an unspecified category (useful for highlighting)

![[Pasted image 20260209193210.png]]

##### Color ramps
![[Pasted image 20260209193222.png | 450]]

- Avoid color ramps problematic for color-blindness people.
- Use **spectrum-based color ramp** when its use is deeply embedded in the culture of the users.
- To **reveal fine details use pseudo-color sequences that varying in luminance**, not only in chomaticity

# References
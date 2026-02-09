---
Data: 2025-12-04T18:37:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Applied Perception

We have seen that information visualization is about transforming data into a visual representation so that a human can extract useful information out of it.

![[Pasted image 20260208202750.png]]

- The **effectiveness** of a visual representation is not arbitrary: it strongly depends on **how the brain works**
![[Pasted image 20260208203656.png | 400]]

- Understanding how perception works can help one make informed decisions about visualization designs
![[Pasted image 20260208203728.png | 500]]

**Example:** when we look at circles, we tend to compare diameters not areas

![[Pasted image 20260208203758.png | 500]]

### Eye Structure
![[Pasted image 20260208203832.png]]
###### Retina
- The retina is composed by a large number of **photoreceptors** (**rods** and **cones**).
- 100 millions of rods, 6 millions of cones.
- **Cones** are concentrated in the fovea (1.5-2 degrees).
- Retinal ganglion cells send information, through the **optic nerve**, to the brain.

###### Rods and Cones Distribution
![[Pasted image 20260208203949.png | 500]]

### Visual Acuity
L’**acuità visiva** è la misura di **quanto finemente** il sistema visivo riesce a distinguere dettagli nello spazio: in altre parole, la **risoluzione spaziale** dell’occhio + cervello in certe condizioni (distanza, illuminazione, contrasto, correzione ottica).

![[Pasted image 20260208204109.png | 200]]

- Points – 1 minute of arc.
- Gratings – 1-2 minutes of arc.
- Letter – 5 minutes of arc.
- Vernier acuity (the ability to see if two segments are colinear) – 10 seconds of arc.

![[Pasted image 20260208204057.png | 400]]

###### Contrast Sensitivity Function (CSF)
La **Contrast Sensitivity Function (CSF)** descrive **quanto bene il sistema visivo umano riesce a percepire differenze di luminanza (contrasto)** in funzione della **frequenza spaziale** del dettaglio osservato.
- high CSG its equal to high sensitivity
- low CSG means low sensitivity

Our perception is sensitive to **pattern contrast**, **frequency** and **orientation**. Also **color** influences the CSF.

![[Pasted image 20260208204317.png | 500]]

### Visual Cortex
**LGN (Lateral Geniculate Nucleus)** forwards pulses to V1. It is also connected with V2 and V3.

![[Pasted image 20260209144432.png]]

- **V1** is the primary visual cortex. It performs edge detection and global organization (inputs from V2, V3).
- **V2** handles depth, foreground, illusory contours.
- **V3** supports global motion understanding.
- **V4** recognizes simple geometric shape.
- **V5/MT**: motion perception integration and eye movements guidance.

###### Receptive Field
The **receptive field** of a cell is the visual area over which **a cell responds to light**. Retinal ganglion cells are **organized with circular receptive fields**. Stimulated:
- on-center they are **excited**
- off-center they are **inhibited**.

![[Pasted image 20260209145541.png]]

A good mathematical model is the Difference of [[Gaussian Curvature|Gaussian]] (DoG):
$$
f(x) = k_1 e^{(-\frac{x}{\sigma_1})²} - k_2 e^{(-\frac{x}{\sigma_2})²}
$$
It is an algorithm to improve characteristics that involves the subtraction of Gaussian from another less Gaussian. This model explain why many neurons behave as bandpass filters in space.

![[Pasted image 20260209150125.png | 350]]
![[Pasted image 20260209150143.png | 350]]
![[Pasted image 20260209150215.png | 350]]

### Mach Banding
Do you remember the problem of costant shading (when you set a constant normal without intermpolation)? This perceptual effect is called **Mach Banding**.

**March Banding**: optical illusion where optical human system exagerate constrast, **Abrupt** (brush) changes are strongly perceived.

![[Pasted image 20260209150527.png | 500]]

### Hermann Grid Illusion
Grey spots appear at the intersections of white grid a block background

![[Pasted image 20260209150652.png | 300]]

The classic explanations:
- **On cross**: center -> high excitation, surround -> high inhibition, result small response
- **On line**: center -> high excitation, surround -> low inhibition, result lower response

![[Pasted image 20260209151015.png | 400]]

Other experiments demonstrate that this theory is insufficient. An alternative theory is that the illusion is due to the S1 type simple cells.

### The Chevreul Illusion
One a sequence of uniform bands is shown, such bands appear darker at one edge.

![[Pasted image 20260209151211.png | 500]]

These visual effects can result in large errors when reading quantitative information map displayed using a **greyscale map**. Use greyscale maps to represent few values.
### Cornsweet Illusion
**Lateral inhibition** can be considered part of an edge detection process in a scene under viewing. Pseudo-edges can be seen depending on the stimulus. The brain **does perceptual interpolation so that regions affected by such edges can appear lighter or darker**. This is called **Cornsweet illusion**

![[Pasted image 20260209151808.png | 500]]

The Cornsweet effect can be used to highlight bounded regions.
![[Pasted image 20260209151829.png | 400]]

Does not use for maps or to compare many values. Use to highlights:
- Bounded regions
- Important items (by reduce luminance contrast of unimportant items)
- Adjust background luminance to obtain better readability

### Eye Movements
- **Saccadic movements**: ballistic movements of the eyes that change the point of fixation. They can be voluntary or stimulus-elicited.
- **Smooth-pursuit movements**: slow tracking movements of the eyes to keep a moving stimulus on the fovea.
- **Vergence movements**: align the fovea of each eye to a target according to its distance.
- **Vestibulo-ocular movements**: stabilize the eyes compensating for head movements.

#### Saccadic Movements and Fixations
![[Pasted image 20260209152118.png | 550]]

- Saccade takes 20-180 ms.
- Both eyes move in the same direction.
- The movement may be not a simple linear trajectory.
- A fixation is composed of slower and fine movements (microsaccades, tremor and drift) that help the eye align with the target.
- A fixation varies between 50-600 ms.
- Typical movements during reading: 2 degrees.
- Typical movements (in general): 2-5 degrees.
- > 20 degrees -> head movement is required.

### Preattentive Processes
Some visual stimulus “pop up” from their surroundings. Initially, researchers thinked that they happened before attention (erroneous). **Attention is a part of the process.**

In the subconscious accumulation of information from the environment:
- all available info are pre-attentive processed
- the the brain filter what is important

When a visual stimulus is preattentive ?
![[Pasted image 20260209152811.png | 350]]

Visual features that are preattentively processed:
- Orientation ; Curvature ; Shape ; Size ; Color ; Light/Dark ; Enclosure ; Concavity/Convexity ; Addition

Some of them are not symmetric. Visual features that are not preattentively processed:
- Juncture ; Parallelism

###### Orientation
![[Pasted image 20260209152909.png | 400]]

###### Shape
![[Pasted image 20260209152926.png | 400]]
###### Color
![[Pasted image 20260209154038.png | 400]]
###### Light/Dark
![[Pasted image 20260209154058.png | 400]]
###### Curvature
![[Pasted image 20260209154119.png | 400]]
###### Length
![[Pasted image 20260209154146.png | 400]]

#### Asymmetry
Some preattentive process are not simmetric:
- Adding marks is more efficient than removing marks.
- Increase sharpness is more efficients than decrease sharpness.
- A big object surrounded by small objects is more efficient than a small object surrounded by big objects.
###### Marks
![[Pasted image 20260209154243.png | 400]]
###### Size Ration
![[Pasted image 20260209154304.png | 400]]
###### Sharpness
![[Pasted image 20260209154322.png | 400]]

#### Combination of Preattentive Features
Note that the combinations of preattentive visual features may not be preattentive. Examples:
- Shape + Color
- Size + Color
- Shape + Motion

###### Example
Where is the red circle ?
![[Pasted image 20260209154442.png | 450]]

![[Pasted image 20260209154454.png | 450]]

### Gestalt Laws
From Gestalt School of Psychology (founded in 1912 by Max Westhemer, Kurt Koffka and Wolfgang Koheler). The first serious attempt to understand pattern perception. The neural mechanisms proposed do not pass the test of the time BUT the laws have proven to be valid.
###### Proximity
Objects close to each other are perceived to form a group.
![[Pasted image 20260209182530.png | 250]]
###### Similarity
Similar objects are perceived to from a group.
![[Pasted image 20260209182556.png | 250]]
###### Connectedness
Connected objects are perceived as related. Connecting different objects with a line is a powerful way to express that there is some relationship between them.
![[Pasted image 20260209182632.png | 250]]
###### Continuity
We expect that a line or an edge continue to follow its direction and does not deviate from it.
![[Pasted image 20260209182728.png | 350]]
###### Simmetry
Objects arranged simmetrically are perceived as forming a visual whole instead of being preceived as separated entities. Simmetry is best perceived for horizontal and vertical axes.
![[Pasted image 20260209182802.png | 350]]
###### Closure
We tend to perceive the complete appearance of an object. Our brain fills the gap in case of missing parts.
![[Pasted image 20260209182837.png | 250]]
###### Common Fate
We tend to perceive as a group objects that moves in the same direction.
![[Pasted image 20260209182912.png | 400]]
###### Figure-Ground
This perceptual effect regards the formation of a figure from the background.
![[Pasted image 20260209182941.png | 200]]

### Müller-Lyer Illusion
These two lines have equal length but we perceive that they have different length. Two explanations:
- Perspective explanation
- Centroid explanation
![[Pasted image 20260209183128.png | 300]]
### Wundt Illusion
Wilhelm Wundt (1832-1920) (“father of experimental psychology”). Not completely explained.
![[Pasted image 20260209183535.png | 200]]
Two straight lines appears distorted for distorted lines on background.

### Hering Illusion
Another similar illusion (inverted effect of Wundt illusion). Possible explanations:
- Lateral inhibition
- Perspective effect
- Temporal delays in visual processing
![[Pasted image 20260209183639.png | 200]]

### Horizontal–Vertical Illusion
Another simple illusion discovered by Wundt. The vertical line **is perceived 30% more length** than the horizontal line. Cross-cultural (small) differences have been noticed. This is true also for intersecting lines.
![[Pasted image 20260209183740.png | 150]]
### Flannery’s Perceptual Scaling
Comparing area is difficult (remember the area of circles just mentioned). When we compare areas the proportions are underestimated (worst for volumes).

Flannery (1970) proposed to compensate the perception by applying a **perceptual scaling factor**. Tufte, in his famous The Visual Display of Quantitative Information (2001), opposed to anything but absolute scaling, i.e. to excludes compensation for human perceptual failings.
![[Pasted image 20260209183843.png | 500]]

Perceptual scaling may be insufficient. Things are more complex from a perceptual point of view that happens for **Heidensberg illusion**
![[Pasted image 20260209183923.png | 350]]
### Weber’s Law
Ernst Heinrich Weber (1795–1878) conducted studies on the perception of physical stimulus by human senses (vision, hearing, taste, touch and smell).
![[Pasted image 20260209184007.png]]
Descrive the relation between actual changes and perceptive changes.

Perception depends by the initial stimulus. Ratios are more important than absolute values.
![[Pasted image 20260209184058.png | 350]]


# References
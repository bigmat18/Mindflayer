---
Data: 2026-02-11T15:02:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Scientific Visualization
Discipline studying computing techniques for the generation of interactive visual representations of acquired or simulated spatio-temporal data (natural link with 3D world).

Scientific visualization as a human activity preceeds by thousands of years visualization as a discipline in computing. The illustration and visual comunication of knowledge are part of our history.

The modern process of visualization consists of **taking raw data** (acquired through sensors or simulated) and **converting it to a form that is understandable to humans**. Data in ever increasing sizes make a graphical approach necessary.

1987: the US National Science Foundation starts “Visualization in scientific computing” as a new discipline, and a panel of the ACM coins the term “scientific visualization”. Interest stimulated by increasing power of scientists’ workstations, progress in algorithms, larger datasets.

**Scientific visualization**: The use of computer graphics for the analysis and presentation of computed or measured scientific data [Oxford English Dictionary, 1989]
- 1990 first dedicated conferences
- Over the years, impressive progress, and now a mature discipline on its own

In Scientific Visualization, a dataset is given by a pair **(domain, attribute)**: an input **geometrical object** (the domain) on which a function is defined, representing **the attributes we want to analyse and visualize**.

![[Pasted image 20260211150604.png | 350]]

The visualization techniques depend on the type of domain and function

### Domain
Dimension (2D surfaces, 3D volumes), Discretization, Geometry (embedding). Data are represented **by finite set of samples**. Data are extended in space **via interpolation schemes**

Different **discretizations**, according to **embedding and connectivity**
- Structured vs unstructured domains, 
- fixed vs arbitrary connectivity

![[Pasted image 20260211153244.png | 500]]

##### Unstructured Points
No explicit connectivity information, irregular geometry (also called point clouds). 

![[Pasted image 20260211153331.png | 350]]

##### Unstructured grids
irregular connectivity and geometry. Different combinations of cells permitted. Popular choices: **triangular** and **tetrahedral meshes** (surfaces and volumes)

![[Pasted image 20260211153428.png | 400]]

### Attributes
Co-domain dimension (scalars, vectors, tensors). 
- **Static vs type-dependent** (applies to attributes only, or to domain + attributes)
- **Deterministic vs uncertain** (e.g., due to sampling of parameter space in simulations– ensamble observations). Visualization of uncertain data is a branch of research on its own

![[Pasted image 20260211153538.png | 500]]

#### [[Scalar Fields Visualization]]

#### [[Vector Visualization]]

#### [[Tersor fields Visualization]]

### Scientific Illustration
Scientific illustration is a discipline that deals with **drawing** or **rendering images** of scientific subjects to inform and communicate.

**Artwork and science**: aesthetics skills in combination with scientifically informed observations. Some examples are Biological illustration, medical illustration, technical illustration..

#### Non-Photorealistic rendering
Contrast to traditional computer graphics which focuses on realism, NPR is inspired by painting, drawing, cartoons...

![[Pasted image 20260212180141.png | 550]]

Example techniques for NPR includes:
- **sparse line drawing**s: drawing of a sparse set of linear contours, which illustrate some features
- **stippling**: creation of a pattern using small dots, which can simulate various degrees of shading and solidity

![[Pasted image 20260212180221.png | 450]]

#### Expressive visualization
Focus on different layers of some object or phenomenon

![[Pasted image 20260212180310.png | 450]]

### Physical Reproductions
The tangible **nature of physical reproductions triggers different cognitive** and **perceptual** **processes** than exclusively visual stimuli alone, and it enables the perception and manipulation of spatial relationships and mechanisms

**For example**, tangible 3D molecular models can conceptualize complex phenomena in a stimulating and engaging format, which well complements computer-generated graphical representations. This is especially true for learning environments.

![[Pasted image 20260212180404.png]]


# References
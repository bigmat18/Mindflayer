**Data time:** 12:08 - 27-03-2025

**Status**: #note #youngling 

**Tags:** 

**Area**: 
# Visualizing Spatial & Geometric data
The position channels in this case is to use the provided spatial position as the substrate for the visual layout. Rather than to visually encode other attributes using the spatial position channel.

### Choropleth maps
Good for overview, less good for accurate comparison. Color channel for quantitative data.
### Symbol maps
A symbol (a disk, a square, a shape) is placed in a point and scaled such that its area proportial to the quantity associated to the point. We can use area shape, and color channels.
### Dot density maps
Mark, typically points, are positioned over a map. 
- Each dots rappreent a diven constant quantity of items (from 1 to N)
- All dots are equal: it's their quogramsantity and distribution that gives information.
### Contiguous cartograms
Geographic position and boundaries among regions are preserved, while dimensions are distorted according to a quantitative attribute
- global procedure
- aim: preserving as much as possible the original position, shape, boundary and minimize distortion
### Grid cartograms
Si cerca di risolvere ancora il problema di disparità di dimensione fra le regioni. Si rappresentano le regioni il, più possibile all'interno di una griglia, si distorgono le regioni, ma avendo un allineamento a griglia sarà minmore rispetto alle contiguos cartograms. 
Sono meno intuitive ed assumono che si conosca la geografia e quindi non c'è bisogno di avere una rappresentazione precisa.
# References
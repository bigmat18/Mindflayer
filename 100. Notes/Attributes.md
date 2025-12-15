---
Data: 2025-12-15T14:02:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Attributes

**Attributes**: some specific property that can be measured, observed, or logged
- in practice, the properties of objects/entities
- also called variables

![[Pasted image 20251215140326.png | 400]]

Disclaimer: different taxonomies in different fields or by different authors. The major distinction is between **categorical vs ordered**.

![[Pasted image 20251215140408.png | 500]]

### Categorical Attributes
**Categorical attributes** are attributes whose values describe categories and that do not have an implicit ordering
- e.g., favorite fruits, movie genre, city names, ...
- they can have an external ordering imposed (e.g., alphabetical ordering of city names, or ordering by population size) but it is not intrinsic to the attribute itself

### Ordered attributes
**Ordered attributes** have an implicit ordering. They can be further subdivided into
- **Ordinal**: attributes which have a well-defined ordering but do not support full-fledged arithmetics, e.g., shirt size, rankings, ...
- **Quantitative**: attributes whose values represent measured quantities/magnitudes and that support arithmetic comparison
	- e.g., temperature, height, weight, price, number of skirts sold, ...
	- their values can be ordered, and also the distance between values can be computed and makes sense

###### Attribute semantics
Ordered attributes can have different semantics:
- **Sequential**: there is a homogeneous range from a minimum to a maximum value
	- e.g., a person’s height
- **Diverging**: there are two sequences  pointing in opposite directions that meet at a common zero point
	- e.g., temperature (it has a zero value, and positive/negative values)
- **Ciclyc**: the values wrap around back to a starting point
	- e.g., hour of the day, day of the week, month of the year, seasons

![[Pasted image 20251215142300.png]]

**Hierarchical attributes**: e.g., type of product (clothes) with subcategories (shirt, skirt, trouser, sweater)
**Temporal attributes**: e.g., day of assumption


One must select appropriate visual representations according to the attribute types and semantics
- Example: sequential color scheme (good idea here) vs diverging color scheme (bad idea here)

![[Pasted image 20251215142422.png | 450]]

- Example:
![[Pasted image 20251215142446.png | 450]]
# References
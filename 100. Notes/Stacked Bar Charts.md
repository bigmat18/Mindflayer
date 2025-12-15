---
Data: 2025-12-15T20:04:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Stacked Bar Charts

A more complex glyph for each bar, with multiple sub-bars stacked vertically. Two key attributes:
- as many composite glyphs/bars as the number of values for the first key attribute, and
- as many subcomponents within each bar as the values for the other key attribute

Both the length of the composite glyph/bar and the length of each subcomponent encode values (multidimensional tables)

![[Pasted image 20251215200618.png | 350]]

###### Example
Visualizing the number of items sold (quantitative attribute) for each store (key attribute #1, categorical) and for item type (key attribute #2, categorical)
- as many bars as the number of categories for key attribute #1 (stores), and as many segments within each bar as the values for the key attribute #2 (type of clothes)

![[Pasted image 20251215200713.png | 600]]

- Color is often used alongside length coding
- The full bar height shows the value for the combination of all items in the stack
- The height of the full combined bars is easy to compare against other bars because they can be read off against a flat baseline (recall: position along a common scale)
- This holds for the lowest bar component as well, while the others are more difficult to compare because their starting points are not aligned
- The order of stacking is significant, as it determines the patterns which are most easily visible

### Normalized stacked bar charts
Variant of stacked bar charts in which the quantitative attribute is normalized. Proportions are shown instead of absolute values
- Example: crop in different sites and regions

![[Pasted image 20251215200849.png | 450]]

### Grouped Bar Charts
- Representation of two key attributes. 
- The same bar chart is repeated multiple times for the number of categories in the second attribute
- Good for comparison of individual values

![[Pasted image 20251215200951.png | 400]]

### Stacked and grouped bar charts

![[Pasted image 20251215201028.png | 550]]

# References
---
Data: 2025-12-15T19:42:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Bar Charts
Bar charts use a line mark and (usually) encode
- a categorical (or ordinal) attribute (key) using position along the horizontal axis as a channel, and
- a quantitative attribute (value) with position along the vertical axis (i.e., the height of the bar)
- Each bar is in a separate region of space, and there is one for each level of the categorical attribute
- Bars are all aligned within a common frame, so that the highest accuracy aligned position channel is used

Useful to visualize how a measured quantity distributes across categories (also for looking up individual values)

![[Pasted image 20251215194437.png | 250]] ![[Pasted image 20251215194456.png | 400]]

Bars can be ordered following
- the alphabetical ordering of the categories: this makes lookup by name easy, but it often hides what could be meaningful patterns in the dataset
- the values of the quantitative attribute that is encoded by the bar heights: this data-driven ordering makes it easier to see dataset trends

![[Pasted image 20251215194622.png | 600]]

Scalability issues: there must be enough room to have white space interleaved between the bar line marks so that they are distinguishable (up to dozen/hundred bars)
# References
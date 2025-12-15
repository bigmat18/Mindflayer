---
Data: 2025-12-15T19:01:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Discriminability

If you encode data using a particular visual channel, are the difference between items perceptible to the human as intended? 

We have to quantify the number of bins that are available for use within a visual channel, with each bin a distinguishable step or level from the other
- e.g., line width: changing the line size only works for a small number of steps. Therefore, line width can work well to show three or four different values of a data attribute, not dozens or hundreds

![[Pasted image 20251215190430.png | 400]]

- Match the ranges: the number of different values that one needs to show for the attribute being encoded must not be greater than the number of bins available for the visual channel used to encode it 
- If they do not match, one should either use a different channel or aggregate the attributes into meaningful bins


# References
---
Data: 2025-12-15T13:37:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Basics of Data Visualization

> **Computer-based** visualization systems provide **visual** **representations** of **datasets** designed **to help people** carry on **tasks** more effectively. By Tamara Munzner, 2014

###### Why have a human in the loop?
Visualization allows people to **analyse data** when they do not know exactly what questions they need to ask in advance
- otherwise, they can use purely computational techniques from e.g. statistics or learning
- example: stock markets vs natural disasters

If there are many possible questions to ask, the best path forward is an analysis process with a human in the loop, with a vis system augmenting human capabilities, rather than replacing it.

###### Why have a computer in the loop?
In the past, visualization systems were drawn by hand, today, most datasets are so large they are infeasible to process and draw by hand, and can also change over time:
- Computers are needed to both analyse the data and create the graphical representation
- Computer-generated graphical representations can be inspired to hand-drawn designs

###### Why use an external representation?
External representations augment human capacity by allowing us to surpass the limitations of our own internal cognition
- Vis allows people to offload internal cognition and memory usage to the perceptual system, using images as external representations (or external memory)
- Replace cognition with perception

###### Why depend on vision?
Visualization exploits the human visual system as a means of communication. A significant amount of visual information process occurs in parallel at the preconscious level, e.g., popout. 

- Find the odd one out (in 0.2 sec)
![[Pasted image 20251215132131.png | 500]]

- Find the odd one out (in 0.5 sec)
![[Pasted image 20251215132211.png|500]]

The other senses can be rules out because of less parallelism (e.g., sonification) or technological limitations (smell, taste, touch)

###### Why show the data in detail?
Vis tools help people in situations where seeing the dataset structure in detail is better than seeing only a brief summary of it.
- Anscombe’s quartet (1973)

![[Pasted image 20251215132324.png | 300]]

![[Pasted image 20251215132354.png | 500]]

![[Pasted image 20251215132420.png | 350]]

Statistical characterization of datasets is a very powerful approach, but it has the intrinsic limitation of losing information through summarization
- A single summary is often an oversimplification that hides the true structure of the dataset
- This applies even more to large and complex datasets!

![[Pasted image 20251215132503.png | 250]]

###### Why focus on tasks?
The task of the user is an equally important constraint as the data we have
- Vis tools can support presentation, discovery, enjoyment of information, or even production of more information for subsequent use
- A vis tool which is fit to a task on a dataset is not necessarily fit to another task on the same dataset (not to mention different datasets)
- We must learn to understand which graphical representation to use depending on both data and task

###### Why focus on effectiveness?
The focus on effectiveness is a corollary of defining vis to have the goal of supporting user tasks, The goals of the designer are not met if the result is beautiful but not effective
- “it’s not just about making pretty pictures”

![[Pasted image 20251215133218.png]]

The vast majority of the possibilities in the design space can be be ineffective in a specific usage context
- maybe because the design is a poor match with the properties of the human perceptual and cognitive system
- maybe because the design is a bad match with the intended task

###### The search space metaphor for vis design
A fundamental principle is to consider multiple alternatives and then choose the best

![[Pasted image 20251215133259.png | 600]]

###### Why use interactivity?
With large datasets, the limitations of both people and displays preclude just showing everything at once. 
- With interaction, user actions cause the view to change
- Interactive vis tools support investigation at multiple levels of details

![[Pasted image 20251215133451.png | 500]]
###### Resource limitations
- Human limitations: memory and attention are finite resources
- Computational capacity: scalability is a concern
- Display limitations
	- Information density: the amount of information in a picture with respect to the unused space

![[Pasted image 20251215133545.png | 250]] ![[Pasted image 20251215133614.png | 350]]

## What-Why-How Method for Visualization Design

- **What**: what data are being visualized? What data are shown in the views?
- **Why**: why does the user need a visualization? Which is the task being performed?
- **How**: how is the vis idiom constructed in terms of design choices?

![[Pasted image 20251215134055.png]]

### What: [[Data and Task Abstractions]]

### Why: 

### How: [[Mark and Channels]]
# References

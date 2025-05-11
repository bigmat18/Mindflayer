**Data time:** 18:39 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Communication Latency

The message-passing paradigm is based of independent entities (processes) having private resources only and communicating using **channels**. **Inter-process communication mechanism** (send and receive) is the effect is that the message value is copied in the target variable.

![[Pasted image 20250511180922.png]]
A linear relationship is usually adopted to predict the **inter-process communication latency**, the model is:
$$L_{com} = T_{setup} + s \cdot T_{transm}$$
where $T_{setup}$ is a constants time and $s$ is the size of the message to transfer.

![[Pasted image 20250511181419.png | 400]]
##### Overlapping
Message-passing parallel programs need a proper support to communication overlapping to mask the impact of the **communication overhead**. The overhead is mostly dominated by a **message  copy** (at least one copy in case of zero-copy implementations).

The **Communication overlapping** are a different scenarios and different implementations/architectural features supporting it. Below 3 cases:

![[Pasted image 20250511181901.png]]

1. Without overlapping (case a)
$$T_{id-Q} = T_{calc-Q} + L_{com}$$
2. Communication overheads totally or overheads partially masked (case b and c)
$$T_{id-Q} = \max\{T_{calc - Q}, L_{com}\}$$

# References
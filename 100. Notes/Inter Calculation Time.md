**Data time:** 19:03 - 11-05-2025

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Inter Calculation Time

#### Inter-Arrival Time
the **inter-arrival time** $T_{A-Q}$ to process Q is the average time interval between the arrival of two consecutive stream items from P

#### Inter-Departure Time
The **inter-departure time** $T_{D-Q}$ time from process Q is the average time interval between the transmission of two consecutive results by Q. It can be calculate by:
$$T_{D-Q} = \max\{T_{A-Q}, T_{id-Q}\}$$
where $T_{A-Q}$ is the inter-arrival time and $T_{id-Q}$ is the [[Ideal Service Time]]
# References
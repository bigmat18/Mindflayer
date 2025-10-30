**Data time:** 19:04 - 11-05-2025

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Utilization Factor

The utilization factor of a process Q is defined as $\rho_{Q} = T_{id-Q}/T_{A-Q}$. That is the [[Ideal Service Time]] and the [[Inter Calculation Time|inter-arrival time]]. Two cases:

- **Bottleneck**: $\rho > 1$ the load is too much for the PE
- **Non-bottleneck**: $\rho < 1$ there is idle-time 
- **Perfect**: $\rho = 1$ the PE is always busy.

If Utilization Factor is too less of 1 is bad because there is a wasted resourced. This because if the ideal-service time is less then inter-arrival time that means that there is idle-time (the time to receive a new message is too slow).
# References
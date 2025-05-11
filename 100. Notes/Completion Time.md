**Data time:** 18:39 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Completion Time

We are often interested in the total execution time to process the whole stream i.e. the **completion time $T_{C-\Sigma}$**. The temporal diagram below shows what happens with a system $\Sigma$ having four replicas of Q.

![[Pasted image 20250511175830.png]]

From $Y_1$ to end all input work in parallel. The following relation:$$T_{C-\Sigma} = T_{filling} + m \cdot T_{\Sigma}$$ where $T_{filling}$ is the initial **filling transient phase**, while $T_{\Sigma}$ is the interdarture time of the system. Since the stream length is infinite or large we have 
$$T_{C-\Sigma}\sim m \cdot T_{\Sigma}$$

# References
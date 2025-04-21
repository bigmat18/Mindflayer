**Data time:** 12:39 - 04-04-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[SLURM Polices Algorithms]]

**Area**: [[Master's degree]]
# Priority queue algorithm

SLURM uses a global queue, where jobs are ordered based on their priority (the highest-priority job gets executed first). Each job is assigned a priority value absed on factors such as:
- **Fairshare** (users with lower recent usage get higher priority)
- **Age** (older jobs gain priority over time)
- **Size**: larger jobs may be prioritized
- **QoS and partition configuration**


###### Queue Management
Jobs are placed in a queue and sorted by priority. If enabled, lower jobs may be **preempted** to make room for higher-priority jobs.
###### Potential Issue
If resources are insufficient, the highest-propriety job may delay other jobs, this cause a **blocking** situations for some jobs.
# References
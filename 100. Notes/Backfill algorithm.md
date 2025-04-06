**Data time:** 12:37 - 04-04-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[SLURM Polices Algorithms]]

**Area**: [[Master's degree]]
# Backfill algorithm

This is the default algorithm attempts to schedule low-priority jobs if they do not prevent higher-priority jobs from starting at the scheduled time. The **goal** of this algorithm is fill in gaps the schedule without delay reserved resources for higher-priority jobs.

![[Pasted image 20250404124445.png | 350]]

- All jobs arrive at the same time ($t_0$) 
- Jobs 1, 2, and 3 start immediately
- Job 4 cannot start. It waits for jobs 2 termination because needs the same resources.
- Job 5 can start immediately if it finishes no later than Job 2 (ensuring Job 4 is not delayed)
- Job 6 must wait until Job 4 completes
- Job 7 can start immediately using the available resources, provided it completes before Job 6 starts

#### Advantages
- Maximize cluster utilization preventing some resources from being idle
- Improve system throughput by executing "small" jobs earlier
- Does not affect high-priority jobs, thus ensuring fairness
#### Disadvantages
- Relies on accurate job time estimate (often users overestimate job duration, leading to inefficiencies)
- May starve some jobs (Low-priority jobs may keep getting backfilled and delayed indefinitely)
# References
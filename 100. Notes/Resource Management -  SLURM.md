**Data time:** 11:58 - 04-04-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] 

**Area**: [[Master's degree]]
# Resource Management -  SLURM

Resource Management Systems (**RMSs**) are software systems that:
- Allocate and schedule computing resources (CPUs, memory, GPUs) in a multi-user, multi-node enviroment.
- Manages user's jobs, their submission for execution
- Optimize resource utilization and fair allocation of such resources among all users.

They are extensively used in HPC clusters, cloud, computing enviroment and data centers. For example **SLURM, PBS, LSF, SGE**

![[Pasted image 20250404120156.png | 350]]

Managing the resources of a supercomputer with thousands of nodes is not trivial, the nodes of a cluster are divided into login (entry) nodes, **RMS** nodes and **computational** nodes (usually, there is more than one login nodes). 

To access the compute nodes you should know which Resource Manager/Job Scheduler is user, in this course cluster, we use **SLURM**.
##### Resource Allocation
- Assign nodes to users
- Static and Dynamic allocation
- Granularity: set of entire nodes, set of cores of single node
##### Manage the schedule
- Decide when a given job starts on the assigned nodes and relinquish the resources at the end of program execution
- FCFS, Backfilling, Priority-Based, Fair-Share
##### Distribute the workloads
- Balancing the work over all available nodes
- Round-robin, load-aware scheduling
- Parallel job distribution, example MPI
##### Monitoring and Accounting
- Display running and waiting jobs
- Account for resource utilization

### SLURM
The **SLURM** is an open-source job scheduler and resource manager for HPC clusters (Simple Linux Utility for Resource Management), it was developed starting from 2001 at the Lawrence Livermore National Laboratory.

- Manages the allocation of computing resources for parallel and batch jobs
- Computiational nodes are grounded into partitions (logicas sets of resources)
- Partitions can be configured in different ways (limiting the number of hours and nodes per job)
- Resources can be allocated on demand with **salloc** command
- It supports job dependencies
- We can use the **sinfo** command to know which partitions are available on the cluster.

#### Policies
SLURM assigns available nodes within a partition until resources are fully allocated. Jobs are scheduled **based on priority** (highest priority first). SLURM determines job execution order using multiple factors:
- Job size and resources requested
- Job age (waiting time in the queue)
- User fair-share usage (to prevent a single user or group from monopolizing the partition/system)
- Partition priority

We have two possible configurable scheduling polices:
##### [[Backfill algorithm]]
It allows lower-priority jobs to start if they don't delay higher-priority ones. It helps to increase cluster utilization by filling in gaps.
##### [[Priority queue algorithm]]
It schedules jobs is strict priority order within each partition/queue. It is used for time-sensitive or reserved workloads.
# References
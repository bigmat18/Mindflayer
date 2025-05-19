**Data time:** 13:18 - 19-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Synchronization Basics

### Problems
##### Mutual Exclusion

![[Pasted image 20250519132014.png]]

**Implementation:**
- **Message-passing model:** S is encapsulate is a process S_control communicating with P and Q (request-only, or asynchronous)
- **Shared-variables model**: S is a shared variable, and atomicity is implemented using mechanisms like **locks**, **monitors** and others.
##### Event Notification

![[Pasted image 20250519132255.png]]

**Implementation:**
- **Message-passing model:** events and ordering relations are implemented by communications through **send** and **receive**
- **Shared-variables model**: ordering relations are implemented using specific mechanisms such as **wait** and **notify** notification primitibes.

**Example**: an integer S initialized to zero, P increments S by one, Q by two.

![[Pasted image 20250519132639.png | 550]]

### Solutions
#### [[Locking|Locking mechanism]]
- Mechanism to deal with **symmetric synchronization**
- The lock acquisition (sometimes also the release in some implementations) operates on **shared variables** and must be **atomic** itself
- Atomicity is the responsibility of the [[Level-based view for parallel computing|firmware level]], ie, **indivisibility bit** to make more accesses to the memory atomic

![[Pasted image 20250519132955.png]]
#### [[Notification|Notify-based mechanism]]
- Mechanism to deal with **asymmetric synchronization**
- On shared-memory architectures, its is implemented through **shared variables** (one boolean flag for each distinct event) or **IO inter-processor communications**
- **No atomicity** issues arises

![[Pasted image 20250519135203.png]]

### Software Lockout
Operating systems are parallel nowadays. OS runtime need to be **reentrant** and **thread-safe** to allow more processors to execute kernel functionalities in parallel. The scheduler data structure are potentially shared by multiple **kernel instances** running in different PEs. Other example are IO drivers and memory management.

![[Pasted image 20250519135424.png | 250]]
- **L/E** where 
	- **L** is the avg length of lock sections 
	- **E** is avg length sections outside locks

Software lockout was a serious problem of first **operating systems** for multi-processors. OSs that are mere extensions of the ones for uni-processors had a very high L/E ratio. Therefore, **modern OSs have been completely rewritten** in order to minimize the impact of software lockout
# References
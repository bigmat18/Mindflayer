---
Data: 2026-01-03T15:01:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Message Passing Interface (MPI)]]"
Area: "[[Master's degree]]"
---
# Group and Communications

A group is an ordered set of processes. Each process in a group is associated with a **unique** rank (from 0 to groupsize - 1). A group is typically associated with a communicator object, **which defines the context**. The group of a communicator can be obtained by calling:

```c++
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
```

MPI provides operations to construct new process groups based on existing groups:

```c++
MPI_Group_union(), MPI_Group_intersection(), MPI_Group_difference(), MPI_Group_incl(), MPI_Group_excl()
```

The code snippet splits the application group with p processes in two groups (even and odd)
###### Example code:
```c++
int even = (p+1)/2;
for(i=0; i < even; ++i) 
	members= 2*i;
MPI_Group, even_group, odd_group;
MPI_Comm_group(MPI_COMM_WORLD, &world_group);
MPI_Group_incl(world_group, even, members, &even_group);
MPI_Group_excl(world_group, even, members, &odd_group)
```

### Communicators
A communicator encompasses a group of processes that can communicate. A communicator binds a **process group** and a **contexts**.
- **Intra-communicator**: used for communication within a group
- **Inter-communicator**: used for cross groups communications

 Purposes: collective communications, user-defined virtual topologies. MPI provides more than 40 functions/routines related to groups, communicators and virtual topologies. Some of them related to communicator management are:
 - `int MPI_Comm_create(MPI_Comm old_comm, MPI_Group group, MPI_Comm* new_comm)`
 
 - `int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int* res)`
 
 - `int MPI_Comm_dup(MPI_Comm comm, MPI_Comm* new_comm)`
 
 - `int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* new_comm)`
	 - Useful to partition communication among process subsets. It is a collective operation
	 - It partitions comm into disjoint subgroups, one for each value of color
	 - Within each group, the MPI processes are ranked in the order defined by the value of the key, with ties broken according to their rank in the parent group (if all keys are the same, then all processes will have the relative rank order as they have in the parent group)

###### Example code
```c++
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int color = rank/4; // one color for each logical row
MPI_Comm_row_comm; // new row communicator
// the ordering of ranks in new_comm is the original one
MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm)
```

![[Pasted image 20260103151954.png | 400]]

#### Virtual Topologies
A virtual topology represents the way that MPI processes communicate 
- logical process arrangement in regular topological patterns such as 2D or 3D grid
- logical process arrangement can also be described by a graph

**Cartesian Topologies**: for regular problems, MPI provides a convenient multidimensional mesh organization
```c++
int MPI_Cart_create(MPI_Comm comm_old, int ndims, int* dims, int* periods, int reorder, MPI_Comm* comm_cart)
```

**Graph Topologies:**
```c++
int MPI_Graph_create(MPI_Comm comm_old, int nnodes, int* index, int* edges, int reorder, MPI_Comm* cgraph
```



# References
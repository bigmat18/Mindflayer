**Data time:** 14:29 - 05-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Message Passing Interface (MPI)]]

**Area**: [[Master's degree]]
# Symmetric Communication

### Synchronous Communication
Let’s consider two logical primitives: `ssend` and `srecv`. They return when the message transfer is completed:
- `ssend` (synchronous send): Waits until the complete message can be accepted by the receiving process before sending the message
- `srecv` (synchronous receive): Waits until the message arrives

In MPI, `ssend` is implemented by `MPI_Ssend` and `srecv` is implemented by `MPI_Recv`, which is actually a blocking receive. `ssend` performs two actions: **transfers** data and **synchronizes processes**

![[Pasted image 20250605143256.png | 300]]

###### `ssend` and `srecv` using a 3-way protocol
![[Pasted image 20250605143326.png | 550]]

### Asynchronous communication
Communication routines that do not wait for actions to complete before returning. Usually require local storage for message buffering, in general, there is no synchronization between communicating processes.

![[Pasted image 20250605143428.png | 550]]

### Blocking
A blocking call return after its local actions are completed (ie, the send buffer is free to be reused), even though the message transfer may not have been completed. With limited buffer capacity (MPI internals), a blocking send turns out to behave as a synchronous send
#### Send
```c
int MPI_Send(void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm)
```
The message buffer is fully defined by the triple <buf, count, dt>
- **`count`** is the number of items to send, NOT the number of bytes
- **`dest`** is the destination rank in the communicator **comm**; if `MPI_PROC_NULL`, the send has no effect (i.e., completes immediately)
- **`tag`** is an integer for message classification
- **`comm`** selects the communicator context

Once `MPI_Send` returns, the buffer can be safely reused. The actual message may still be “in flight” and not yet received by the peer. If the MPI’s internal buffering is exhausted, `MPI_Send` may block until the
matching `MPI_Recv` is posted (i.e., rendez-vous). This is still compliant with the blocking send semantics.
#### Receive
```c++
int MPI_Recv (void* buf, int count, MPI_Datatype dt, int source, int tag,
			  MPI_Comm comm, MPI_Status* status)
```
The message buffer is fully defined by the triple <buf, count, dt>
- receiving fewer elements than **`count`** is allowed
- **`source`** is the source rank in the communicator **`comm`**; `MPI_ANY_SOURCE` allows the user to receive from any source rank in the communicator `comm`

After `MPI_Recv` waits until a matching (both `source` and `tag`) message is entirely received
- No partial read of the message is possible. If the sender sends K items, a matching receive will return when all K items have been received
- On return, **status** contains the actual source, **tag**, and count (via `MPI_Get_count`)
#### Communication and Deadlock
```c
if (myrank == 0) {
	MPI_Send(buf, count, MPI_INT, 1, 200, MPI_COMM_WORLD);
	MPI_Recv(buf, count, MPI_INT, 1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
} else if (myrank == 1) {
	MPI_Send(buf, count, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
	MPI_Recv(buf, count, MPI_INT, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
}
```
The MPI_Send may block until the message is received by the destination process. If it blocks, then deadlock!

Another case: sending data to the right-hand side neighbor on a ring:
- The same problem as Dijkstra’s "Dining Philosopher problem"
- Deadlock can be easily removed by breaking the circular wait
```c
if (myrank%2 == 1) {
	MPI_Send(…); MPI_Recv(…);
} else {
	MPI_Recv(…); MPI_Send(…);
}
```
Why might `MPI_Send` get blocked? **It returns when it is safe to reuse the application buffer**

The MPI standard permits the use of a system buffer but does not necessarily require it. Some implementations may use a synchronous protocol to implement the blocking send. For large enough messages, the `MPI_Send` is blocking

![[Pasted image 20250605170831.png | 600]]

### Non-blocking
A non-blocking call returns immediately after the operation is started
- The transfer continues asynchronously in the background
- It assumes that the data storage used for the transfer is not modified. Do not modify (or free) the user buffer until you know the operation has completed! It is left to the programmer to ensure this (e.g., by using `MPI_Wait`/`MPI_Test` or their variants)
- Non-blocking calls let you **overlap communication and computation**
#### Send
```c
int MPI_Isend(void* buf, int count, MPI_Datatype dt, int dest, int tag, 
              MPI_Comm comm,MPI_Request* req)
```

#### Receive
```c
int MPI_Isend(void* buf, int count, MPI_Datatype dt, int dest, int tag, 
              MPI_Comm comm, MPI_Request* req)
```

Processing continues immediately without waiting for the completion of the operation. A communication request handle (**`req`**) is returned for handling the pending message status.

By calling `MPI_Wait`() or `MPI_Test`() (or their \*all variants), it is possible to determine when the operation completes. Only then is it safe to reuse (or free) the message buffer. It is okay to also mix non-blocking and blocking calls:
```
MPI_Isend() <--> MPI_Recv(), and, MPI_Send <--> MPI_Irecv()
```

#### Waiting and checking non-blocking calls
```c
int MPI_Wait(MPI_Request* req, MPI_Status* status)
```
- Blocks until a non-blocking send or receive operation has completed
- For waiting multiple non-blocking operations: `MPI_Waitany`(), `MPI_Waitall`(), `MPI_Waitsome`()

```c
int MPI_Test(MPI_Request* req, int* flag, MPI_Status* status)
```
- Check the status of a non-blocking send or receive
- The flag parameter is set to 1 if the operation has completed, 0 otherwise
- For checking multiple non-blocking operations: `MPI_Testany`(), `MPI_Testall`(), `MPI_Testsome`()

### Message exchange
```c
int MPI_Sendrecv(void* sndbuf, int sndcount, MPI_Datatype snddt, int dest, int sndtag,
				 void* rcvbuf, int rcvcount, MPI_Datatype rcvdt, int source, int rcvtag,
				 MPI_Comm comm, MPI_Status* status)
```
- Exchanges messages in a single call (both send and receive
- Why `Sendrecv`? **To avoid deadlock MPI handles deadlock issues**
- `MPI_Sendrecv_replace` uses the same buffer for both send and receive

Besides avoiding deadlocks, non-blocking communication is usually employed to overlap
computation and communication

###### Example: ping-pong on an ordered ring
```c++
int main(int argc, char *argv[]) {
    int myId; // Rank of the current process
    int numP; // Total number of processes

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &numP);

    // Get the ID (rank) of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    // Check for command-line argument
    if (argc < 2) {
        // Only the first process (rank 0) prints the error message
        if (!myId) { // Equivalent to if(myId == 0)
            std::cout << "ERROR: The syntax of the program is " << argv[0]
                      << " num_ping_pong" << std::endl;
        }
        // All processes abort
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_ping_pong = std::atoi(argv[1]); // Number of ping-pong iterations
    int ping_pong_count = 0; // Counter for current ping-pong iteration for this process
    int next_id = myId + 1;  // ID of the next process in the ring
    int prev_id = myId - 1;  // ID of the previous process in the ring

    // Handle ring topology (wrap around)
    if (next_id >= numP) {
        next_id = 0; // Last process sends to process 0
    }
    if (prev_id < 0) {
        prev_id = numP - 1; // Process 0 receives from the last process
    }

    MPI_Request rq_send, rq_recv; // MPI request handles for non-blocking operations
    MPI_Status status;            // MPI status object for receive operations

    // Main loop for ping-pong iterations
    while (ping_pong_count < num_ping_pong) {
        // Increment local count for this iteration
        ping_pong_count++;

        // Phase 1: Send to next_id, Receive from prev_id
        // Non-blocking send of the current ping_pong_count to the next process
        MPI_Isend(&ping_pong_count, 1, MPI_INT, next_id, 0, MPI_COMM_WORLD,
                  &rq_send);
        std::cout << "Process " << myId << " sends PING number "
                  << ping_pong_count << " to process " << next_id
                  << std::endl;

        // Non-blocking receive from the previous process, data will be stored in ping_pong_count
        MPI_Irecv(&ping_pong_count, 1, MPI_INT, prev_id, 0, MPI_COMM_WORLD,
                  &rq_recv);
        // Note: This cout prints ping_pong_count *before* it's updated by the MPI_Irecv
        std::cout << "Process " << myId << " (expects to) receive PING data (current val: "
                  << ping_pong_count << ") from process " << prev_id
                  << std::endl;

        // Wait for the receive from prev_id to complete.
        // ping_pong_count is updated here with the value received from prev_id.
        MPI_Wait(&rq_recv, &status);
        // At this point, ping_pong_count holds the value sent by prev_id.

        // Phase 2: Send to prev_id, Receive from next_id
        // Non-blocking send of the (potentially updated by prev_id) ping_pong_count to the previous process
        // Note: rq_send is reused. The buffer &ping_pong_count now holds data from prev_id.
        MPI_Isend(&ping_pong_count, 1, MPI_INT, prev_id, 0, MPI_COMM_WORLD,
                  &rq_send);
        std::cout << "Process " << myId << " sends PONG number "
                  << ping_pong_count << " to process " << prev_id
                  << std::endl;

        // Non-blocking receive from the next process
        MPI_Irecv(&ping_pong_count, 1, MPI_INT, next_id, 0, MPI_COMM_WORLD,
                  &rq_recv);
        // Note: This cout prints ping_pong_count *before* it's updated by this MPI_Irecv
        std::cout << "Process " << myId << " (expects to) receive PONG data (current val: "
                  << ping_pong_count << ") from process " << next_id
                  << std::endl;

        // Wait for the receive from next_id to complete.
        // ping_pong_count is updated here with the value received from next_id.
        MPI_Wait(&rq_recv, &status);
        // At this point, ping_pong_count holds the value sent by next_id.
    }

    // Terminate MPI environment
    MPI_Finalize();

    return 0;
}
```

###### Example: Numerical Integration
MPI parallelization: Split the interval into P sub-intervals, where P is the number of processes used.
Each process computes the area of the local interval and sends the partial results to the master (process
with rank 0), which computes the final sum. Simple **[[Map Parallelization|Map-Reduce]] computation**.
![[Pasted image 20250605174336.png | 250]]

```c++
int main(int argc, char *argv[]) {
    const double a = 0.0;        // Start of the interval
    const double b = 1.0;        // End of the interval
    const int n = 10000000;      // Number of trapezoids (steps)
    double partial_result;       // Partial result computed by each process
    double result = 0.0;         // Final result, aggregated by master
    int myrank;                  // Rank of the current MPI process
    int size;                    // Total number of MPI processes

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process computes a partial result by calling the trapezoid function.
    // The trapezoid function would need to know which part of the work to do.
    partial_result = trapezoid(myrank, size, a, b, n);

    if (myrank != 0) { // All processes except the master (rank 0)
        // Send their partial result to the master process (rank 0).
        // Tag 100 is used for this message.
        MPI_Send(&partial_result, 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
    } else { // This is the master process (rank 0)
        // The master starts with its own partial result.
        result = partial_result;
        // It then collects partial results from all other processes.
        for (int p = 1; p < size; ++p) {
            // Receive a partial result from process 'p'.
            // The tag must match the sender's tag (100).
            // MPI_STATUS_IGNORE means we are not interested in the status object.
            MPI_Recv(&partial_result, 1, MPI_DOUBLE,
                     p, // <-- Potential deadlock: Master waits for process 'p' 
                        // specifically.
                        // If process 'p' is delayed or sends arrive out of order (p+1 
                        // before p),
                        // the master will wait for 'p' even if other messages are ready.
                        // Using MPI_ANY_SOURCE is generally safer for this pattern,
                        // allowing the master to process messages as they arrive.
                     100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Add the received partial result to the total.
            result += partial_result;
        }
        // Master prints the final aggregated result.
        // Using C-style printf for formatted output.
        std::printf("Area: %.18f\n", result);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
```


###### Example: [[Cyclic Computation Graphs|Server-Clients]]
Clients send a random number of requests with TAG `msg_tag` to the Server and then a final message with TAG `eos_tag` to signal the End-`Of`-Stream (EOS). The Server (rank 0) accepts requests from
`MPI_ANY_SOURCE` and with `MPI_ANY_TAG` until all Clients send the EOS message.

![[Pasted image 20250605175005.png | 250]]

### Double Buffering
Technique used to hide communication latency and increase throughput. Used mainly for small/medium-sized data since it requires more memory. It uses two communication buffers per neighbor peer (e.g., `bufA`, `bufB`).
- While `bufA` is «in flight», `bufB` can be used to compute
- On completion, swap buffers, so communication of the next round overlaps computation of the current round

```c++
void *rbufA, *rbufB; // receive buffers
MPI_request *rA, *rB; // receive requests
void *sbufA, *sbufB; // send buffers
MPI_request *sA, *sB; // send requests
MPI_Irecv(…rbufA, rA );
MPI_Irecv(…rbufB), rB;
while(true) {
	MPI_Wait(rA,…); // wait for recv completion
	pw = rbufA; // current work buffer
	swap({rbufA,rA}, {rbufB, rB}); // swap pointers
	MPI_Irecv( …rbufA, rA); // re-post a receive
	do_work(pw); // work on the received data
	MPI_Wait(sA, …); // wait for send completion
	copy(pw, sbufA);
	MPI_Isend(…,sbufA, sA); // send data
	swap({sbufA,sA}, {sbufB, sB}); // swap pointers
}
```

![[Pasted image 20250605175304.png | 450]]


### Farm skeleton in MPI
In this example, we implement a classical (standalone) [[Farm]] skeleton (Emitter-Workers-Collector) employing **asynchronous communications** and **double buffering** in all stages to maximize the overlap of computation and communication.

Possible extensions: within a Worker, we can use OpenMP/CUDA/FastFlow to reduce Worker’s
service time. **NOTE**: If using multi-threading in MPI, you must use `MPI_Init_thread` instead of `MPI_Init`

![[Pasted image 20250605175439.png | 350]]


# References
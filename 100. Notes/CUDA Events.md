**Data time:** 13:28 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Concurrency and Streams]]

**Area**: [[Master's degree]]
# CUDA Events

An event is a special marker (a sort of **punctuation**) inserted in a [[CUDA Streams|stream]]. It can be used to check if the execution of tasks in a stream has reached that specific marker.

![[Pasted image 20250602133000.png | 450]]

CUDA events can be utilized for two main purposes:
- To **synchronize** the execution of stream
- To **monitor** the device progress

The CUDA API provides some primitives to insert (**record**) an event in any position of a CUDA stream, and to query the device to check if the event has been reached or not.

Events on the **default stream** synchronize all previous operations issued on any stream (under legacy semantics).

### API
An event is a special data type in CUDA with name `cudaEvent_t`. Primitive to create a new event
```c
cudaError_t cudaEventCreate(cudaEvent_t *event);
```
Each event has an internal **Boolean state** indicating whether the event has occurred or not yet. To eliminate an event:
```c
cudaError_t cudaEventDestroy(cudaEvent_t event);
```
To register an event in a specific CUDA stream
```c
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
```
To block the host program until the stream execution reaches the given event
```c
cudaError_t cudaEventSynchronize(cudaEvent_t event);
```
When the host program returns from the primitive above, the event has been occurred (internal state is `true`)

To probe the state of an event without blocking the host program:
```c
cudaError_t cudaEventQuery(cudaEvent_t event);
```
It is possible to block the processing of tasks on a given stream on the device until the event has been reached (without blocking the host program)
```c
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event);
```

In the above primitive, all future tasks that will be enqueue on the given stream will be processed by the device when the event status becomes `true`. It is also useful, often for profiling reasons, to measure the
elapsed time between two events.
```c
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
```

### Cross-Stream Synchronization
Consider the code fragment below:
```c
cudaEvent_t event;
cudaEventCreate(&event);
kernel1<<,,, stream1>>();
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
kernel2<<,,, stream2>>();
cudaEventDestroy(event);
```
It is worth noting that `cudaStreamWaitEvent` is called on **stream2** related to an event recorded to **stream1**. We are creating a **cross-stream synchronization**. It corresponds to the following temporal diagram:

![[Pasted image 20250602133735.png | 500]]

### Measuring Elapsed Time
Code fragment to show how to measure the elapsed time between two events

```c
// create two events
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
// record ‘start’ event on the default stream
cudaEventRecord(start);
// execute kernel
kernel<<<grid, block>>>(arguments);
// record ‘stop’ event on the default stream
cudaEventRecord(stop);
// wait until the stop event completes
cudaEventSynchronize(stop);
// calculate the elapsed time between two events
float time;
cudaEventElapsedTime(&time, start, stop);
// clean up the two events
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

In the example above, we are using the default stream (any other user-defined stream could have been used)
# References
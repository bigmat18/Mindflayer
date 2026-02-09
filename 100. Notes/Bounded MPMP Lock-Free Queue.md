---
Data: 2026-02-05T03:34:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Atomic Operations & Memory Consistency]]"
Area: "[[Master's degree]]"
---
# Bounded MPMP Lock-Free Queue

Use case: implementing a «simple» lock-free bounded MPMC FIFO queue using CAS-loop for the enqueue (push) and dequeue (pop) operations. 

![[Pasted image 20260205033552.png | 300]]

- The buffer is a ring-buffer with two indexes (**pread** and **pwrite**) pointing to the head and tail of the queue
- The buffer stores pointers encapsulated in a node containing the pointer to data and a long atomic index (seq) that is the sequence of elements stored in the queue. It enable threads to book a node. The sequence monotonically increases.
- **push**: If the sequence is equal to the tail (pwrite), then try to book the node with a CAS operation atomically updating the tail pointer. If it succeeds the sequence is updated after storing the data.
- **pop**: If the head pointer (pread) is equal to the next sequence element, then try to book the node with a CAS operation atomically updating the head pointer. If it succeeds the sequence is updated after reading the data to the next sequence value

```c++
// MPMC (Multi-Producer Multi-Consumer) bounded lock-free queue of pointers.
//
// This is a classic ring-buffer MPMC algorithm (Dmitry Vyukov style):
// - There is a circular array of slots (elements).
// - Each slot has a sequence number "seq" that encodes the slot state.
// - Producers/consumers advance global indices (pwrite/pread) with CAS.
// - acquire/release on seq is what publishes/consumes the slot contents.
//
// Assumptions / missing pieces in your snippet (must exist elsewhere):
// - BACKOFF_MIN, BACKOFF_MAX (e.g., 1 and 1024)
// - CACHE_LINE_SIZE (e.g., 64)
// - isPowerOf2(size), nextPowerOf2(size)
// - #include <atomic>, <thread>, <cstddef>

template <typename T>
class MPMC_Ptr_Queue {
private:
  struct element_t {
    // Sequence number used to coordinate producers/consumers for this slot.
    // Its value determines whether the slot is:
    // - available for a producer (seq == index)
    // - contains valid data for a consumer (seq == index + 1)
    // - etc., as the ring wraps around
    std::atomic<unsigned long> seq;

    // The payload: raw pointer stored non-atomically.
    // Safety relies on acquire/release on seq to publish/consume this field.
    T* data;
  };

public:
  MPMC_Ptr_Queue() = default;

  ~MPMC_Ptr_Queue() { delete[] buf; }

  bool init(size_t size) {
    if (size < 2) size = 2;

    // The ring buffer uses index & mask instead of modulo.
    // That requires size to be a power of 2.
    if (!isPowerOf2(size)) size = nextPowerOf2(size);
    mask = static_cast<unsigned long>(size - 1);

    buf = new element_t[size];
    if (!buf) return false;

    // Initialize each slot:
    // - data starts empty
    // - seq is set to its "logical index" i, meaning:
    //   at global write position pw == i, the slot is available.
    for (size_t i = 0; i < size; ++i) {
      buf[i].data = nullptr;
      buf[i].seq.store(static_cast<unsigned long>(i),
                       std::memory_order_relaxed);
    }

    // Global producer and consumer positions start at 0.
    // relaxed is fine at init time (no concurrent access yet).
    pwrite.store(0, std::memory_order_relaxed);
    pread.store(0, std::memory_order_relaxed);
    return true;
  }

  bool push(T* const data) {
    unsigned long pw, seq;
    element_t* node;

    unsigned long bk = BACKOFF_MIN;

    // CAS loop: reserve a slot by incrementing pwrite.
    for (;;) {
      // Load current producer index.
      // relaxed: we only use it to pick a slot and attempt CAS.
      pw = pwrite.load(std::memory_order_relaxed);

      // Pick the slot in the ring.
      node = &buf[pw & mask];

      // Read slot sequence number.
      // acquire: if the slot is in "ready" state for us, we must
      // also see any writes that happened-before the seq was released
      // by the consumer (i.e., when it freed the slot).
      seq = node->seq.load(std::memory_order_acquire);

      // If seq == pw, the slot is free for this write position.
      if (pw == seq) {
        // Try to claim this write position.
        // If CAS succeeds, we have exclusive right to write this slot.
        //
        // NOTE: Many implementations use memory_order_relaxed for this CAS
        // because ordering is carried by node->seq release/acquire.
        // Also note: your code passes only one memory_order; that selects
        // the same order for success/failure (both relaxed).
        if (pwrite.compare_exchange_weak(pw,
                                        pw + 1,
                                        std::memory_order_relaxed)) {
          break;  // reserved slot successfully
        }

        // CAS failed (contention). Backoff to reduce cache-line ping-pong.
        for (unsigned i = 0; i < bk; ++i) {
          std::this_thread::yield();
        }
        bk = (bk < BACKOFF_MAX) ? (bk << 1) : BACKOFF_MAX;

      } else if (pw > seq) {
        // If seq < pw, the consumer has not yet advanced the slot enough
        // to make it free for this pw => queue appears full.
        //
        // Intuition: producer has wrapped and caught up to consumer.
        return false;
      }

      // else: seq > pw means the slot is for a future cycle; retry.
    }

    // We own this slot now. Write the payload (non-atomic).
    node->data = data;

    // Publish the payload by advancing seq.
    // release: ensures node->data write becomes visible to a consumer
    // that later does seq.load(acquire) and observes seq == pw+1.
    node->seq.store(seq + 1, std::memory_order_release);

    return true;
  }

  bool pop(T*& data) {
    unsigned long pr, seq;
    element_t* node;

    unsigned long bk = BACKOFF_MIN;

    // CAS loop: reserve a slot to read by incrementing pread.
    for (;;) {
      // Load current consumer index.
      pr = pread.load(std::memory_order_relaxed);

      // Pick the slot in the ring.
      node = &buf[pr & mask];

      // Read slot sequence number.
      // acquire: if we observe that the producer published the slot,
      // we must also see node->data written before the producer's
      // seq.store(..., release).
      seq = node->seq.load(std::memory_order_acquire);

      // In this algorithm a slot is "full" for consumer position pr when:
      //   seq == pr + 1
      //
      // diff == 0 means full and readable.
      long diff = static_cast<long>(seq) - static_cast<long>(pr + 1);

      if (diff == 0) {
        // Try to claim this read position.
        if (pread.compare_exchange_weak(pr,
                                        pr + 1,
                                        std::memory_order_relaxed)) {
          break;  // reserved slot successfully
        }

        // CAS failed due to contention; backoff.
        for (unsigned i = 0; i < bk; ++i) {
          std::this_thread::yield();
        }
        bk = (bk < BACKOFF_MAX) ? (bk << 1) : BACKOFF_MAX;

      } else if (diff < 0) {
        // seq < pr+1 => slot not yet published by any producer for
        // this cycle => queue empty.
        return false;
      }

      // else diff > 0 => slot corresponds to a later cycle; retry.
    }

    // Now we own this slot for reading.
    data = node->data;

    // Mark slot as free for the next producer cycle.
    //
    // After a consumer reads slot at position pr, it sets seq to:
    //   pr + mask + 1  == pr + buffer_size
    //
    // That matches the next time a producer wraps around to the same slot:
    // producer expects seq == pw, and pw will eventually equal pr+size.
    node->seq.store(pr + mask + 1, std::memory_order_release);

    return true;
  }

private:
  // These are separated onto different cache lines to reduce false sharing:
  // many producers hammer pwrite, many consumers hammer pread.
  alignas(CACHE_LINE_SIZE) std::atomic<unsigned long> pwrite;
  alignas(CACHE_LINE_SIZE) std::atomic<unsigned long> pread;

  element_t* buf = nullptr;
  unsigned long mask = 0;
};
};
```
# References
---
Data: 2026-04-19T20:02:00
Tags:
  - note
  - youngling
Connection:
  - "[[Machine Learning]]"
  - "[[Neural Networks (NN)]]"
Area: "[[Master's degree]]"
---
# Backpropagation

**The problem is the same we have seen** for the other models, starting again with the **LMS** approach (error/data term):
- **GIven** a set of $l$ training example $(x_p, d_p)$ and a (inner) loss measure $L$ (e.g. $L(h(x_p), dp) = (dp - h_w (x_p))^2$ for the MSE)
- **Find**: The weight vector $w$ that minimizes the expected error on the training data (we first focus on the data term), **by computing the gradient of the error function**
$$
E(w) = R_{emp} = \frac{1}{l}\sum^l_{p=1}(d_p - h(x_p))^2
$$
**What we need**: differentiable loss, differentaible activation functions, a netowork to follow the information flow.

### Properties
There are nice properties of the backpropagation algorithm (also for programming):
- Easy because of the compositional form of the model
- It keeps track only of quantities local to each unit (by local variables) -> modularity of the units is preserved
- **Efficiency**: O(W) instead of $O(W^2)$ 
- On the biological plausibility: controversial, but it has been supposed that in the brain, to learn, we have a local suboptimal approximation of BP



# References
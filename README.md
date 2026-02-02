# From-Scratch Transformer (C++)

A small Transformer-style language model implemented **entirely from scratch in C++**, using no external LLM libraries.

This project exists to learn and experiment with the *actual mechanics* of Transformers: attention, softmax, loss, and backprop, without ML frameworks.

---

## What this is

* A **minimal Transformer language model**
* Single-head causal self-attention
* Trained with **per-sequence SGD** (no batching)
* Explicit forward + backward passes
* Multithreaded training with shared weights

The model is intentionally **low-capacity**. The implementation is intentionally **explicit**.

---

## What this is not

* Not a production LLM
* Not optimized for speed or scale
* Not using PyTorch, TensorFlow, CUDA, or autograd

---

## High-level structure

* Token embeddings
* Q / K / V projection matrices
* Causal self-attention
* Linear output projection
* Softmax + cross-entropy loss
* Manual backpropagation

---

## Training

* Updates happen **once per sequence**
* Gradients are clipped
* Weights are shared across threads and updated under a mutex
* Loss is printed per sequence for debugging and inspection

---

## Purpose

This is a **learning and experimentation project**.
When I started this, I had NO idea how LLMs worked, at all. This project is my attempt at mastering LLMs, by making one.

It is meant to answer questions like:

* What is a transformer?
* What are forward/backward passes?
* How does attention actually work?

---

## Notes

* Expect noisy loss curves
* Expect fast early loss drops on easy tokens
* Expect poor text quality

That behavior is normal for a model this small.

---

## License

Use it, break it, learn from it - I know I did!

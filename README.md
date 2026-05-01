# mnist-cuda-hpc-optimization
A bottom-up High-Performance Computing (HPC) study of Multi-Layer Perceptrons on MNIST dataset: Benchmarking performance evolution from PyTorch->NumPy->C-> Custom CUDA kernels.

---

## Overview:

This project explores how abstraction vs control impacts performance in deep learning systems by progressively implementing the same 2-layer MLP across multiple levels:

PyTorch → NumPy → C (CPU) → CUDA (Naive) → CUDA (Optimized)

The goal is to:
Understand how neural networks actually work under the hood
Analyze performance bottlenecks
Apply HPC optimization techniques to accelerate training
Hardware Requirements
NVIDIA GPU (CUDA-enabled)
Recommended: RTX series or above
#Tested on:
GPU: NVIDIA RTX 2060 Super (8GB VRAM)
CPU: AMD Ryzen 5 3600
#Software Requirements
Linux (Ubuntu / WSL recommended)
Latest NVIDIA Drivers
CUDA Toolkit (nvcc 13.2)
Python 3.12+

---

## Project Flow

This project builds the same model step-by-step:

1️⃣ PyTorch (High-Level Abstraction)
Fully abstracted implementation
Uses GPU acceleration automatically
Minimal control over memory or execution
1st code block from this file stores the mnist dataset on the system in "data" file in working directory in binary format.
2️⃣ NumPy (C-Friendly Intermediate)
Manual forward + backward implementation
Still vectorized
Runs on CPU
Mirrors mathematical formulation closely
3️⃣ C (Naive CPU Implementation)
Raw loops for matrix multiplication
No vectorization
No parallelism
Extremely slow → exposes true computational cost
4️⃣ CUDA (Naive Kernel)
Parallelizes matrix multiplication
Each thread computes one output element
Uses global memory only
5️⃣ CUDA (Vectorized / Optimized)
Shared memory tiling
Memory coalescing
Block-level parallelism
Vectorized loads (float4)
Reduced global memory access

---

## Model Architecture
Forward Pass
X → XW₁ + b₁ → ReLU → XW₂ + b₂ → Softmax
Loss (Cross Entropy)
L = -log(p_correct)
Backward Pass
∂L/∂z = (probabilities - one_hot) / batch_size
Gradients
∂L/∂W = Xᵀ @ grad_output
Update Rule
W = W - lr * gradient

---

## Where CUDA Comes In

The bottleneck of neural networks is:

Matrix Multiplication (GEMM)

Shapes involved:

Operation	Shape
Input	(B × 784)
W₁	(784 × 1024)
Hidden	(B × 1024)
W₂	(1024 × 10)

Parallelization Strategy

Instead of:

for i
  for j
    for k

CUDA does:

Each thread → computes one (i, j)
Tiling (Optimized CUDA)
Load chunks of matrices into shared memory
Reuse data across threads
Reduce expensive global memory access

---

## CUDA Implementations
Naive CUDA
1 thread = 1 output element
Direct global memory access
Simple but effective

✔ Pros:

Easy to understand
Good baseline

❌ Cons:

Memory inefficient
No caching
Vectorized CUDA

Includes:

Shared memory tiling
Memory coalescing
2D block tiling
Vectorized loads (float4)
Reduced memory bandwidth usage

✔ Pros:

HPC-level optimization techniques
Demonstrates real GPU architecture usage

❌ Cons:

Higher overhead
Not always faster for small matrices

---

## Results
🔹 PyTorch (GPU)
Total time: 4.8s
Final loss: 0.1435
🔹 NumPy (CPU)
Total time: 17.6s
Final loss: 0.1382
🔹 C (CPU - Reduced Config)
Total time: 49.6s
(3 epochs only due to slowness)
🔹 CUDA (Naive)
Total time: 4.6s
Final loss: 0.1429

✔ Best balance of speed + correctness

🔹 CUDA (Vectorized)
Total time: 5.7s
Final loss: 0.1425

## Key Observations
1️⃣ Abstraction vs Performance
PyTorch ≈ CUDA Naive > NumPy >> C
2️⃣ CPU Bottleneck is Real

C implementation shows:

Forward + Backward ≈ 95% of total time
3️⃣ CUDA Speedup
~10x faster than NumPy
~12x faster than C
4️⃣ Optimization Tradeoff

Vectorized CUDA:

✔ Faster forward pass
❌ Slower backward pass
❌ Higher overhead

👉 For small batch sizes (32):

Naive CUDA > Optimized CUDA
5️⃣ Real HPC Insight
Optimization ≠ Always Faster

Depends on:

Matrix size
Memory reuse
Kernel launch overhead

---

## Charts

Include:

⏱️ Execution time comparisons

<img width="1180" height="684" alt="mnist time comparison output" src="https://github.com/user-attachments/assets/4b8ebe26-0d74-417c-a3f6-f1365b57e6a5" />

📉 Loss convergence plots 

<img width="855" height="557" alt="mnist loss comparison output" src="https://github.com/user-attachments/assets/4346197c-8b14-4e1b-9802-6540cbaeef62" />

---

## Important Notes
C implementation uses smaller config due to runtime constraints:
Hidden size: 256
Batch size: 4
Epochs: 3
All other implementations use:
INPUT_SIZE 784
HIDDEN_SIZE 1024
OUTPUT_SIZE 10
BATCH_SIZE 32
EPOCHS 10

---

## What You Learn From This Project
How neural networks actually compute
How backprop works numerically
How matrix multiplication dominates compute
How GPUs accelerate workloads
Why memory access patterns matter
When optimization helps — and when it doesn’t

---

## Conclusion

This project demonstrates that:
Understanding fundamentals > blindly using frameworks and Efficient systems require both mathematical correctness and hardware awareness

---

## Future Work
cuBLAS integration (compare with industry standard)
Mixed precision (FP16)
Larger batch sizes
Kernel fusion
Multi-GPU scaling

---

# Acknowledgements
MNIST Dataset
CUDA Programming Guide: https://youtu.be/86FAWCzIe_4?si=1A8exVTYSpz9k5ax 
HPC optimization resources: https://siboehm.com/articles/22/CUDA-MMM

# CUDA kernel project

## CUDA programming basics

### 1. The Kernel: A "Parallel" C++ Function

In C++, a function runs once. In CUDA, a Kernel is a function that runs *N* times in parallel on the GPU. To tell the compiler a funciton belongs on the GPU, we use the `__global__` specifier. It must always return `void`.

```c++
// This runs on the GPU, called from the CPU
__global__ void my_kernel(float* data) {
    // logic goes here
}
```

### 2. Grids, Blocks, and Threads

To handle thousands of tasks, CUDA organises threads into a hierarchy. Mental image: a large warehouse (the **Grid**), filled with teams (the **Blocks**), where each team has individual workers (the **Threads**).

Inside the kernel, every thread is running the exact same code. To make them do different work (like processing different parts of an array), we use built-in variables to find their "ID".

- `threadIdx.x`: The ID of the thread within its block.
- `blockIdx.x`: The ID of the block within the grid.
- `blockDim.x`: The number of threads in one block.

Important formula to keep in mind, this is how we can calculate a thread's unique global index $i$ in a 1D array:
$$
i = \text {blockIdx.x} \times \text {blockDim.x} + \text {threadIdx.x}
$$

### 3. Memory management

The GPU cannot see the CPU's RAM. We have to manually move data back and forth, similar to how we use `malloc` and `free` in C.

| C++ (CPU) | CUDA (GPU) | Purpose |
| --- | --- | --- |
| `malloc()` | `cudaMalloc()` | Allocate memory. |
| `free()` | `cudaFree()` | Free memory. |
| `memcpy()` | `cudaMemcpy()` | Move data between CPU + GPU. |

### 3. Launching the Kernel

To run the kernel, we use the trpile-angle bracket syntax: `<<<blocks, threads>>>`.

- **Threads per block:** A common choice is **256** or **512**.
- **Number of blocks:** We need enough blocks to cover our total size `s`.

**The formula:**
$$
\text {blocks} = \frac {s + \text {threadsPerBlock} - 1} {\text {threadsPerBlock}}
$$


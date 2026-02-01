# CUDA kernel project

## CUDA programming basics

In C++, a function runs once. In CUDA, a Kernel is a function that runs *N* times in parallel on the GPU. To tell the compiler a funciton belongs on the GPU, we use the `__global__` specifier. It must always return `void`.

```c++
// This runs on the GPU, called from the CPU
__global__ void my_kernel(float* data) {
    // logic goes here
}
```

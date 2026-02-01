#include <iostream>


__global__ void my_first_kernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Launch kernel with 1 block of 10 threads
    my_first_kernel<<<1, 10>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    return 0;
}
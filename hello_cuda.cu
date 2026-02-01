#include <iostream>
#include <cstdlib>


__global__ void my_first_kernel(int* A, int* B, int* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // The "Guard": Only do work if we are inside the array bounds
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // allocate memory on the host and fill with data
    size_t s = 100000000;
    int* a = (int*) malloc(sizeof(int)*s);
    int* b = (int*) malloc(sizeof(int)*s);
    int* c = (int*) malloc(sizeof(int)*s);

    for (int i = 0; i < s; i++){
        a[i] = i; b[i] = i;
    }

    // allocate memory on the device 
    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, sizeof(int)*s);
    cudaMalloc((void**)&d_b, sizeof(int)*s);
    cudaMalloc((void**)&d_c, sizeof(int)*s);

    // copy the data from host to device
    // cudaMemcpy(destination, source, size, direction);
    cudaMemcpy(d_a, a, sizeof(int)*s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int)*s, cudaMemcpyHostToDevice);

    // launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (s + threadsPerBlock - 1) / threadsPerBlock;

    my_first_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, sizeof(int)*s, cudaMemcpyDeviceToHost);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    for (int i = 0; i < 3; ++i){
        std::cout << c[i] << std::endl;
    }

    // Free all memory allocations
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
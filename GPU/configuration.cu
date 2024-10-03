#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32        
#define MAX_THREADS 1024   
#define SHARED_MEM_SIZE 128 * 1024  
#define CONSTANT_MEM_SIZE 64 * 1024 


__constant__ int constant_NNZ_data[];
__constant__ int constant_rowind[];
__constant__ float constant_sparsity_list[];


__global__ void ApSpGEMMKernel(int* d_A, int* d_B, int* d_C, int numRows, int numCols, int numNNZ) {

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < numRows) {

    }
}


int main() {

    int *d_A, *d_B, *d_C;
    int numRows;
    int numCols ;
    int numNNZ ;


    cudaMalloc((void**)&d_A, numNNZ * sizeof(int));
    cudaMalloc((void**)&d_B, numCols * sizeof(int));
    cudaMalloc((void**)&d_C, numRows * sizeof(int));


    cudaMemcpyToSymbol(constant_NNZ_data, /* data */, sizeof(int) * numNNZ);
    cudaMemcpyToSymbol(constant_rowind, /* data */, sizeof(int) * numRows);
    cudaMemcpyToSymbol(constant_sparsity_list, /* data */, sizeof(float) * numRows);


    int numBlocks = (numRows + MAX_THREADS - 1) / MAX_THREADS;
    ApSpGEMMKernel<<<numBlocks, MAX_THREADS>>>(d_A, d_B, d_C, numRows, numCols, numNNZ);
    

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // cudaMemcpy(C, d_C, numRows * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

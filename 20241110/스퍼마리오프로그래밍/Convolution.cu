#include <cuda_runtime.h>
#include <iostream>
#include <algorithm> // for std::max and std::min

__global__
void cudaConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    int convLen = aLen + bLen - 1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int n = index; n < convLen; n += step)
    {
        float prod = 0;
        int kMax = min(aLen, n + 1);
        for (int k = 0; k < kMax; ++k)
        {
            if (n - k < bLen)
            {
                prod += a[k] * b[n - k];
            }
        }
        res[n] = prod;
    }
}

void myConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    int convLen = aLen + bLen - 1;
    int blockSize = 256;
    int numBlocks = (convLen + blockSize - 1) / blockSize;
    cudaConvolve<<<numBlocks, blockSize>>>(a, b, res, aLen, bLen);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

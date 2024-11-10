#include <iostream>
#include <cuda_runtime.h>

// Declare the myConvolve function
void myConvolve(float *a, float *b, float *res, int aLen, int bLen);

int main()
{
    const int aLen = 5;
    const int bLen = 3;
    float h_a[aLen] = {1, 2, 3, 4, 5};
    float h_b[bLen] = {1, 2, 3};
    const int resLen = aLen + bLen - 1;
    float h_res[resLen];

    // Allocate device memory
    float *d_a, *d_b, *d_res;
    cudaMalloc(&d_a, aLen * sizeof(float));
    cudaMalloc(&d_b, bLen * sizeof(float));
    cudaMalloc(&d_res, resLen * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, h_a, aLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bLen * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution
    myConvolve(d_a, d_b, d_res, aLen, bLen);

    // Copy result back to host
    cudaMemcpy(h_res, d_res, resLen * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result: ";
    for (int i = 0; i < resLen; ++i)
    {
        std::cout << h_res[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return 0;
}

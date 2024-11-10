#include <cuda_runtime.h>
#include <iostream>

// Matrix structure definition
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Function to get a sub-matrix
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Function to get an element from a matrix
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Function to set an element in a matrix
__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

// Matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}

// Matrix multiplication - Host code
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A;
    d_A.width = d_A.stride = A.width; 
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width; 
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = d_C.stride = C.width; 
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

void InitializeMatrix(Matrix &mat, int width, int height, const float* elements) {
    mat.width = width;
    mat.height = height;
    mat.stride = width;
    mat.elements = (float*)malloc(width * height * sizeof(float));
    memcpy(mat.elements, elements, width * height * sizeof(float));
}

void PrintMatrix(const Matrix &mat) {
    for (int i = 0; i < mat.height; ++i) {
        for (int j = 0; j < mat.width; ++j) {
            std::cout << mat.elements[i * mat.stride + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    Matrix A, B, C;

    // Example matrices (3x3 for simplicity)
    float elementsA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float elementsB[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    float elementsC[9] = {0};

    // Initialize matrices A, B, and C
    InitializeMatrix(A, 3, 3, elementsA);
    InitializeMatrix(B, 3, 3, elementsB);
    InitializeMatrix(C, 3, 3, elementsC);

    // Matrix multiplication
    MatMul(A, B, C);

    // Print result
    std::cout << "Matrix A:" << std::endl;
    PrintMatrix(A);
    std::cout << "Matrix B:" << std::endl;
    PrintMatrix(B);
    std::cout << "Matrix C (Result):" << std::endl;
    PrintMatrix(C);

    // Free host memory
    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "cuda_utils.cuh"
#include "../kernels/sgemm_cublas.cuh"
#include <cublas_v2.h>

#define N 8192
int main()
{
    // Seed RNG
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    // Allocate host memory
    float *A_h = static_cast<float *>(std::malloc(bytes));
    float *B_h = static_cast<float *>(std::malloc(bytes));
    float *C_h = static_cast<float *>(std::malloc(bytes));

    if (!A_h || !B_h || !C_h)
    {
        std::cerr << "Host malloc failed!\n";
        std::free(A_h);
        std::free(B_h);
        std::free(C_h);
        return EXIT_FAILURE;
    }

    // Initialize matrices
    for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i)
    {
        A_h[i] = static_cast<float>(std::rand()) / RAND_MAX;
        B_h[i] = static_cast<float>(std::rand()) / RAND_MAX;
        C_h[i] = 0.0f;
    }

    // Device pointers
    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;

    CHECK_CUDA(cudaMalloc(&A_d, bytes));
    CHECK_CUDA(cudaMalloc(&B_d, bytes));
    CHECK_CUDA(cudaMalloc(&C_d, bytes));

    CHECK_CUDA(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(C_d, C_h, bytes, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Create cuBLAS handle (outside timed region)
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cublasCreate failed: " << stat << "\n";
        return EXIT_FAILURE;
    }

    // Warm-up
    run_sgemm(handle, A_d, B_d, C_d, N, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Create timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Time multiple iterations
    CHECK_CUDA(cudaEventRecord(start));

    run_sgemm(handle, A_d, B_d, C_d, N, alpha, beta);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));

    float ms = ms_total; // avg milliseconds per GEMM

    // Compute GFLOP/s
    double flops = 2.0 * double(N) * N * N;
    double gflops = flops / (ms * 1e6);

    std::cout << "Performance:  " << gflops << " GFLOP/s\n";

    // Destroy resources
    cublasDestroy(handle);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));

    std::free(A_h);
    std::free(B_h);
    std::free(C_h);

    return 0;
}

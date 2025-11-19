#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "cuda_utils.cuh"
#include "../kernels/sgemm.cuh"

#define N 8192

int main()
{
    // seeding random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // bytes needed 4096x4096x4
    size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

    float *A_h = nullptr;
    float *B_h = nullptr;
    float *C_h = nullptr;

    A_h = static_cast<float *>(std::malloc(bytes));
    B_h = static_cast<float *>(std::malloc(bytes));
    C_h = static_cast<float *>(std::malloc(bytes));

    if (!A_h || !B_h || !C_h)
    {
        std::cerr << "failed to allocate host memory!" << std::endl;
        std::free(A_h);
        std::free(B_h);
        std::free(C_h);

        return EXIT_FAILURE;
    }

    // initializing matrices
    for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i)
    {
        A_h[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        B_h[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        C_h[i] = 0.0f;
    }

    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;

    CHECK_CUDA(cudaMalloc(&A_d, bytes));
    CHECK_CUDA(cudaMalloc(&B_d, bytes));
    CHECK_CUDA(cudaMalloc(&C_d, bytes));

    CHECK_CUDA(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(C_d, C_h, bytes, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // FLOP count for N x N SGEMM
    double flops = 2.0 * static_cast<double>(N) * N * N;

    CHECK_CUDA(cudaEventRecord(start));
    run_sgemm(A_d, B_d, C_d, N, alpha, beta);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double gflops = flops / (ms * 1e6);

    std::cout << ms << " ms, "
              << gflops << " GFLOP/s\n";

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

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

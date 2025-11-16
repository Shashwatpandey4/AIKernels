#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "cuda_utils.cuh"
#include "../kernels/sgemm_naive.cuh"

#define N 4096

int main()
{
    // seeding random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // bytes needed 4096x4096x4
    size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

    float *A_h = nullptr;
    float *B_h = nullptr;
    float *C_h = nullptr;
    float *C_h_ref = nullptr; // cublas result for reference

    A_h = static_cast<float *>(std::malloc(bytes));
    B_h = static_cast<float *>(std::malloc(bytes));
    C_h = static_cast<float *>(std::malloc(bytes));
    C_h_ref = static_cast<float *>(std::malloc(bytes));

    if (!A_h || !B_h || !C_h || !C_h_ref)
    {
        std::cerr << "failed to allocate host memmory !" << std::endl;
        std::free(A_h);
        std::free(B_h);
        std::free(C_h);
        std::free(C_h_ref);
        return EXIT_FAILURE;
    }

    // initialinzing matrices
    for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i)
    {
        A_h[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        B_h[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        C_h[i] = 0.0f;
        C_h_ref[i] = 0.0f;
    }

    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;
    float *C_d_ref = nullptr;

    CHECK_CUDA(cudaMalloc(&A_d, bytes));
    CHECK_CUDA(cudaMalloc(&B_d, bytes));
    CHECK_CUDA(cudaMalloc(&C_d, bytes));
    CHECK_CUDA(cudaMalloc(&C_d_ref, bytes));

    CHECK_CUDA(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(C_d, C_h, bytes, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    // launching the kernel
    // time it
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    run_sgemm_naive(A_d, B_d, C_d, N, alpha, beta);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop)); // wait till the kernel is done

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double ops = 2.0 * static_cast<double>(N) * N * N;
    double secs = ms * 1e-3;
    double gflops = (ops / secs) * 1e-9;

    std::cout << "kernel time: " << ms << " ms\n";
    std::cout << "kernel perf: " << gflops << " GFLOPS\n";

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // back to host
    CHECK_CUDA(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    // now run cublas gemm to check results
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUBLAS(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    B_d, N,
                    A_d, N,
                    &beta,
                    C_d_ref, N));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_cublas = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cublas, start, stop));

    double ops_cublas = 2.0 * static_cast<double>(N) * N * N;
    double secs_cublas = ms_cublas * 1e-3;
    double gflops_cublas = (ops_cublas / secs_cublas) * 1e-9;

    std::cout << "cuBLAS time:  " << ms_cublas << " ms\n";
    std::cout << "cuBLAS perf:  " << gflops_cublas << " GFLOPS\n";
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(C_h_ref, C_d_ref, bytes, cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i)
    {
        double diff = std::abs(static_cast<double>(C_h[i]) - static_cast<double>(C_h_ref[i]));
        if (diff > max_diff)
            max_diff = diff;
    }

    std::cout << "Max difference vs cuBLAS: " << max_diff << std::endl;

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
    CHECK_CUDA(cudaFree(C_d_ref));

    std::free(A_h);
    std::free(B_h);
    std::free(C_h);
    std::free(C_h_ref);

    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                  \
    do                                                                    \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
#define CHECK_CUBLAS(call)                                \
    do                                                    \
    {                                                     \
        cublasStatus_t stat = call;                       \
        if (stat != CUBLAS_STATUS_SUCCESS)                \
        {                                                 \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", \
                    stat, __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

int main()
{
    const int M = 8192;
    const int N = 8192;
    const int K = 8192;

    const int iters = 20;

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    // host alloc + random init
    float *h_A = (float *)malloc(sizeA * sizeof(float));
    float *h_B = (float *)malloc(sizeB * sizeof(float));
    float *h_C = (float *)malloc(sizeC * sizeof(float));

    srand(123);
    for (size_t i = 0; i < sizeA; ++i)
        h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < sizeB; ++i)
        h_B[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < sizeC; ++i)
        h_C[i] = 0.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, sizeC * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // warmup
    for (int i = 0; i < 5; ++i)
    {
        /*
            CHECK_CUBLAS(cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_A, M,
                d_B, K,
                &beta,
                d_C, M));*/
        CHECK_CUBLAS(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_A, CUDA_R_32F, M,
                d_B, CUDA_R_32F, K,
                &beta,
                d_C, CUDA_R_32F, M,
                CUDA_R_32F, // compute type
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // timed loop
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters; // per GEMM

    double flops = 2.0 * (double)M * N * K;
    double t_sec = ms / 1e3;
    double gflops = flops / t_sec / 1e9;

    printf("M=N=K=%d, time=%.3f ms, perf=%.2f GFLOP/s\n", M, ms, gflops);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

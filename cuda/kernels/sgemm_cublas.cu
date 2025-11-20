#include <cublas_v2.h>
#include <cstdio>

// Row-major GEMM wrapper for:  C = alpha*A*B + beta*C
// A, B, C are row-major N×N
void run_sgemm(const float *A_d,
               const float *B_d,
               float *C_d,
               int N,
               float alpha,
               float beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS assumes column-major storage.
    //
    // Row-major A*B  is equivalent to:
    //   column-major B * A  (swap A <-> B)
    //
    // So call:
    //   C = alpha * B*A + beta*C    (column-major)
    cublasStatus_t stat =
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, // m
                    N, // n
                    N, // k
                    &alpha,
                    B_d, N, // B becomes "left" matrix
                    A_d, N, // A becomes "right" matrix
                    &beta,
                    C_d, N);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS SGEMM failed: %d\n", stat);
    }

    cublasDestroy(handle);
}

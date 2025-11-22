#include <cublas_v2.h>
#include <cstdio>

void run_sgemm(cublasHandle_t handle,
               const float *A_d,
               const float *B_d,
               float *C_d,
               int N,
               float alpha,
               float beta)
{
    cublasStatus_t stat =
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    B_d, N,
                    A_d, N,
                    &beta,
                    C_d, N);

    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("cuBLAS SGEMM failed: %d\n", stat);
}

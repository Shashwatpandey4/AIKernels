
#pragma once

#include <cublas_v2.h>

void run_sgemm(cublasHandle_t handle,
               const float *A_d,
               const float *B_d,
               float *C_d,
               int N,
               float alpha,
               float beta);

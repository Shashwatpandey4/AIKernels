#pragma once
#include <cuda_runtime.h>

void run_sgemm(const float *A_d, const float *B_d, float *C_d, int N, float alpha, float beta);
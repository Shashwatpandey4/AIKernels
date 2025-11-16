#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

inline void __check_cuda_error(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " (" << err << ") at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) __check_cuda_error((call), __FILE__, __LINE__)

inline void __check_cublas(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error (" << status << ") at "
                  << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUBLAS(call) __check_cublas((call), __FILE__, __LINE__)

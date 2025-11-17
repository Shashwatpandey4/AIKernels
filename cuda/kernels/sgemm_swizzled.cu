#include "sgemm_swizzled.cuh"

#define TILE 32 // 32x32 block -> 1024 threads

__global__ void sgemm_smem_swizzled_kernel(const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float *__restrict__ C,
                                           int N,
                                           float alpha, float beta)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int t = 0; t < N; t += TILE)
    {
        // Load A tile: no swizzle
        if (row < N && (t + tx) < N)
            As[ty][tx] = A[row * N + (t + tx)];
        else
            As[ty][tx] = 0.0f;

        // Load B tile: swizzled in the shared-memory column dimension
        if ((t + ty) < N && col < N)
        {
            int col_s = (tx + ty) & (TILE - 1); // swizzled column index
            Bs[ty][col_s] = B[(t + ty) * N + col];
        }
        else
        {
            int col_s = (tx + ty) & (TILE - 1);
            Bs[ty][col_s] = 0.0f;
        }

        __syncthreads();

// Compute: correct SGEMM math with inverse access pattern
#pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            float a = As[ty][k]; // A[row, t + k]

            int col_s = (tx + k) & (TILE - 1);
            float b = Bs[k][col_s]; // B[t + k, col] recovered via swizzle

            acc = fmaf(a, b, acc);
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        int idx = row * N + col;
        C[idx] = alpha * acc + beta * C[idx];
    }
}

void run_sgemm_smem_swizzled(const float *A_d, const float *B_d,
                             float *C_d, int N, float alpha, float beta)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    sgemm_smem_swizzled_kernel<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}

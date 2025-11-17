#include "sgemm_smem.cuh"

#define TILE 32 // 32x32 block -> 1024 threads, fits fine on your RTX 4060

// ------------------
// 1) bank-conflicted
// ------------------
__global__ void sgemm_smem_conflicted_kernel(const float *__restrict__ A,
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

    // loop over tiles of K dimension
    for (int t = 0; t < N; t += TILE)
    {
        // load a TILE x TILE tile from A and B into shared memory
        if (row < N && (t + tx) < N)
            As[ty][tx] = A[row * N + (t + tx)];
        else
            As[ty][tx] = 0.0f;

        if ((t + ty) < N && col < N)
            Bs[ty][tx] = B[(t + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // *** Deliberately bank-conflicted access pattern ***
        // We make each warp load down a column: stride = TILE = 32,
        // so addresses differ by 32 * 4 bytes = 128B -> same bank.
        //
        // Warp layout: blockDim = (32, 32) so each warp is one row (ty fixed, tx 0..31)
        // Here we use lane id to choose the *row* we read, so:
        //   addr(lane) ~ base + lane*32 -> all lanes hit same bank.
        int lane = tx; // lane id in warp

#pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            int row_idx = lane;       // 0..31 row index -> stride-32 across lanes
            float a = As[row_idx][k]; // bank-conflicted column load
            float b = Bs[k][ty];      // normal-ish access (ty fixed across warp)
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

void run_sgemm_smem_conflicted(const float *A_d, const float *B_d,
                               float *C_d, int N, float alpha, float beta)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    sgemm_smem_conflicted_kernel<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}

/*
smem tiled
*/

#define TILE_SIZE 16

__global__ void sgemm(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t)
    {
        int tiledColA = t * TILE_SIZE + threadIdx.x;
        int tiledRowB = t * TILE_SIZE + threadIdx.y;

        if (row < N && tiledColA < N)
        {
            As[threadIdx.y][threadIdx.x] = A[row * N + tiledColA];
        }
        else
        {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiledRowB < N && col < N)
        {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        }
        else
        {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        int idx = row * N + col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}

void run_sgemm(const float *A_d, const float *B_d, float *C_d, int N, float alpha, float beta)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    sgemm<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}
// 2x2 Register tiling

#define TILE_SIZE 16

__global__ void sgemm(const float *__restrict__ A,
                      const float *__restrict__ B,
                      float *__restrict__ C,
                      int N,
                      float alpha,
                      float beta)
{
    // 8x8 threads per block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output tile origin
    int tileRow = blockIdx.y * TILE_SIZE;
    int tileCol = blockIdx.x * TILE_SIZE;

    // each thread computes a 2×2 block:
    int row0 = tileRow + ty * 2;
    int col0 = tileCol + tx * 2;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float c00 = 0.f, c01 = 0.f;
    float c10 = 0.f, c11 = 0.f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        int globalK = t * TILE_SIZE;

        int loadRow = tileRow + ty * 2 + (tx / 8);
        int loadCol = globalK + (tx % 8) * 2 + (ty / 8);

        int sRow = ty * 2 + (tx / 8);
        int sCol = (tx % 8) * 2 + (ty / 8);

        // load A
        if (loadRow < N && loadCol < N)
            As[sRow][sCol] = A[loadRow * N + loadCol];
        else
            As[sRow][sCol] = 0.f;

        // load B
        loadRow = globalK + ty * 2 + (tx / 8);
        loadCol = tileCol + (tx % 8) * 2 + (ty / 8);

        if (loadRow < N && loadCol < N)
            Bs[sRow][sCol] = B[loadRow * N + loadCol];
        else
            Bs[sRow][sCol] = 0.f;

        __syncthreads();

        // Compute 2×2 tile
        for (int k = 0; k < TILE_SIZE; k++)
        {
            float a0 = As[ty * 2 + 0][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2 + 0];
            float b1 = Bs[k][tx * 2 + 1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // Store output (with bounds checks)
    if (row0 < N)
    {
        if (col0 < N)
            C[row0 * N + col0] = alpha * c00 + beta * C[row0 * N + col0];
        if (col0 + 1 < N)
            C[row0 * N + col0 + 1] = alpha * c01 + beta * C[row0 * N + col0 + 1];
    }

    if (row0 + 1 < N)
    {
        if (col0 < N)
            C[(row0 + 1) * N + col0] = alpha * c10 + beta * C[(row0 + 1) * N + col0];
        if (col0 + 1 < N)
            C[(row0 + 1) * N + col0 + 1] = alpha * c11 + beta * C[(row0 + 1) * N + col0 + 1];
    }
}

void run_sgemm(const float *A_d, const float *B_d, float *C_d, int N, float alpha, float beta)
{
    dim3 block(TILE_SIZE / 2, TILE_SIZE / 2);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    sgemm<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}
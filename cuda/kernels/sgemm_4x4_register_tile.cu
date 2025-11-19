/*
  smem tiled + 4x4 register tiling, TILE_SIZE = 32
*/

#define TILE_SIZE 32

__global__ void sgemm(const float *__restrict__ A,
                      const float *__restrict__ B,
                      float *__restrict__ C,
                      int N,
                      float alpha,
                      float beta)
{
    // 8x8 threads per block
    int tx = threadIdx.x; // 0..7
    int ty = threadIdx.y; // 0..7

    // Each block computes a 32x32 tile of C
    int tileRow = blockIdx.y * TILE_SIZE;
    int tileCol = blockIdx.x * TILE_SIZE;

    // This thread's 4x4 micro-tile inside that 32x32 tile
    int row0 = tileRow + ty * 4; // base row of this thread's 4x4 patch
    int col0 = tileCol + tx * 4; // base col of this thread's 4x4 patch

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 4x4 accumulators in registers
    float c[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            c[i][j] = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int linearTid = ty * blockDim.x + tx;  // 0..63
    int loaders = blockDim.x * blockDim.y; // 64

    for (int t = 0; t < numTiles; ++t)
    {
        int kBase = t * TILE_SIZE;

        // ----------------------------------------------------
        // Load As and Bs tiles cooperatively into shared memory
        // Each block loads 32x32 = 1024 elements; 64 threads
        // → 16 elements per thread (strided loop)
        // ----------------------------------------------------
        for (int idx = linearTid; idx < TILE_SIZE * TILE_SIZE; idx += loaders)
        {
            int r = idx / TILE_SIZE;    // 0..31
            int ccol = idx % TILE_SIZE; // 0..31

            int aRow = tileRow + r;
            int aCol = kBase + ccol;

            if (aRow < N && aCol < N)
                As[r][ccol] = A[aRow * N + aCol];
            else
                As[r][ccol] = 0.0f;

            int bRow = kBase + r;
            int bCol = tileCol + ccol;

            if (bRow < N && bCol < N)
                Bs[r][ccol] = B[bRow * N + bCol];
            else
                Bs[r][ccol] = 0.0f;
        }

        __syncthreads();

// Compute this thread's 4x4 micro-tile from this K-tile
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            // Load 4 A values (4 rows for this thread)
            float a_vec[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int sRow = ty * 4 + i; // 0..31 within tile
                a_vec[i] = As[sRow][k];
            }

            // Load 4 B values (4 cols for this thread)
            float b_vec[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                int sCol = tx * 4 + j; // 0..31 within tile
                b_vec[j] = Bs[k][sCol];
            }

// 4x4 outer product → accumulate into c[i][j]
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    c[i][j] += a_vec[i] * b_vec[j];
        }

        __syncthreads();
    }

// Write results to C (with bounds checks)
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int r = row0 + i;
        if (r >= N)
            continue;

#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int ccol = col0 + j;
            if (ccol >= N)
                continue;

            int idx = r * N + ccol;
            C[idx] = alpha * c[i][j] + beta * C[idx];
        }
    }
}

void run_sgemm(const float *A_d,
               const float *B_d,
               float *C_d,
               int N,
               float alpha,
               float beta)
{
    dim3 block(TILE_SIZE / 4, TILE_SIZE / 4); // 8 x 8 threads
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    sgemm<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}

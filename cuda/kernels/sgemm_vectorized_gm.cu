// vectorized ld/st
#define TILE_SIZE 32

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

    // Each block computes a 32x32 tile of C
    int tileRow = blockIdx.y * TILE_SIZE;
    int tileCol = blockIdx.x * TILE_SIZE;

    // This thread's 4x4 micro-tile inside that 32x32 tile
    int row0 = tileRow + ty * 4; // base row of this thread's 4x4 patch
    int col0 = tileCol + tx * 4; // base col of this thread's 4x4 patch

    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // 4x4 accumulators in registers
    float c[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            c[i][j] = 0.0f;

    int numTiles = N / TILE_SIZE;          // assumes N % TILE_SIZE == 0
    int linearTid = ty * blockDim.x + tx;  // 0..63
    int loaders = blockDim.x * blockDim.y; // 64

    // reinterpret bases once; we index in units of float4
    const float4 *A4 = reinterpret_cast<const float4 *>(A);
    const float4 *B4 = reinterpret_cast<const float4 *>(B);

    // Each 32x32 tile = 1024 floats = 256 float4s
    const int totalVecs = (TILE_SIZE * TILE_SIZE) / 4; // 256

    for (int t = 0; t < numTiles; ++t)
    {
        int kBase = t * TILE_SIZE;

        // Vectorized load: GMEM -> SMEM
        // No bounds checks; assumes all tiles are "interior".

        for (int vidx = linearTid; vidx < totalVecs; vidx += loaders)
        {
            int elemIdx = vidx * 4;       // scalar index in 0..1020
            int r = elemIdx / TILE_SIZE;  // row 0..31
            int c4 = elemIdx % TILE_SIZE; // col {0,4,8,...,28}

            // Global scalar indices
            int aIndex = (tileRow + r) * N + (kBase + c4);
            int bIndex = (kBase + r) * N + (tileCol + c4);

            // Because:
            // - N % 4 == 0
            // - tileRow, tileCol, kBase multiples of 32
            // => aIndex and bIndex are multiples of 4
            // => aIndex/4, bIndex/4 are valid indices into float4*
            float4 a4 = A4[aIndex / 4];
            float4 b4 = B4[bIndex / 4];

            // Scatter into shared memory as scalars
            As[r][c4 + 0] = a4.x;
            As[r][c4 + 1] = a4.y;
            As[r][c4 + 2] = a4.z;
            As[r][c4 + 3] = a4.w;

            Bs[r][c4 + 0] = b4.x;
            Bs[r][c4 + 1] = b4.y;
            Bs[r][c4 + 2] = b4.z;
            Bs[r][c4 + 3] = b4.w;
        }

        __syncthreads();

        // Compute this thread's 4x4 micro-tile for this K-tile
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            float a_vec[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int sRow = ty * 4 + i;
                a_vec[i] = As[sRow][k];
            }

            float b_vec[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                int sCol = tx * 4 + j;
                b_vec[j] = Bs[k][sCol];
            }

#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    c[i][j] += a_vec[i] * b_vec[j];
        }

        __syncthreads();
    }

    // --------------------------------------------------------
    // Write results to C
    // Here we *could* skip bounds checks for nice N, but they’re
    // cheap compared to the inner loop, and this path runs once.
    // --------------------------------------------------------
#pragma unroll
    // Write results to C (vectorized when possible)
    for (int i = 0; i < 4; ++i)
    {
        int r = row0 + i;
        if (r >= N)
            continue;

        int baseIdx = r * N + col0; // index of (r, col0)

        // Try vectorized path if the whole 4-wide row is in-bounds AND aligned
        if ((col0 + 3) < N && ((baseIdx & 3) == 0)) // baseIdx % 4 == 0 → 16B aligned
        {
            float4 acc; // accumulators for this row
            acc.x = c[i][0];
            acc.y = c[i][1];
            acc.z = c[i][2];
            acc.w = c[i][3];

            // load 4 existing C values
            float4 old = *reinterpret_cast<const float4 *>(&C[baseIdx]);

            // alpha * acc + beta * old
            float4 out;
            out.x = alpha * acc.x + beta * old.x;
            out.y = alpha * acc.y + beta * old.y;
            out.z = alpha * acc.z + beta * old.z;
            out.w = alpha * acc.w + beta * old.w;

            // store back
            *reinterpret_cast<float4 *>(&C[baseIdx]) = out;
        }
        else
        {
            // Fallback: scalar stores with bounds checks
            for (int j = 0; j < 4; ++j)
            {
                int ccol = col0 + j;
                if (ccol >= N)
                    continue;

                int idx = baseIdx + j;
                float old = C[idx];
                C[idx] = alpha * c[i][j] + beta * old;
            }
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
    // assumes N is a multiple of TILE_SIZE and 4
    dim3 block(TILE_SIZE / 4, TILE_SIZE / 4); // 8 x 8 threads
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);

    sgemm<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}

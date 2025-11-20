// cp.async + double-buffered SGEMM

#define TILE_SIZE 32
#define PAD 4 // 32 + 4 = 36 → row stride 36*4 = 144B = 16 * 9 (nice for cp.async dest)

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

    // This thread's 4x4 micro-tile
    int row0 = tileRow + ty * 4;
    int col0 = tileCol + tx * 4;

    // Double-buffered, padded shared memory
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + PAD];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + PAD];

    // 4x4 accumulators in registers
    float c[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            c[i][j] = 0.0f;

    int numTiles = N / TILE_SIZE;                      // assumes N % TILE_SIZE == 0
    int tid = ty * blockDim.x + tx;                    // 0..63
    int loaders = blockDim.x * blockDim.y;             // 64
    const int totalVecs = (TILE_SIZE * TILE_SIZE) / 4; // 256 float4-equivalent chunks

    // Helper: issue one 16B cp.async from global to shared
    auto cp_async_16 = [](void *dst_smem, const void *src_gmem)
    {
        unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(dst_smem));
        unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(src_gmem);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_addr));
    };

    int buf = 0;

    // -----------------------------
    // Preload tile 0 into buffer 0
    // -----------------------------
    if (numTiles > 0)
    {
        int kBase0 = 0;

        for (int vidx = tid; vidx < totalVecs; vidx += loaders)
        {
            int elemIdx = vidx * 4;       // scalar index in 0..1020
            int r = elemIdx / TILE_SIZE;  // 0..31
            int c4 = elemIdx % TILE_SIZE; // 0,4,...,28

            // Global scalar indices for A and B
            int aIndex = (tileRow + r) * N + (kBase0 + c4);
            int bIndex = (kBase0 + r) * N + (tileCol + c4);

            const float *aPtr = &A[aIndex];
            const float *bPtr = &B[bIndex];

            // 16B async copies into padded shared tiles
            cp_async_16(&As[buf][r][c4], aPtr);
            cp_async_16(&Bs[buf][r][c4], bPtr);
        }

        // Finish cp.async group 0 for tile 0
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    // -----------------------------
    // Main K-loop with double buffering
    // -----------------------------
    for (int t = 0; t < numTiles; ++t)
    {
        int next = buf ^ 1;

        // Prefetch next tile into the other buffer
        if (t + 1 < numTiles)
        {
            int kBaseNext = (t + 1) * TILE_SIZE;

            for (int vidx = tid; vidx < totalVecs; vidx += loaders)
            {
                int elemIdx = vidx * 4; // scalar index in 0..1020
                int r = elemIdx / TILE_SIZE;
                int c4 = elemIdx % TILE_SIZE;

                int aIndex = (tileRow + r) * N + (kBaseNext + c4);
                int bIndex = (kBaseNext + r) * N + (tileCol + c4);

                const float *aPtr = &A[aIndex];
                const float *bPtr = &B[bIndex];

                cp_async_16(&As[next][r][c4], aPtr);
                cp_async_16(&Bs[next][r][c4], bPtr);
            }

            asm volatile("cp.async.commit_group;\n");
        }

        // -----------------------------
        // Compute on current tile buffer
        // -----------------------------
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            float a_vec[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int sRow = ty * 4 + i;
                a_vec[i] = As[buf][sRow][k];
            }

            float b_vec[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                int sCol = tx * 4 + j;
                b_vec[j] = Bs[buf][k][sCol];
            }

#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    c[i][j] += a_vec[i] * b_vec[j];
        }

        if (t + 1 < numTiles)
        {
            // Wait for next tile to be ready, then sync before reading As[next]/Bs[next]
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            buf = next;
        }
    }

    // -----------------------------
    // Write results to C (same as before)
    // -----------------------------
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int r = row0 + i;
        if (r >= N)
            continue;

        int baseIdx = r * N + col0; // index of (r, col0)

        // Vectorized path if the whole 4-wide row is in-bounds AND aligned
        if ((col0 + 3) < N && ((baseIdx & 3) == 0)) // baseIdx % 4 == 0 → 16B aligned
        {
            float4 acc;
            acc.x = c[i][0];
            acc.y = c[i][1];
            acc.z = c[i][2];
            acc.w = c[i][3];

            float4 old = *reinterpret_cast<const float4 *>(&C[baseIdx]);

            float4 out;
            out.x = alpha * acc.x + beta * old.x;
            out.y = alpha * acc.y + beta * old.y;
            out.z = alpha * acc.z + beta * old.z;
            out.w = alpha * acc.w + beta * old.w;

            *reinterpret_cast<float4 *>(&C[baseIdx]) = out;
        }
        else
        {
#pragma unroll
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

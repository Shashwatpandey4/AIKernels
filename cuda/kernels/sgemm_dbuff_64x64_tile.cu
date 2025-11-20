// 64x64 tile per block, 16x16 threads, 4x4 micro-tile per thread
// cp.async + double-buffered tiles in shared memory.

#define BLOCK_M 64
#define BLOCK_N 64
#define K_TILE 32
#define PAD 4 // to make row strides 16B-aligned for cp.async dest

__global__ void sgemm(const float *__restrict__ A,
                      const float *__restrict__ B,
                      float *__restrict__ C,
                      int N,
                      float alpha,
                      float beta)
{
    // 16x16 threads per block
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    int tid = ty * blockDim.x + tx;        // 0..255
    int loaders = blockDim.x * blockDim.y; // 256

    // Each block computes a 64x64 tile of C
    int blockRow = blockIdx.y * BLOCK_M; // row offset in C
    int blockCol = blockIdx.x * BLOCK_N; // col offset in C

    // This thread's 4x4 micro-tile inside that 64x64 tile
    int row0 = blockRow + ty * 4; // base row of this thread's 4x4 patch
    int col0 = blockCol + tx * 4; // base col of this thread's 4x4 patch

    // Double-buffered, padded shared memory tiles:
    // A: [BLOCK_M x K_TILE]
    // B: [K_TILE  x BLOCK_N]
    __shared__ float As[2][BLOCK_M][K_TILE + PAD];
    __shared__ float Bs[2][K_TILE][BLOCK_N + PAD];

    // 4x4 accumulators in registers
    float c[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            c[i][j] = 0.0f;

    // Number of K-tiles
    int numTiles = N / K_TILE; // assumes N % K_TILE == 0

    // Total 16B chunks in each A and B tile
    const int totalVecsA = (BLOCK_M * K_TILE) / 4; // 64*32/4 = 512
    const int totalVecsB = (K_TILE * BLOCK_N) / 4; // 32*64/4 = 512

    // Helper: one 16B cp.async from global to shared
    auto cp_async_16 = [](void *dst_smem, const void *src_gmem)
    {
        unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(dst_smem));
        unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(src_gmem);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_addr));
    };

    int buf = 0;

    // ---------------------------------
    // Preload K-tile 0 into buffer 0
    // ---------------------------------
    if (numTiles > 0)
    {
        int kBase0 = 0;

        // Load A tile: [BLOCK_M x K_TILE]
        for (int idx = tid; idx < totalVecsA; idx += loaders)
        {
            int elem = idx * 4;     // scalar index 0..(BLOCK_M*K_TILE-1)
            int r = elem / K_TILE;  // 0..63
            int k4 = elem % K_TILE; // 0,4,...,28 (since /4 then *4)

            int aIndex = (blockRow + r) * N + (kBase0 + k4);
            const float *aPtr = &A[aIndex];

            cp_async_16(&As[buf][r][k4], aPtr);
        }

        // Load B tile: [K_TILE x BLOCK_N]
        for (int idx = tid; idx < totalVecsB; idx += loaders)
        {
            int elem = idx * 4;      // scalar index
            int k = elem / BLOCK_N;  // 0..31
            int c4 = elem % BLOCK_N; // 0,4,...,60

            int bIndex = (kBase0 + k) * N + (blockCol + c4);
            const float *bPtr = &B[bIndex];

            cp_async_16(&Bs[buf][k][c4], bPtr);
        }

        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    // ---------------------------------
    // Main K-loop with double buffering
    // ---------------------------------
    for (int t = 0; t < numTiles; ++t)
    {
        int next = buf ^ 1;

        // Prefetch next K-tile into the other buffer
        if (t + 1 < numTiles)
        {
            int kBaseNext = (t + 1) * K_TILE;

            // A: [BLOCK_M x K_TILE] at K = kBaseNext
            for (int idx = tid; idx < totalVecsA; idx += loaders)
            {
                int elem = idx * 4;
                int r = elem / K_TILE;
                int k4 = elem % K_TILE;

                int aIndex = (blockRow + r) * N + (kBaseNext + k4);
                const float *aPtr = &A[aIndex];

                cp_async_16(&As[next][r][k4], aPtr);
            }

            // B: [K_TILE x BLOCK_N] at K = kBaseNext
            for (int idx = tid; idx < totalVecsB; idx += loaders)
            {
                int elem = idx * 4;
                int k = elem / BLOCK_N;
                int c4 = elem % BLOCK_N;

                int bIndex = (kBaseNext + k) * N + (blockCol + c4);
                const float *bPtr = &B[bIndex];

                cp_async_16(&Bs[next][k][c4], bPtr);
            }

            asm volatile("cp.async.commit_group;\n");
        }

        // -----------------------------
        // Compute on current buffer
        // -----------------------------
#pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk)
        {
            float a_vec[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int sRow = ty * 4 + i; // 0..63
                a_vec[i] = As[buf][sRow][kk];
            }

            float b_vec[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                int sCol = tx * 4 + j; // 0..63
                b_vec[j] = Bs[buf][kk][sCol];
            }

#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    c[i][j] += a_vec[i] * b_vec[j];
        }

        if (t + 1 < numTiles)
        {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            buf = next;
        }
    }

    // ---------------------------------
    // Write results to C (vectorized when possible)
    // ---------------------------------
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int r = row0 + i;
        if (r >= N)
            continue;

        int baseIdx = r * N + col0; // index of (r, col0)

        // Vectorized path if 4-wide and 16B aligned (always true for nice N, but keep check)
        if ((col0 + 3) < N && ((baseIdx & 3) == 0))
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
    // assumes N is a multiple of 64 and 4
    dim3 block(16, 16);                  // 16 x 16 threads
    dim3 grid(N / BLOCK_N, N / BLOCK_M); // N / 64 in each dimension

    sgemm<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}

// 64x64 tile per block, 8x8 threads, 8x8 micro-tile per thread
// cp.async + double-buffered tiles in shared memory
// General GEMM: C = alpha * A * B + beta * C

#define BLOCK_M 64
#define BLOCK_N 64
#define K_TILE 32
#define PAD 4 // to keep row strides 16B-friendly for cp.async dest

__global__ void sgemm(const float *__restrict__ A,
                      const float *__restrict__ B,
                      float *__restrict__ C,
                      int N,
                      float alpha,
                      float beta)
{
    // 8x8 threads per block → 64 threads → 2 warps
    int tx = threadIdx.x; // 0..7
    int ty = threadIdx.y; // 0..7

    int tid = ty * blockDim.x + tx;        // 0..63
    int loaders = blockDim.x * blockDim.y; // 64

    // Each block computes one 64x64 tile of C
    int blockRow = blockIdx.y * BLOCK_M; // row offset in C
    int blockCol = blockIdx.x * BLOCK_N; // col offset in C

    // This thread's 8x8 micro-tile inside that 64x64 tile
    int row0 = blockRow + ty * 8; // base row of this thread's 8x8 patch
    int col0 = blockCol + tx * 8; // base col of this thread's 8x8 patch

    // Double-buffered, padded shared memory tiles:
    // A: [BLOCK_M x K_TILE]
    // B: [K_TILE  x BLOCK_N]
    __shared__ float As[2][BLOCK_M][K_TILE + PAD];
    __shared__ float Bs[2][K_TILE][BLOCK_N + PAD];

    // 8x8 accumulators in registers (64 floats)
    float c[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
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

        // Load A tile: [BLOCK_M x K_TILE] = [64 x 32]
        for (int idx = tid; idx < totalVecsA; idx += loaders)
        {
            int elem = idx * 4;     // scalar index 0..(BLOCK_M*K_TILE-1)
            int r = elem / K_TILE;  // 0..63
            int k4 = elem % K_TILE; // 0,4,...,28

            int aIndex = (blockRow + r) * N + (kBase0 + k4);
            const float *aPtr = &A[aIndex];

            cp_async_16(&As[buf][r][k4], aPtr);
        }

        // Load B tile: [K_TILE x BLOCK_N] = [32 x 64]
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
        int kBaseNext = (t + 1) * K_TILE;

        // Prefetch next K-tile into the other buffer
        if (t + 1 < numTiles)
        {
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
        // Each thread computes 8x8 outputs
        // -----------------------------
#pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk)
        {
            float a_vec[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                int sRow = ty * 8 + i; // 0..63
                a_vec[i] = As[buf][sRow][kk];
            }

            float b_vec[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                int sCol = tx * 8 + j; // 0..63
                b_vec[j] = Bs[buf][kk][sCol];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j)
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
    // Write results to C
    // C = alpha * acc + beta * C
    // Vectorized across 8-wide row (two float4s)
    // ---------------------------------
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        int r = row0 + i;
        if (r >= N)
            continue;

        int baseIdx = r * N + col0; // index of (r, col0)

        // For nice N (multiple of 64), this will always be true,
        // but keep the checks for generality.
        if ((col0 + 7) < N && ((baseIdx & 3) == 0)) // baseIdx % 4 == 0 → 16B aligned
        {
            // First 4 columns
            float4 acc0;
            acc0.x = c[i][0];
            acc0.y = c[i][1];
            acc0.z = c[i][2];
            acc0.w = c[i][3];

            float4 old0 = *reinterpret_cast<const float4 *>(&C[baseIdx]);

            float4 out0;
            out0.x = alpha * acc0.x + beta * old0.x;
            out0.y = alpha * acc0.y + beta * old0.y;
            out0.z = alpha * acc0.z + beta * old0.z;
            out0.w = alpha * acc0.w + beta * old0.w;

            *reinterpret_cast<float4 *>(&C[baseIdx]) = out0;

            // Next 4 columns
            float4 acc1;
            acc1.x = c[i][4];
            acc1.y = c[i][5];
            acc1.z = c[i][6];
            acc1.w = c[i][7];

            float4 old1 = *reinterpret_cast<const float4 *>(&C[baseIdx + 4]);

            float4 out1;
            out1.x = alpha * acc1.x + beta * old1.x;
            out1.y = alpha * acc1.y + beta * old1.y;
            out1.z = alpha * acc1.z + beta * old1.z;
            out1.w = alpha * acc1.w + beta * old1.w;

            *reinterpret_cast<float4 *>(&C[baseIdx + 4]) = out1;
        }
        else
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                int ccol = col0 + j;
                if (ccol >= N)
                    continue;

                int idx = baseIdx + j;
                float old = C[idx];
                C[idx = idx] = alpha * c[i][j] + beta * old;
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
    dim3 block(8, 8);                    // 8 x 8 threads = 64 threads per block
    dim3 grid(N / BLOCK_N, N / BLOCK_M); // N / 64 in each dimension

    sgemm<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}

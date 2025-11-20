// sgemm naive with gm coalesing

__global__ void sgemm(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N)
        return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
    {
        sum += A[row * N + k] * B[k * N + col];
    }
    int idx = row * N + col;
    C[idx] = alpha * sum + beta * C[idx];
}

void run_sgemm(const float *A_d, const float *B_d, float *C_d, int N, float alpha, float beta)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    sgemm<<<grid, block>>>(A_d, B_d, C_d, N, alpha, beta);
}
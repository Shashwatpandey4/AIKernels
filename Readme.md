# AI kernels 

This repository contains, CUDA, Triton, Mojo implementations of various kernels. with a worklog of each implementation

these number are on rtx 4060 laptop gpu

naive with gloabl memory coalescing - 850 gflops
smem tiled - 1120 gflops
2x2 register tiled - 2192 gflops
4x4 register tiled - 3532 gflops
vectorized ld/st - 3846 gflops


profiling : `ncu --set full ./sgemm`
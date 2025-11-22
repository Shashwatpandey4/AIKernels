# AI kernels 

This repository contains, CUDA, Triton, Mojo implementations of various kernels. with a worklog of each implementation

CuBLAS
--------
kernel | device| GFLOPS
-------|-------|-------
CuBLAS |A6000| 23286.411
CuBLAS | RTX 4060 (laptop)| 7380

CUDA SGEMM Kernels
----
kernel | device| GFLOPS
-------|-------|-------
Naive (Global Memory Coalesing) |A6000| x
Naive (Global Memory Coalesing) | RTX 4060 (laptop)| 835.507
Shared Mem Tiled |A6000||
Shared Mem Tiled |RTX 4060|1107.403|
2x2 Register Tiling |A6000||
2x2 Register Tiling |RTX 4060||
4x4 Register Tiling |A6000||
4x4 Register Tiling |RTX 4060||
Vectorized ld/st |A6000||
Vectorized ld/st |RTX 4060||
cp_async with double buffering |A6000||
cp_async with double buffering |RTX 4060||
cp_async with double buffering with 64x64 tile |A6000||
cp_async with double buffering with 64x64 tile |RTX 4060||
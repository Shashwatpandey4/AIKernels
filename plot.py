#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Raw GFLOP/s data per matrix size (from your logs)
# ---------------------------------------------------------------------

data_gflops = {
    1024: [
        22537.91,
        22539.04,
        22610.80,
        22610.80,
        22623.00,
        22619.57,
        22623.00,
    ],
    2048: [
        26155.14,
        26206.21,
        26210.11,
        26210.31,
        26204.16,
        26206.08,
        26206.72,
    ],
    3072: [
        27253.44,
        27258.68,
        27258.87,
        27259.34,
        27257.37,
        27260.65,
        27262.79,
        27256.72,
    ],
    4096: [
        27173.70,
        28641.67,
        28724.42,
        28852.86,
        29673.83,
        28720.12,
        28811.98,
        28836.74,
        28927.56,
    ],
    5120: [
        27361.35,
        28168.60,
        28137.30,
        28115.88,
        28747.17,
        28130.51,
        28506.96,
        28138.36,
        28195.41,
    ],
    6144: [
        28020.92,
        28048.34,
        28109.08,
        28085.90,
        28166.67,
        28187.51,
        28198.13,
        27814.80,
        27793.55,
        28220.54,
    ],
    7168: [
        28058.22,
        28089.29,
        28002.68,
        27990.65,
        28037.66,
        27963.48,
        27987.00,
        28012.23,
    ],
    8192: [
        27060.64,
        27054.88,
        27007.01,
        27216.17,
        26892.35,
        26828.96,
        26818.74,
        26842.20,
        26787.12,
    ],
}

# ---------------------------------------------------------------------
# Compute averages (convert GFLOP/s -> TFLOP/s)
# ---------------------------------------------------------------------

sizes = np.array(sorted(data_gflops.keys()))
avg_tflops = np.array([np.mean(data_gflops[n]) / 1e3 for n in sizes])

for n, avg in zip(sizes, avg_tflops):
    print(f"M=N=K={n:4d}  avg = {avg:.3f} TFLOP/s")

# ---------------------------------------------------------------------
# Plot: green measured line, red dashed theoretical peak at 32 TFLOP/s
# ---------------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(
    sizes,
    avg_tflops,
    marker="o",
    color="green",
    label="Measured avg TFLOPs",
)

theoretical_peak = 32.0
plt.axhline(
    theoretical_peak,
    color="red",
    linestyle="--",
    label="Theoretical peak (32 TFLOPs)",
)

plt.xlabel("Matrix size (M = N = K)")
plt.ylabel("TFLOP/s")
plt.xticks(sizes)

plt.title("FP16 Tensor Core GEMM Throughput vs Matrix Size")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save instead of (or in addition to) showing
plt.savefig("tc_flops_vs_size.png", dpi=200)
# plt.show()

print("Saved plot to tc_flops_vs_size.png")

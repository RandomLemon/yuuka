# Hardware and Toolchain Report

This document records measured facts for the reference machine. It is Phase 0
evidence, not a specification. Re-run the commands after a kernel, firmware, or
ROCm change before trusting the numbers.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-11-11 |
| Commit | implementation commit containing this report |
| Host kernel | Linux 6.17.13-21-pve |
| OS | Debian GNU/Linux 13 (trixie) |
| CPU | AMD Ryzen AI MAX+ 395, 16 cores / 32 threads |
| Memory | 120 GiB visible, shared with the GPU |
| GPU | AMD Radeon 8060S Graphics, `gfx1151` |
| GPU compute units | 40 per `rocminfo`; `hipDeviceProp_t` reports 20 workgroup processors |
| Wavefront | 32 |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| HIP compiler | AMD clang 24.0.0git |
| Go | 1.24.4 |
| CMake | 3.31.6 |
| amdgpu module | loaded |

`rocminfo` reports 40 CUs. `hipDeviceProp_t.multiProcessorCount` reports 20,
which corresponds to workgroup processors (two CUs each on this RDNA part).
Kernel dispatch decisions use the 40-CU figure from `rocminfo`, not the HIP
property.

## Validation commands

```bash
./scripts/hardware_report.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
./build/yuuka_hardware_probe
```

The C ABI test and the quantized GEMV test run through CTest. The hardware probe
is a benchmark and is run explicitly.

## Memory measurements

Size is 256 MiB per operation. Values are measured with HIP events and include
five repetitions. These are effective throughputs for the whole operation, not
hardware counters. `D2D` counts both read and write traffic as the buffer size.

| Operation | Effective throughput |
| --- | ---: |
| `hipMemcpy` host-to-device, pinned source | 67.4 GB/s |
| `hipMemcpy` device-to-host, pinned destination | 65.9 GB/s |
| `hipMemset` device memory | 230.1 GB/s |
| `hipMemcpy` device-to-device | 105.6 GB/s |
| Mapped host memory written by the GPU (`hipMemcpy` semantics) | 178.6 GB/s |
| `hipMalloc` 16 MiB, 64 allocations | 46.9 ms total |

The measured host/device copy path is much slower than the device-local write
rate. The initial memory policy is therefore:

- keep weights in device allocations for the model lifetime;
- stage uploads through pinned host memory;
- keep activation, KV, and recurrent state in device memory;
- do not assume mapped host memory is a zero-cost replacement for device
  allocations;
- size scratch once during planning, not per token.

These measurements do not yet cover page migration, prefetch behavior, or
long-running thermal effects.

## rocBLAS

A square FP32 SGEMM with `M=N=K=1024`, five repetitions:

| Operation | Time | Effective throughput |
| --- | ---: | ---: |
| `rocblas_sgemm` 1024^3 | 0.004 s | 3.01 TFLOP/s |

This is a smoke test and a baseline, not a tuned result. The matrix is small
relative to the 40-CU device, and the measurement was not collected under a
controlled power profile.

## rocWMMA

A `16x16x16` FP16 WMMA tile with FP32 accumulation compiles for `gfx1151` and
matches a CPU reference within `1e-3`. This confirms that the RDNA WMMA path is
available to the toolchain and that a wave32 launch works on this device.

## C ABI and kernels

- `ctest` passes the allocation, upload, stream, event, and error-propagation
  round trip.
- Quantized GEMV kernels for `F16`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K` match a
  CPU dequantization reference on random packed rows and alignment boundaries.
- The HIP kernels are integrated into the model executor through a planned
  operation stream. The CPU reference path remains the default and the
  correctness oracle.

## Decisions recorded

1. Device execution uses explicit non-blocking streams and single-use events.
2. Weights remain packed; kernels dequantize inside the multiply.
3. Host/device transfers use pinned staging buffers.
4. Mapped host memory is measured but not selected by default.
5. WMMA is available; the prefill GEMM path can target it in a later phase.
6. The CPU reference path remains the correctness oracle after HIP integration.

## Remaining Phase 0 work

- Power, clock, and thermal characterization under sustained load.
- Firmware and driver version capture through a stable source.
- A larger rocBLAS shape sweep and a WMMA tiled GEMM benchmark.
- A final decision on which buffer classes, if any, should use mapped memory.

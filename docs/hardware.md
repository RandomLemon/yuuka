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
scripts/thermal_probe.sh 15 ./build/yuuka_hardware_probe
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
./build/yuuka_hardware_probe
```

## Firmware and power management

Captured on the reference machine with `./scripts/hardware_report.sh`:

| Item | Value |
| --- | --- |
| VBIOS | `113-STRXLGEN-001` |
| IFWI | `AMD STRIX_HALO_GENERIC`, built 2024/06/17, part `113-STRXLGEN-001`, version `023.011.000.039.000001` |
| `rlc_fw_version` | `0x11530506` |
| `smc_fw_version` | `0x0a640700` |
| `sdma_fw_version` | `0x00000012` |
| `vcn_fw_version` | `0x09118037` |
| `mes_fw_version` | `0x00000096` |
| `power_dpm_force_performance_level` | `auto` |
| `power_dpm_state` | `performance` |

`amd-smi static -d` reports the `amdgpu` driver name but `VERSION: N/A`; the
kernel module does not expose a version string on this host. The kernel version
above is the stable identifier for driver comparisons. `amd-smi metric -v`
reports all throttle accumulators as `N/A` on this APU, so throttle residency
cannot be quantified from that interface.

## Sustained-load telemetry

`scripts/thermal_probe.sh` samples the hwmon `power1_input` and `temp1_input`
plus the active `pp_dpm_sclk` entry while repeating a load command. A 15-second
run of the hardware probe on the reference machine produced:

| Metric | Result |
| --- | ---: |
| Samples | 14 |
| Socket power average | 28.40 W |
| Socket power minimum / maximum | 24.00 W / 30.01 W |
| Edge temperature average | 46.29 °C |
| Edge temperature minimum / maximum | 46.00 °C / 47.00 °C |
| Active SCLK at end | 1215 MHz |

This is a short smoke test, not a thermal soak: it does not establish steady
state, fan behavior, or long-running throttling. The hwmon power reading is
socket power, not a GPU-only counter. Re-run the script with a longer duration
before using the values for a sustained-performance claim.

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

Run-to-run variation: a later probe on the same machine measured `hipMemset`
at 190.5 GB/s, while H2D/D2H, D2D, mapped memory, allocation, and rocBLAS
results stayed within a few percent of the table. The memset figure is sensitive
to the run; re-measure under a controlled power profile before using it for a
bandwidth budget.

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

## Post-gate follow-up measurements

The Phase 0 exit criteria are met by the measurements above. The following
follow-ups would strengthen the evidence but are not Phase 0 gates:

- Longer thermal soak with fan and steady-state power recording.
- A larger rocBLAS shape sweep and a WMMA tiled GEMM benchmark.
- A final decision on which buffer classes, if any, should use mapped memory.

## NPU capability probe (diagnostic)

`go run ./cmd/yuuka npu-probe` on the reference host reports:

```text
state: device_only
device: /dev/accel/accel0
driver: true
userspace: false
reason: device and amdxdna driver are present, but no XRT userspace runtime was found
fallback: selected backend
```

The XDNA device node and the `amdxdna` kernel module are present. No
`xrt-smi`, `libxrt_core*`, or XRT installation directory was found, so the NPU
userspace prerequisite is unmet. The project does not claim NPU inference
support; see [npu.md](npu.md).

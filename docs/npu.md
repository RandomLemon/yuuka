# NPU Heterogeneous Expansion

## 1. Status and position

The XDNA 2 NPU in Ryzen AI Max+ 395 is a potential accelerator for `yuuka`, but
it is **not part of the first release**. The GPU path must remain correct,
performant, and self-sufficient without any NPU runtime.

Theoretical NPU TOPS are not equivalent to LLM inference speedup. End-to-end
benefit depends on data movement, quantization compatibility, graph compilation,
synchronization, and how much of the model can run on the NPU without
fragmenting execution.

## 2. Execution strategies

| Strategy | Description | Status |
| --- | --- | --- |
| **GPU-only** | All model stages run on the HIP backend | Default; first release |
| **CPU + GPU** | CPU reference or fallback for unsupported operations; GPU for the main path | Correctness and fallback |
| **NPU prefill + GPU decode** | Whole-phase split: NPU processes prompt chunks, GPU generates tokens | Experimental |
| **NPU subgraph + GPU subgraph** | Partition the model graph across NPU and GPU with explicit transfers | Experimental, later |

Phase-level splitting is the first candidate because it avoids per-layer
handoffs. Subgraph partitioning is more complex and requires a cost model,
device-affinity metadata, and transfer scheduling.

## 2.1 Current probe result

`yuuka npu-probe` and the opt-in `--npu` flag report the following on the
reference host (2026-11-11):

```text
state: device_only
device: /dev/accel/accel0
driver: true
userspace: false
reason: device and amdxdna driver are present, but no XRT userspace runtime was found
fallback: selected backend
```

The XDNA device node and the `amdxdna` kernel module are present, but no
`xrt-smi`, `libxrt_core`, or XRT library directory was found. The toolchain
prerequisite in section 6 is therefore unmet, and no NPU kernel, graph compiler,
or subgraph partition is shipped. `--npu` records the probe result and falls back
to the selected GPU backend; it never silently changes results. The Phase 5
deliverable is therefore the capability probe and explicit fallback; NPU
prefill/decode and subgraph execution are deferred until the activation criteria
in section 6 are met.

## 3. Why it is deferred

- The Linux NPU toolchain, driver, and runtime may not be available or stable
  enough for the target deployment.
- GGUF quantization layouts may not map directly to the NPU graph compiler;
  conversion can add startup cost and memory copies.
- Keeping a second weight copy for the NPU may exceed the memory budget on an
  APU.
- KV-cache and execution-state handoff between devices can erase the benefit of
  offloading a phase.
- CPU-mediated synchronization and control can dominate small decode steps.
- The GPU path must be measured and optimized first; otherwise there is no
  meaningful baseline for comparison.

## 4. Interface reservations

`backend/npu` currently exposes only a capability probe and fallback decision.
The architecture reserves the following concepts without implementing them in
the first release:

| Concept | Purpose |
| --- | --- |
| `Subgraph` | A region of the execution plan that can be assigned to a device |
| `DeviceAffinity` | Constraints and preferences for placing a subgraph |
| `Transfer` | Explicit data movement between device memory spaces |
| `Event` | Cross-device dependency and completion |
| Capability descriptor | Supported dtypes, quantization types, shapes, and graph features |

The GPU backend must not assume it is the only backend. The planner must be able
to represent an unsupported operation as a fallback rather than a hard failure.
However, these abstractions must not add overhead to the GPU-only hot path.

## 5. Technical questions

Before an NPU path can be enabled, answer these with measurements:

1. **Toolchain and runtime.** Which compiler, driver, and runtime versions are
   available on the target Linux distribution? What is the support matrix?
2. **Quantization compatibility.** Can the NPU compiler consume GGUF-native
   quantized layouts, or is a converted format required? What is the conversion
   cost and numerical impact?
3. **Weight residency.** Does the NPU require a separate weight copy? What is
   the memory cost on a unified-memory APU?
4. **KV handoff.** Can KV cache or execution state be transferred between NPU
   and GPU efficiently, or must each phase recompute state?
5. **Synchronization.** What is the cost of CPU-mediated synchronization per
   prefill chunk or decode step?
6. **Partitioning.** Which whole phases or subgraphs have a favorable
   compute-to-transfer ratio?
7. **Fallback.** Can the runtime detect an unsupported operation and fall back
   to GPU without restarting the request?

## 6. Activation criteria

An NPU path is considered only when (none of these are met on the current host;
the first condition fails because the XRT userspace runtime is absent):

- the toolchain and runtime are available and reproducible on the target;
- a correctness path exists against the CPU reference and GPU output;
- end-to-end prefill or decode improves measurably under the benchmark protocol
  in [testing.md](testing.md);
- the path is opt-in and can fall back to GPU;
- memory consumption remains within the APU budget;
- the GPU-only path is not blocked, slowed, or complicated by NPU support.

The default remains GPU-only even after an NPU path exists.

## 7. Non-goals

- Automatic fine-grained operator partitioning across GPU and NPU.
- An NPU-only first release.
- NPU scheduling in the first release.
- Relying on theoretical TOPS as evidence of LLM speedup.
- Duplicating model weights without accounting for APU memory pressure.

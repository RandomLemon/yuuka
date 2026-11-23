#!/usr/bin/env bash
# Prints the compiler-reported VGPR, SGPR, LDS, scratch, and occupancy figures
# for the dispatched kernel families. Dynamic shared memory is allocated by the
# launcher and is not included in the static LDS number.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hipcc="${HIPCC:-hipcc}"
arch="${YUUKA_GPU_TARGET:-gfx1151}"
out="${TMPDIR:-/tmp}/yuuka-kernel-resources.o"

for source in \
    kernels/hip/gemv/gemv_quant.hip \
    kernels/hip/moe/moe.hip \
    kernels/hip/gemm/wmma_gemm.hip \
    kernels/hip/gemm/mmq_gemm.hip \
    kernels/hip/moe/moe_mmq.hip \
    kernels/hip/gemm/lm_head.hip \
    kernels/hip/prefill/prefill.hip \
    kernels/hip/prefill/attention_wmma.hip; do
    echo "== ${source}"
    definitions=()
    if [[ "${arch}" == gfx1151 && "${source}" == kernels/hip/gemm/wmma_gemm.hip ]]; then
        definitions+=(-DYUUKA_DIRECT_FP16_WMMA)
    fi
    "${hipcc}" --offload-arch="${arch}" -O3 -std=c++17 \
        -I"${root}/backend/hip/bridge" -I"${root}/kernels/hip" \
        "${definitions[@]}" \
        -c "${root}/${source}" -o "${out}" \
        -Rpass-analysis=kernel-resource-usage 2>&1 |
        grep -E 'Function Name|TotalSGPRs|VGPRs:|ScratchSize|Occupancy|VGPRs Spill|LDS Size'
done
rm -f "${out}"

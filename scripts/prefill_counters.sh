#!/usr/bin/env bash
# Reproduce the prefill counter attribution of docs/benchmarks/qwen3.5_4b.md.
#
# Run build/yuuka_prefill_sweep under rocprofv3 with the
# instruction counters and again with the stall/occupancy counters, optionally
# trace a llama.cpp reference run, and reduce both databases to the per-MMA
# table. The reduction pairs gemm_mmq dispatches with sweep cells in submission
# order. Denominator: seven submissions per cell (two warmups plus five timed),
# and n_in * n_out * rows / 4096 useful MMAs per submission. See
# scripts/prefill_counters.py for the executed-MMA correction on padded cells.
#
# Requirements: a built build/libyuuka_hip.so and build/yuuka_prefill_sweep
# (cmake --build build), plus rocminfo, rocprofv3, and python3 with sqlite3.
# rocprofv3 does not flush its database for a Go process, so this script never
# profiles the Go binaries; per-operation attribution for the engine comes from
# TestGPUProfilePrefill (docs/testing.md section 6.3).
#
# The recorded attribution belongs to one GGUF and one device. The script fails
# when the GPU is not gfx1151, or when a supplied model file does not match the
# recorded SHA-256.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${YUUKA_BUILD_DIR:-${root}/build}"
sweep="${build}/yuuka_prefill_sweep"
library="${build}/libyuuka_hip.so"
reducer="${root}/scripts/prefill_counters.py"

# The attribution is only comparable against this
# model and this device.
recorded_model_sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
expected_target="${YUUKA_GPU_TARGET:-gfx1151}"

model="${YUUKA_TEST_MODEL:-}"
llama_bench=""
outdir=""
keep=0

usage() {
    cat <<'EOF'
usage: prefill_counters.sh [options]

Collect and reduce the per-MMA prefill counters.

  --model PATH        GGUF used for the reference run. Its SHA-256 must match
                      the recorded attribution hash. Defaults to
                      $YUUKA_TEST_MODEL.
  --llama-bench PATH  llama-bench binary. When set, the script traces
                      "llama-bench -m MODEL -p 512 -n 0 -r 1 -ngl 99 -b 512
                      -ub 512" and reports the reference counters.
  --reference-kernel-regex RE
                      Selects the reference MMQ kernels (default: mul_mat_q).
  --reference-mma-counts PATH
                      One executed MMA count per matched reference dispatch, in
                      trace order. Without it the reference side is normalized
                      per wave, because an MMA count needs the reference's
                      matrix shapes.
  --output DIR        Keep rocprofv3 databases and logs in DIR. Default is a
                      temporary directory that is removed on exit.
  -h, --help          Print this help.

Environment: YUUKA_BUILD_DIR, YUUKA_GPU_TARGET, YUUKA_TEST_MODEL,
YUUKA_LLAMA_BENCH_ARGS (replaces the default llama-bench arguments),
YUUKA_LLAMA_BENCH_KERNEL_REGEX.

Exit status: 0 on a complete run, 2 on a usage, device, model, or reduction
failure.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --model)
            (($# >= 2)) || die "--model needs a path"
            model="$2"
            shift 2
            ;;
        --llama-bench)
            (($# >= 2)) || die "--llama-bench needs a path"
            llama_bench="$2"
            shift 2
            ;;
        --reference-kernel-regex)
            (($# >= 2)) || die "--reference-kernel-regex needs a pattern"
            reference_kernel_regex="$2"
            shift 2
            ;;
        --reference-mma-counts)
            (($# >= 2)) || die "--reference-mma-counts needs a path"
            reference_mma_counts="$2"
            shift 2
            ;;
        --output)
            (($# >= 2)) || die "--output needs a directory"
            outdir="$2"
            keep=1
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -n "${reference_kernel_regex:-}" ]] || reference_kernel_regex="${YUUKA_LLAMA_BENCH_KERNEL_REGEX:-mul_mat_q}"
[[ -n "${reference_mma_counts:-}" ]] || reference_mma_counts=""

for tool in rocprofv3 rocminfo python3 sha256sum grep sed mktemp cmp; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done
[[ -f "$library" ]] || die "missing $library; run: cmake --build $build"
[[ -x "$sweep" ]] || die "missing $sweep; run: cmake --build $build"
[[ -f "$reducer" ]] || die "missing $reducer"

rocm_info="$(rocminfo 2>/dev/null)" || die "rocminfo failed"
# Print the rocminfo agent block whose device name is the expected target. The
# name field is matched per line because a whole-block match would also accept
# the "amdgcn-amd-amdhsa--gfx1151" ISA line without proving the agent identity.
gpu_agent="$(
    printf '%s\n' "$rocm_info" | awk -v want="$expected_target" '
        /^Agent [0-9]+/ {
            if (gpu) exit
            gpu = 0
            block = ""
        }
        $0 ~ ("^[[:space:]]*Name:[[:space:]]*" want "[[:space:]]*$") { gpu = 1 }
        { if (gpu) block = block $0 "\n" }
        END { printf "%s", block }
    '
)"
[[ -n "$gpu_agent" ]] ||
    die "rocminfo reported no '$expected_target' agent; the recorded attribution was measured on gfx1151"

device_target="$(printf '%s\n' "$gpu_agent" | sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9a-f]*\)[[:space:]]*$/\1/p' | head -n1)"
device_name="$(printf '%s\n' "$gpu_agent" | sed -n 's/^[[:space:]]*Marketing Name:[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1/p' | head -n1)"
device_cus="$(printf '%s\n' "$gpu_agent" | sed -n 's/^[[:space:]]*Compute Unit:[[:space:]]*\([0-9]*\).*/\1/p' | head -n1)"
[[ "$device_target" == "$expected_target" ]] ||
    die "device mismatch: found '$device_target', required target is '$expected_target' (gfx1151 in the recorded attribution)"

rocm_version="$(rocprofv3 --version 2>/dev/null | sed -n 's/^[[:space:]]*rocm_version:[[:space:]]*//p' | head -n1)"
hip_version=""
if command -v hipcc >/dev/null 2>&1; then
    hip_version="$(hipcc --version 2>/dev/null | sed -n 's/^HIP version:[[:space:]]*//p' | head -n1)"
fi
power_level=""
for level in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
    [[ -r "$level" ]] || continue
    power_level="$(<"$level")"
    break
done

if [[ -n "$model" ]]; then
    [[ -r "$model" ]] || die "model file not readable: $model"
    model_hash="$(sha256sum "$model" | cut -d' ' -f1)"
    [[ "$model_hash" == "$recorded_model_sha256" ]] ||
        die "model hash mismatch for $model: got $model_hash, recorded $recorded_model_sha256"
    model_state="verified"
else
    model_hash="$recorded_model_sha256"
    model_state="unverified (no --model; the sweep does not read the GGUF)"
fi

if [[ -n "$llama_bench" ]]; then
    [[ -x "$llama_bench" ]] || die "llama-bench is not executable: $llama_bench"
    [[ -n "$model" ]] || die "--llama-bench needs --model so the reference run compares the same GGUF"
fi
if [[ -n "$reference_mma_counts" && -z "$llama_bench" ]]; then
    die "--reference-mma-counts needs --llama-bench"
fi

if [[ -z "$outdir" ]]; then
    outdir="$(mktemp -d "${TMPDIR:-/tmp}/yuuka-prefill-counters.XXXXXX")"
fi
mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"
cleanup() {
    if ((keep == 0)); then
        rm -rf "$outdir"
    fi
}
trap cleanup EXIT

shape_lines() {
    grep -oE '^[A-Za-z0-9_]+ +M= *[0-9]+ +N= *[0-9]+ +K= *[0-9]+' "$1" || true
}

echo "== environment"
echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "host=$(uname -sr)"
echo "device=$device_name ($device_target), rocminfo_compute_units=$device_cus"
echo "power_dpm_force_performance_level=${power_level:-unknown}"
echo "rocm_version=${rocm_version:-unknown}"
echo "hip_version=${hip_version:-unknown}"
echo "model_sha256=$model_hash ($model_state)"
echo "sweep=$sweep"
echo "rocprofv3_version=$(rocprofv3 --version 2>/dev/null | sed -n 's/^[[:space:]]*version:[[:space:]]*//p' | head -n1)"
echo

echo "== unprofiled sweep (cell list and achieved throughput)"
sweep_log="${outdir}/sweep.log"
if "$sweep" >"$sweep_log" 2>"${outdir}/sweep.err"; then
    cat "$sweep_log"
else
    status=$?
    if [[ "$status" -eq 77 ]]; then
        die "no HIP device available to $sweep"
    fi
    die "$sweep failed with status $status; see ${outdir}/sweep.err"
fi
[[ -s "$sweep_log" ]] || die "$sweep produced no output"

profile() {
    local name="$1"
    shift
    local log="${outdir}/${name}.log"
    echo "== rocprofv3 ${name}"
    # rocprofv3 drops a .rocprofv3 scratch directory in the working directory;
    # run it inside the output directory so the run leaves no stray files.
    if ! (cd "$outdir" && rocprofv3 -d "$outdir" -o "$name" -f rocpd --kernel-trace true "$@" \
        -- "$sweep") >"$log" 2>"${outdir}/${name}.err"; then
        die "rocprofv3 ${name} failed; see ${outdir}/${name}.err"
    fi
    [[ -f "${outdir}/${name}_results.db" ]] ||
        die "rocprofv3 ${name} produced no database in $outdir"
    if ! cmp -s <(shape_lines "$sweep_log") <(shape_lines "$log"); then
        die "the profiled ${name} run exercised different cells than the unprofiled sweep"
    fi
}

profile instructions --pmc SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVES
profile stalls --pmc LDSBankConflict ALUStalledByLDS LdsLatency OccupancyPercent

reference_args=()
if [[ -n "$llama_bench" ]]; then
    if [[ -n "${YUUKA_LLAMA_BENCH_ARGS:-}" ]]; then
        read -r -a llama_args <<<"$YUUKA_LLAMA_BENCH_ARGS"
    else
        llama_args=(-m "$model" -p 512 -n 0 -r 1 -ngl 99 -b 512 -ub 512)
    fi
    profile_reference() {
        local name="$1"
        shift
        echo "== rocprofv3 ${name} (llama.cpp reference)"
        if ! (cd "$outdir" && rocprofv3 -d "$outdir" -o "$name" -f rocpd --kernel-trace true "$@" \
            -- "$llama_bench" "${llama_args[@]}") >"${outdir}/${name}.log" 2>"${outdir}/${name}.err"; then
            die "rocprofv3 ${name} failed; see ${outdir}/${name}.err"
        fi
        [[ -f "${outdir}/${name}_results.db" ]] ||
            die "rocprofv3 ${name} produced no database in $outdir"
    }
    profile_reference reference-instructions --pmc SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVES
    profile_reference reference-stalls --pmc LDSBankConflict ALUStalledByLDS LdsLatency OccupancyPercent
    reference_args=(
        --reference-instructions-db "${outdir}/reference-instructions_results.db"
        --reference-stalls-db "${outdir}/reference-stalls_results.db"
        --reference-kernel-regex "$reference_kernel_regex"
    )
    if [[ -n "$reference_mma_counts" ]]; then
        reference_args+=(--reference-mma-counts "$reference_mma_counts")
    fi
fi

echo
python3 "$reducer" \
    --cells "$sweep_log" \
    --instructions-db "${outdir}/instructions_results.db" \
    --stalls-db "${outdir}/stalls_results.db" \
    ${reference_args[@]+"${reference_args[@]}"} ||
    die "counter reduction failed"

echo
if ((keep == 1)); then
    echo "rocprofv3 databases and logs kept in $outdir"
else
    echo "rocprofv3 databases and logs removed (pass --output DIR to keep them)"
fi

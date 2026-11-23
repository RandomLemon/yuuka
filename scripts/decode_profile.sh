#!/usr/bin/env bash
# Reproduce the decode matvec attribution of docs/benchmarks/qwen3.5_4b.md.
#
# One command for both halves of the attribution:
#
# 1. run build/yuuka_gemv_decode_sweep once unprofiled, then trace it with
#    rocprofv3 kernel tracing and the instruction counters, and reduce both to
#    the per-shape GB/s table of docs/benchmarks/qwen3.5_4b.md with the per-weight
#    instruction counts. The sweep keeps every matrix resident, so its numbers
#    are the hot-matrix upper bound.
# 2. run the rows = 1 output-projection crossover sweep, the measurement behind
#    lmHeadMinOutputs, the int8 matvec comparison, and the isolated delta-net
#    RECURRENT cell.
# 3. run TestGPUProfileDecode, which attributes one real decode step per planned
#    operation and per tensor.
#
# Every step here is a diagnostic, not a release measurement; see
# docs/testing.md section 6.3.
#
# Requirements: a built build/libyuuka_hip.so and build/yuuka_gemv_decode_sweep
# (cmake --build build), plus rocminfo, rocprofv3, and python3 with sqlite3.
# rocprofv3 cannot write its database for a Go process, so this script profiles
# only the native sweep.
#
# The recorded attribution belongs to one GGUF and one device. The script fails
# when the GPU is not gfx1151, or when a supplied model file does not match the
# recorded SHA-256.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${YUUKA_BUILD_DIR:-${root}/build}"
sweep="${build}/yuuka_gemv_decode_sweep"
library="${build}/libyuuka_hip.so"
reducer="${root}/scripts/decode_profile.py"

# The attribution is only comparable against this
# model and this device.
recorded_model_sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
expected_target="${YUUKA_GPU_TARGET:-gfx1151}"

model="${YUUKA_TEST_MODEL:-}"
outdir=""
keep=0
waves_only=0
run_engine=1

usage() {
    cat <<'EOF'
usage: decode_profile.sh [options]

Collect and reduce the per-shape decode matvec table.

  --model PATH   GGUF whose shape list the sweep reproduces. Its SHA-256 must
                 match the recorded attribution hash. Defaults to
                 $YUUKA_TEST_MODEL; without a model the hash is only echoed.
  --waves-only   Trace SQ_WAVES only, without the instruction counters. Use it
                 when the instruction counters are unavailable on the device.
  --no-engine    Skip the per-operation attribution of one real decode step
                 (TestGPUProfileDecode), which needs the model.
  --output DIR   Keep rocprofv3 databases and logs in DIR. Default is a
                 temporary directory that is removed on exit.
  -h, --help     Print this help.

Environment: YUUKA_BUILD_DIR, YUUKA_GPU_TARGET, YUUKA_TEST_MODEL.

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
        --waves-only)
            waves_only=1
            shift
            ;;
        --no-engine)
            run_engine=0
            shift
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

for tool in rocprofv3 rocminfo python3 sha256sum grep sed mktemp; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done
[[ -f "$library" ]] || die "missing $library; run: cmake --build $build"
[[ -x "$sweep" ]] || die "missing $sweep; run: cmake --build $build"
[[ -f "$reducer" ]] || die "missing $reducer"

rocm_info="$(rocminfo 2>/dev/null)" || die "rocminfo failed"
# Match the agent block whose Name is the target; a whole-block match would also
# accept the "amdgcn-amd-amdhsa--gfx1151" ISA line without proving the identity.
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
    die "device mismatch: found '$device_target', required target is '$expected_target'"

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
    model_state="unverified (no --model; the sweep reproduces the shape list, it does not read the GGUF)"
fi

if [[ -z "$outdir" ]]; then
    outdir="$(mktemp -d "${TMPDIR:-/tmp}/yuuka-decode-profile.XXXXXX")"
fi
mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"
cleanup() {
    if ((keep == 0)); then
        rm -rf "$outdir"
    fi
}
trap cleanup EXIT

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

echo "== unprofiled sweep (host timing, hot matrix)"
cells="${outdir}/cells.csv"
if "$sweep" >"$cells" 2>"${outdir}/sweep.err"; then
    cat "$cells"
else
    status=$?
    if [[ "$status" -eq 77 ]]; then
        die "no HIP device available to $sweep"
    fi
    die "$sweep failed with status $status; see ${outdir}/sweep.err"
fi
[[ -s "$cells" ]] || die "$sweep produced no output"

# The traced run must exercise the same cells in the same order, otherwise the
# reduction would pair a dispatch with the wrong shape. The timings differ
# between the two runs, so only the identity columns are compared.
image_lines() {
    awk -F, 'NR > 1 { print $1 "," $2 "," $3 "," $4 "," $8 "," $9 }' "$1"
}

if ((waves_only == 1)); then
    pmc_args=(--pmc SQ_WAVES)
else
    pmc_args=(--pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS)
fi
echo
echo "== rocprofv3 kernel trace (${pmc_args[*]})"
profile_log="${outdir}/profile.log"
# rocprofv3 drops a .rocprofv3 scratch directory in the working directory, so
# run it inside the output directory to leave no stray files behind.
if ! (cd "$outdir" && rocprofv3 -d "$outdir" -o profile -f rocpd --kernel-trace true \
    "${pmc_args[@]}" -- "$sweep") >"$profile_log" 2>"${outdir}/profile.err"; then
    die "rocprofv3 failed; see ${outdir}/profile.err"
fi
[[ -f "${outdir}/profile_results.db" ]] ||
    die "rocprofv3 produced no database in $outdir"
if ! cmp -s <(image_lines "$cells") <(image_lines "$profile_log"); then
    die "the traced run exercised different cells than the unprofiled sweep"
fi

echo
reducer_args=(--cells "$cells" --db "${outdir}/profile_results.db")
if ((waves_only == 1)); then
    reducer_args+=(--waves-only)
fi
python3 "$reducer" "${reducer_args[@]}" || die "reduction failed"

echo
echo "== output projection crossover, rows = 1 (lmHeadMinOutputs)"
if ! "$sweep" --threshold; then
    die "$sweep --threshold failed"
fi

echo
echo "== int8 decode matvec against the FP32 streaming kernel"
if ! "$sweep" --int8; then
    die "$sweep --int8 failed"
fi

echo
echo "== delta-net unit of one recurrent layer (RECURRENT cell)"
if ! "$sweep" --recurrent; then
    die "$sweep --recurrent failed"
fi

echo
if ((run_engine == 1)); then
    if [[ -z "$model" ]]; then
        echo "== per-operation attribution skipped: no --model or YUUKA_TEST_MODEL" >&2
    else
        echo "== per-operation attribution of one decode step (TestGPUProfileDecode)"
        if ! (cd "$root" && YUUKA_TEST_MODEL="$model" \
            go test -tags hip ./engine/gpu -run TestGPUProfileDecode -v -count=1) \
            >"${outdir}/engine.log" 2>&1; then
            sed -n '1,200p' "${outdir}/engine.log" >&2
            die "TestGPUProfileDecode failed; see ${outdir}/engine.log"
        fi
        # Drop the go test framing and the file:line prefix so the report reads
        # as one table set.
        sed -n 's/^[[:space:]]*decode_profile_test\.go:[0-9]*: //p' "${outdir}/engine.log"
        sed -n 's/^--- PASS.*/&/p' "${outdir}/engine.log"
    fi
fi

echo
if ((keep == 1)); then
    echo "rocprofv3 database, logs, and the engine report kept in $outdir"
else
    echo "rocprofv3 database and logs removed (pass --output DIR to keep them)"
fi

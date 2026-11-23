#!/usr/bin/env bash
# Reproduce the decode comparison against llama.cpp recorded in docs/benchmarks/qwen3.5_4b.md.
#
# One command for the two halves of the comparison:
#
# 1. the token-rate table: `yuuka bench` and `llama-bench` on the same GGUF,
#    same device, same session, at the prompt/decode cells of the recorded
#    protocol, with the token rate and the ratio per cell.
# 2. the per-shape weight-bandwidth table: the yuuka per-tensor attribution of
#    one real decode step (TestGPUProfileDecode) joined by (dtype, K, N) with a
#    rocprofv3 kernel trace of the reference decode loop, reduced by
#    scripts/decode_reference_shapes.py.
#
# The comparison is only valid for one model and one device, so the script
# fails when the GPU is not the recorded target or the model hash differs.
# Sessions are not clock-pinned, so absolute rates drift by a few percent; the
# ratio per cell is the figure that survives the drift.
#
# Requirements: the HIP Go build (the script builds it unless --yuuka is given),
# a built llama.cpp with llama-bench, rocminfo, rocprofv3, python3, sha256sum.
# Raw rocprofv3 databases stay under /tmp (or --output DIR) and never enter the
# repository.
#
# The reference trace is reduced from the dispatch pairing documented in
# scripts/decode_reference_shapes.py: the decode loop runs one quantize_q8_1 per
# matvec, so the quantize grid is that matvec's K.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reducer="${root}/scripts/decode_reference_shapes.py"

# The comparison is only comparable against this model and this device.
recorded_model_sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
expected_target="${YUUKA_GPU_TARGET:-gfx1151}"

model="${YUUKA_TEST_MODEL:-}"
llama_bench="${YUUKA_LLAMA_BENCH:-}"
yuuka_bin="${YUUKA_BIN:-}"
outdir=""
keep=0
reps=3
warmup=2
cells="128:128,128:32,128:512,512:128,2048:32"
run_trace=1

usage() {
    cat <<'EOF'
usage: decode_compare.sh --model PATH [options]

Print the decode comparison tables recorded in docs/benchmarks/qwen3.5_4b.md.

  --model PATH        GGUF both engines read. Its SHA-256 must match the recorded
                      comparison hash. Defaults to $YUUKA_TEST_MODEL.
  --llama-bench PATH  llama-bench binary. Defaults to $YUUKA_LLAMA_BENCH, then
                      ../llama.cpp/build/bin/llama-bench.
  --yuuka PATH        HIP-enabled yuuka binary. Defaults to $YUUKA_BIN; without
                      it the script builds ./cmd/yuuka with -tags hip into the
                      output directory.
  --cells LIST        prompt:decode cells, default "128:128,128:32,128:512,
                      512:128,2048:32".
  --reps N            measured repetitions, default 3.
  --warmup N          warmup repetitions, default 2.
  --no-trace          Skip the per-shape bandwidth table and the trace.
  --output DIR        Keep the built binary, binaries' logs, and the rocprofv3
                      database in DIR. Default is a temporary directory that is
                      removed on exit.
  -h, --help          Print this help.

Environment: YUUKA_TEST_MODEL, YUUKA_LLAMA_BENCH, YUUKA_BIN,
YUUKA_GPU_TARGET, YUUKA_DECODE_REPS.

Exit status: 0 on a complete run, 2 on a usage, device, model, or reduction
failure. One benchmark process must own the GPU; do not run this in parallel
with another benchmark.
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
        --yuuka)
            (($# >= 2)) || die "--yuuka needs a path"
            yuuka_bin="$2"
            shift 2
            ;;
        --cells)
            (($# >= 2)) || die "--cells needs a list"
            cells="$2"
            shift 2
            ;;
        --reps)
            (($# >= 2)) || die "--reps needs a number"
            reps="$2"
            shift 2
            ;;
        --warmup)
            (($# >= 2)) || die "--warmup needs a number"
            warmup="$2"
            shift 2
            ;;
        --no-trace)
            run_trace=0
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

[[ -n "$model" ]] || die "--model is required (or set YUUKA_TEST_MODEL)"
[[ -r "$model" ]] || die "model file not readable: $model"
[[ -f "$reducer" ]] || die "missing $reducer"

for tool in rocminfo sha256sum python3 awk sed mktemp; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done
((run_trace == 0)) || command -v rocprofv3 >/dev/null 2>&1 ||
    die "missing required tool: rocprofv3 (or pass --no-trace)"

rocm_info="$(rocminfo 2>/dev/null)" || die "rocminfo failed"
# Match the agent block whose Name is the target; a whole-block substring match
# would also accept the ISA line without proving the device identity.
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
    die "rocminfo reported no '$expected_target' agent; the recorded comparison was measured on gfx1151"
device_name="$(printf '%s\n' "$gpu_agent" | sed -n 's/^[[:space:]]*Marketing Name:[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1/p' | head -n1)"
device_cus="$(printf '%s\n' "$gpu_agent" | sed -n 's/^[[:space:]]*Compute Unit:[[:space:]]*\([0-9]*\).*/\1/p' | head -n1)"

model_hash="$(sha256sum "$model" | cut -d' ' -f1)"
[[ "$model_hash" == "$recorded_model_sha256" ]] ||
    die "model hash mismatch for $model: got $model_hash, the comparison was recorded against $recorded_model_sha256"

if [[ -z "$llama_bench" ]]; then
    for candidate in "${root}/../llama.cpp/build/bin/llama-bench" \
        "${root}/../strix-llama.cpp/build/bin/llama-bench"; do
        if [[ -x "$candidate" ]]; then
            llama_bench="$candidate"
            break
        fi
    done
fi
[[ -n "$llama_bench" ]] || die "no llama-bench found; pass --llama-bench PATH"
[[ -x "$llama_bench" ]] || die "llama-bench is not executable: $llama_bench"

if [[ -z "$outdir" ]]; then
    outdir="$(mktemp -d "${TMPDIR:-/tmp}/yuuka-decode-compare.XXXXXX")"
fi
mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"
cleanup() {
    if ((keep == 0)); then
        rm -rf "$outdir"
    fi
}
trap cleanup EXIT

if [[ -z "$yuuka_bin" ]]; then
    command -v go >/dev/null 2>&1 || die "missing go; pass --yuuka PATH to skip the build"
    echo "== building yuuka (HIP)"
    (cd "$root" && go build -tags hip -o "${outdir}/yuuka-hip" ./cmd/yuuka) ||
        die "the HIP build failed; the backend requires libyuuka_hip.so and -tags hip"
    yuuka_bin="${outdir}/yuuka-hip"
fi
[[ -x "$yuuka_bin" ]] || die "yuuka is not executable: $yuuka_bin"

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

echo "== environment"
echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "host=$(uname -sr)"
echo "device=$device_name ($expected_target), rocminfo_compute_units=$device_cus"
echo "power_dpm_force_performance_level=${power_level:-unknown}"
echo "rocm_version=${rocm_version:-unknown}"
echo "hip_version=${hip_version:-unknown}"
echo "model=$model"
echo "model_sha256=$model_hash (verified)"
echo "yuuka=$yuuka_bin"
echo "llama_bench=$llama_bench"
echo "llama_bench_version=$("$llama_bench" --version 2>&1 | sed -n 's/^version: //p' | head -n1)"
echo "cells=$cells warmup=$warmup reps=$reps"
echo

# run_cell ENGINE PROMPT DECODE writes the engine log and echoes one line of
# numbers: prefill t/s, decode t/s. Both engines report a median already.
run_yuuka_cell() {
    local prompt="$1" decode="$2" log="$3"
    "$yuuka_bin" bench --model "$model" --backend hip --chunk-size 512 \
        --prompt-tokens "$prompt" --decode-tokens "$decode" --kv-type f32 \
        --warmup "$warmup" --reps "$reps" >"$log" 2>&1 || {
        cat "$log" >&2
        return 1
    }
    local prefill decode_rate
    prefill="$(sed -n 's/.*prefill: p50=[^ ]*s p95=[^ ]*s (\([0-9.]*\) tok\/s p50).*/\1/p' "$log" | tail -n1)"
    decode_rate="$(sed -n 's/.*decode: p50=[^ ]*s p95=[^ ]*s (\([0-9.]*\) tok\/s p50).*/\1/p' "$log" | tail -n1)"
    [[ -n "$decode_rate" ]] || {
        cat "$log" >&2
        return 1
    }
    printf '%s %s\n' "${prefill:--}" "$decode_rate"
}

run_llama_cell() {
    local prompt="$1" decode="$2" log="$3"
    "$llama_bench" -m "$model" -ngl 99 -b 512 -ub 512 \
        -p "$prompt" -n "$decode" -r "$reps" -o md >"$log" 2>&1 || {
        cat "$log" >&2
        return 1
    }
    # Markdown rows: | ... | pp<N> | <rate> ± <std> |, and the same for tg.
    local prefill decode_rate
    prefill="$(awk -F'|' -v want="pp${prompt}" '
        NF >= 4 { test = $(NF-2); gsub(/ /, "", test); if (test == want) { rate = $(NF-1); gsub(/ /, "", rate); sub(/±.*/, "", rate); print rate } }' "$log" | tail -n1)"
    decode_rate="$(awk -F'|' -v want="tg${decode}" '
        NF >= 4 { test = $(NF-2); gsub(/ /, "", test); if (test == want) { rate = $(NF-1); gsub(/ /, "", rate); sub(/±.*/, "", rate); print rate } }' "$log" | tail -n1)"
    [[ -n "$decode_rate" ]] || {
        cat "$log" >&2
        return 1
    }
    printf '%s %s\n' "${prefill:--}" "$decode_rate"
}

echo "== token rate per cell (median, one process on the GPU)"
printf '%-16s %12s %12s %12s %12s %8s\n' "cell" "yuuka pp" "llama pp" \
    "yuuka tg" "llama tg" "tg ratio"
IFS=',' read -ra cell_list <<<"$cells"
comparison_rows=()
for cell in "${cell_list[@]}"; do
    [[ "$cell" =~ ^[0-9]+:[0-9]+$ ]] || die "cell '$cell' is not prompt:decode"
    prompt="${cell%%:*}"
    decode="${cell##*:}"
    read -r y_prefill y_decode < <(run_yuuka_cell "$prompt" "$decode" "${outdir}/yuuka-p${prompt}-d${decode}.log") ||
        die "yuuka bench failed for pp${prompt}/tg${decode}"
    read -r l_prefill l_decode < <(run_llama_cell "$prompt" "$decode" "${outdir}/llama-p${prompt}-d${decode}.log") ||
        die "llama-bench failed for pp${prompt}/tg${decode}"
    ratio="$(awk -v a="$y_decode" -v b="$l_decode" 'BEGIN { printf "%.2f", a / b }')"
    printf '%-16s %12s %12s %12s %12s %8s\n' "pp${prompt}/tg${decode}" \
        "$y_prefill" "$l_prefill" "$y_decode" "$l_decode" "$ratio"
    comparison_rows+=("pp${prompt}/tg${decode} $y_decode $l_decode $ratio")
done
echo

if ((run_trace)); then
    echo "== per-shape weight bandwidth inside one decode step"
    echo "   yuuka: TestGPUProfileDecode; llama.cpp: rocprofv3 trace of -p 0 -n 32 -r 2"
    profile_log="${outdir}/yuuka-profile.log"
    (cd "$root" && YUUKA_TEST_MODEL="$model" YUUKA_BUILD_DIR="${YUUKA_BUILD_DIR:-${root}/build}" \
        go test -tags hip ./engine/gpu -run TestGPUProfileDecode -count=1 -v) \
        >"$profile_log" 2>&1 || {
        cat "$profile_log" >&2
        die "TestGPUProfileDecode failed"
    }
    # rocprofv3 drops a .rocprofv3 scratch directory in the working directory.
    trace_dir="${outdir}/reference-trace"
    mkdir -p "$trace_dir"
    if ! (cd "$trace_dir" && rocprofv3 -d "$trace_dir" -o reference -f rocpd \
        --kernel-trace true -- "$llama_bench" -m "$model" -ngl 99 -b 512 -ub 512 \
        -p 0 -n 32 -r 2 >"${outdir}/reference-trace.log" 2>&1); then
        cat "${outdir}/reference-trace.log" >&2
        die "rocprofv3 reference trace failed"
    fi
    python3 "$reducer" --db "$trace_dir" --yuuka-table "$profile_log" ||
        die "the per-shape reduction failed"
    echo
fi

echo "== summary"
for row in "${comparison_rows[@]}"; do
    echo "cell_ratio $row"
done
if ((keep)); then
    echo "artifacts kept in $outdir"
else
    echo "artifacts removed (pass --output DIR to keep them)"
fi

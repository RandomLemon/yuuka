#!/usr/bin/env bash
# Sample gfx1151 power, temperature, and clock telemetry under sustained load.
#
# Usage: scripts/thermal_probe.sh [seconds] [command]
# The default command is the Phase 0 hardware probe. Samples are printed as CSV
# to stdout and summarized at the end.
set -euo pipefail

duration="${1:-15}"
load_command="${2:-./build/yuuka_hardware_probe}"

if ! [[ "$duration" =~ ^[0-9]+$ ]] || [ "$duration" -lt 1 ]; then
    echo "duration must be a positive integer" >&2
    exit 2
fi
if [ ! -x "${load_command%% *}" ]; then
    echo "load command is not executable: ${load_command%% *}" >&2
    exit 2
fi

gpu_device=""
for candidate in /sys/class/drm/card*/device; do
    if [ -d "$candidate/hwmon" ]; then
        gpu_device="$candidate"
        break
    fi
done
if [ -z "$gpu_device" ]; then
    echo "no DRM GPU device with hwmon telemetry found" >&2
    exit 2
fi
hwmon="$(ls -d "$gpu_device"/hwmon/hwmon* 2>/dev/null | head -1 || true)"
power_file="$hwmon/power1_input"
temp_file="$hwmon/temp1_input"
clock_file="$gpu_device/pp_dpm_sclk"

read_value() {
    if [ -r "$1" ]; then
        cat "$1"
    else
        printf 'NA'
    fi
}

current_clock_mhz() {
    if [ ! -r "$clock_file" ]; then
        printf 'NA'
        return
    fi
    awk '/\*/ { sub(/[Mm][Hh]z/, "", $2); print $2; exit }' "$clock_file"
}

printf 'timestamp_s,power_w,temp_c,sclk_mhz\n'
start="$(date +%s)"
sample_file="$(mktemp)"
trap 'rm -f "$sample_file"; kill "$loader" 2>/dev/null || true' EXIT

(
    while :; do
        $load_command >/dev/null 2>&1 || exit 0
    done
) &
loader=$!

while kill -0 "$loader" 2>/dev/null; do
    now="$(date +%s)"
    elapsed=$((now - start))
    if [ "$elapsed" -ge "$duration" ]; then
        break
    fi
    power_uw="$(read_value "$power_file")"
    temp_mc="$(read_value "$temp_file")"
    clock="$(current_clock_mhz)"
    power_w="$(awk -v v="$power_uw" 'BEGIN { if (v ~ /^[0-9]+$/) printf "%.2f", v / 1000000; else print "NA" }')"
    temp_c="$(awk -v v="$temp_mc" 'BEGIN { if (v ~ /^[0-9]+$/) printf "%.2f", v / 1000; else print "NA" }')"
    printf '%d,%s,%s,%s\n' "$elapsed" "$power_w" "$temp_c" "$clock" | tee -a "$sample_file"
    sleep 1
done

kill "$loader" 2>/dev/null || true
wait "$loader" 2>/dev/null || true
printf '\nsummary\n'
awk -F, 'NR > 1 && $2 != "NA" && $3 != "NA" {
    n++; p += $2; t += $3;
    if (n == 1 || $2 < pmin) pmin = $2;
    if (n == 1 || $2 > pmax) pmax = $2;
    if (n == 1 || $3 < tmin) tmin = $3;
    if (n == 1 || $3 > tmax) tmax = $3;
}
END {
    if (n == 0) { print "no numeric samples"; exit 0 }
    printf "samples=%d power_avg=%.2fW power_min=%.2fW power_max=%.2fW\n", n, p/n, pmin, pmax;
    printf "temp_avg=%.2fC temp_min=%.2fC temp_max=%.2fC\n", t/n, tmin, tmax;
}' "$sample_file"

#!/usr/bin/env bash
# Collect the Phase 0 hardware and toolchain facts for the target host.
# Output is plain text so it can be archived with a benchmark report.
set -euo pipefail

echo "== host =="
uname -a
if [ -r /etc/os-release ]; then
    . /etc/os-release
    echo "os=${PRETTY_NAME:-unknown}"
fi
echo

echo "== cpu / memory =="
lscpu | sed -n 's/^\(Model name\|Socket(s)\|Core(s) per socket\|Thread(s) per core\|CPU(s)\)[[:space:]]*: /\\1: /p'
free -h
echo

echo "== gpu =="
rocminfo 2>&1 | sed -n '/^Agent 2/,/^$/p' | sed -n '1,80p'
echo

echo "== kfd topology =="
if [ -r /sys/class/kfd/kfd/topology/nodes/1/properties ]; then
    grep -E 'gfx_target_version|simd_count|mem_banks|max_engine_clk' \
        /sys/class/kfd/kfd/topology/nodes/*/properties || true
fi
echo

echo "== tools =="
hipcc --version 2>&1 | sed -n '1,4p'
cmake --version | head -1
go version
git rev-parse --short HEAD 2>/dev/null || true
echo

echo "== kernel and module =="
uname -r
lsmod | grep -E '^amdgpu|^amdkfd' || true
echo

echo "== c ABI smoke test =="
if [ -x build/yuuka_hip_abi_test ]; then
    build/yuuka_hip_abi_test || status=$?
    if [ "${status:-0}" -eq 77 ]; then
        echo "skipped: no usable HIP device"
    fi
else
    echo "build/yuuka_hip_abi_test is absent; run cmake first"
fi

#!/usr/bin/env bash
# Verify that the tracked tree builds and tests from a clean export.
#
# Usage:
#   scripts/check_clean_build.sh          # Go/CPU tiers
#   scripts/check_clean_build.sh --hip    # also CMake, CTest, and HIP Go tests
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "clean export: $tmp"
git -C "$root" archive --format=tar HEAD | tar -x -C "$tmp"
cd "$tmp"

unformatted="$(gofmt -l $(git -C "$root" ls-files '*.go'))"
if [ -n "$unformatted" ]; then
    printf 'gofmt required for:\n%s\n' "$unformatted"
    exit 1
fi
go vet ./...
go build ./...
go test -race ./...

if [ "${1:-}" = "--hip" ]; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j"$(nproc)"
    ctest --test-dir build --output-on-failure
    go build -tags hip ./...
    go test -tags hip -count=1 ./backend/hip ./engine/gpu
fi

echo "clean build checks passed"

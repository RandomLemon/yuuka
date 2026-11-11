#include "yuuka_hip.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool check(yuuka_status status, const char * what) {
    if (status == YUUKA_OK) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", what, yuuka_last_error());
    return false;
}

} // namespace

int main() {
    if (yuuka_abi_version() != YUUKA_HIP_ABI_VERSION) {
        std::fprintf(stderr, "ABI version mismatch\n");
        return 1;
    }

    yuuka_device * device = nullptr;
    yuuka_status status = yuuka_device_open(0, &device);
    if (status != YUUKA_OK) {
        std::fprintf(stderr, "SKIP: no usable HIP device: %s\n", yuuka_last_error());
        return 77;
    }

    yuuka_device_info info{};
    if (!check(yuuka_device_info_get(device, &info), "device_info")) {
        return 1;
    }
    std::printf("device: %s (%s), CUs=%u, memory=%llu MiB, HIP=%s\n",
                info.name, info.gcn_arch, info.compute_units,
                static_cast<unsigned long long>(info.total_memory / (1024 * 1024)),
                info.hip_version);

    constexpr uint64_t n = 1024;
    const uint64_t bytes = n * sizeof(float);
    std::vector<float> host_a(n);
    std::vector<float> host_b(n);
    std::vector<float> host_out(n);
    for (uint64_t i = 0; i < n; ++i) {
        host_a[i] = static_cast<float>(i) * 0.5f;
        host_b[i] = static_cast<float>(i % 17) + 1.0f;
    }

    yuuka_buffer * a = nullptr;
    yuuka_buffer * b = nullptr;
    yuuka_buffer * out = nullptr;
    yuuka_stream * stream = nullptr;
    yuuka_event * event = nullptr;

    bool ok = check(yuuka_buffer_alloc(device, bytes, &a), "alloc a") &&
              check(yuuka_buffer_alloc(device, bytes, &b), "alloc b") &&
              check(yuuka_buffer_alloc(device, bytes, &out), "alloc out") &&
              check(yuuka_stream_create(device, &stream), "stream") &&
              check(yuuka_event_create(device, &event), "event") &&
              check(yuuka_memcpy_h2d(a, host_a.data(), bytes), "h2d a") &&
              check(yuuka_memcpy_h2d(b, host_b.data(), bytes), "h2d b") &&
              check(yuuka_add_f32(stream, out, a, b, n, event), "add") &&
              check(yuuka_event_wait(event), "event wait") &&
              check(yuuka_memcpy_d2h(host_out.data(), out, bytes), "d2h");

    if (ok) {
        for (uint64_t i = 0; i < n; ++i) {
            const float expected = host_a[i] + host_b[i];
            if (std::fabs(host_out[i] - expected) > 1e-5f) {
                std::fprintf(stderr, "mismatch at %llu: got %f expected %f\n",
                             static_cast<unsigned long long>(i), host_out[i], expected);
                ok = false;
                break;
            }
        }
    }

    if (!ok && std::strlen(yuuka_last_error()) != 0) {
        std::fprintf(stderr, "last error: %s\n", yuuka_last_error());
    }

    // Invalid input must be rejected rather than silently ignored.
    if (yuuka_buffer_alloc(device, 0, &a) != YUUKA_ERR_INVALID_ARGUMENT) {
        std::fprintf(stderr, "zero-byte allocation was not rejected\n");
        ok = false;
    }

    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_buffer_free(out);
    yuuka_buffer_free(b);
    yuuka_buffer_free(a);
    yuuka_device_close(device);
    return ok ? 0 : 1;
}

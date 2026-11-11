#include "yuuka_hip.h"

#include "yuuka_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>

struct yuuka_device {
    hipDeviceProp_t prop;
    int index;
};

struct yuuka_buffer {
    void * ptr;
    uint64_t bytes;
};

struct yuuka_stream {
    hipStream_t stream;
};

struct yuuka_event {
    hipEvent_t event;
};

namespace {

thread_local std::string g_last_error;

yuuka_status fail(yuuka_status status, const char * format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    g_last_error = buffer;
    return status;
}

const char * hip_error(hipError_t err) {
    return hipGetErrorString(err);
}

bool check_gemv_shapes(uint64_t n_in, uint32_t dtype) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
        case YUUKA_DTYPE_F16:
            return n_in != 0;
        case YUUKA_DTYPE_Q8_0:
            return n_in % 32 == 0;
        case YUUKA_DTYPE_Q4_K:
        case YUUKA_DTYPE_Q5_K:
        case YUUKA_DTYPE_Q6_K:
            return n_in % 256 == 0;
        default:
            return false;
    }
}

uint64_t row_bytes(uint64_t n_in, uint32_t dtype) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            return n_in * 4;
        case YUUKA_DTYPE_F16:
            return n_in * 2;
        case YUUKA_DTYPE_Q8_0:
            return (n_in / 32) * 34;
        case YUUKA_DTYPE_Q4_K:
            return (n_in / 256) * 144;
        case YUUKA_DTYPE_Q5_K:
            return (n_in / 256) * 176;
        case YUUKA_DTYPE_Q6_K:
            return (n_in / 256) * 210;
        default:
            return 0;
    }
}

template <typename Fn>
yuuka_status guard(Fn && fn) {
    try {
        return fn();
    } catch (const std::bad_alloc &) {
        return fail(YUUKA_ERR_INTERNAL, "host allocation failed");
    } catch (const std::exception & e) {
        return fail(YUUKA_ERR_INTERNAL, "unexpected exception: %s", e.what());
    } catch (...) {
        return fail(YUUKA_ERR_INTERNAL, "unexpected non-standard exception");
    }
}

} // namespace

extern "C" {

YUUKA_API uint32_t yuuka_abi_version(void) {
    return YUUKA_HIP_ABI_VERSION;
}

YUUKA_API const char * yuuka_last_error(void) {
    return g_last_error.c_str();
}

YUUKA_API yuuka_status yuuka_device_open(int32_t index, yuuka_device ** out) {
    return guard([&]() -> yuuka_status {
        if (out == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "device output handle is null");
        }
        *out = nullptr;
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipGetDeviceCount: %s", hip_error(err));
        }
        if (index < 0 || index >= count) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "device index %d out of range (count=%d)", index, count);
        }
        auto * dev = new yuuka_device{};
        err = hipGetDeviceProperties(&dev->prop, index);
        if (err != hipSuccess) {
            delete dev;
            return fail(YUUKA_ERR_DEVICE, "hipGetDeviceProperties(%d): %s",
                        index, hip_error(err));
        }
        dev->index = index;
        *out = dev;
        return YUUKA_OK;
    });
}

YUUKA_API void yuuka_device_close(yuuka_device * dev) {
    delete dev;
}

YUUKA_API yuuka_status yuuka_device_info_get(const yuuka_device * dev,
                                             yuuka_device_info * info) {
    return guard([&]() -> yuuka_status {
        if (dev == nullptr || info == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "device or info is null");
        }
        std::memset(info, 0, sizeof(*info));
        std::snprintf(info->name, sizeof(info->name), "%s", dev->prop.name);
        std::snprintf(info->gcn_arch, sizeof(info->gcn_arch), "%s", dev->prop.gcnArchName);
        int version = 0;
        if (hipRuntimeGetVersion(&version) == hipSuccess) {
            std::snprintf(info->hip_version, sizeof(info->hip_version), "%d.%d",
                          version / 10000000, (version / 100000) % 100);
        }
        info->total_memory = dev->prop.totalGlobalMem;
        info->compute_units = dev->prop.multiProcessorCount;
        info->clock_mhz = dev->prop.clockRate / 1000;
        info->pci_domain = dev->prop.pciDomainID;
        info->pci_bus = dev->prop.pciBusID;
        info->pci_device = dev->prop.pciDeviceID;
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_buffer_alloc(yuuka_device * dev, uint64_t bytes,
                                          yuuka_buffer ** out) {
    return guard([&]() -> yuuka_status {
        if (dev == nullptr || out == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "device or buffer handle is null");
        }
        *out = nullptr;
        if (bytes == 0) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "cannot allocate a zero-byte buffer");
        }
        hipError_t err = hipSetDevice(dev->index);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipSetDevice(%d): %s", dev->index, hip_error(err));
        }
        void * ptr = nullptr;
        err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_ALLOCATION, "hipMalloc(%llu): %s",
                        static_cast<unsigned long long>(bytes), hip_error(err));
        }
        *out = new yuuka_buffer{ptr, bytes};
        return YUUKA_OK;
    });
}

YUUKA_API void yuuka_buffer_free(yuuka_buffer * buf) {
    if (buf == nullptr) {
        return;
    }
    if (buf->ptr != nullptr) {
        hipError_t err = hipFree(buf->ptr);
        if (err != hipSuccess) {
            g_last_error = std::string("hipFree: ") + hip_error(err);
        }
    }
    delete buf;
}

YUUKA_API uint64_t yuuka_buffer_size(const yuuka_buffer * buf) {
    return buf == nullptr ? 0 : buf->bytes;
}

YUUKA_API void * yuuka_buffer_ptr(const yuuka_buffer * buf) {
    return buf == nullptr ? nullptr : buf->ptr;
}

YUUKA_API yuuka_status yuuka_stream_create(yuuka_device * dev, yuuka_stream ** out) {
    return guard([&]() -> yuuka_status {
        if (dev == nullptr || out == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "device or stream handle is null");
        }
        *out = nullptr;
        hipError_t err = hipSetDevice(dev->index);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipSetDevice(%d): %s", dev->index, hip_error(err));
        }
        hipStream_t stream = nullptr;
        err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipStreamCreateWithFlags: %s", hip_error(err));
        }
        *out = new yuuka_stream{stream};
        return YUUKA_OK;
    });
}

YUUKA_API void yuuka_stream_destroy(yuuka_stream * stream) {
    if (stream == nullptr) {
        return;
    }
    if (stream->stream != nullptr) {
        hipError_t err = hipStreamDestroy(stream->stream);
        if (err != hipSuccess) {
            g_last_error = std::string("hipStreamDestroy: ") + hip_error(err);
        }
    }
    delete stream;
}

YUUKA_API yuuka_status yuuka_stream_synchronize(yuuka_stream * stream) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "stream is null");
        }
        hipError_t err = hipStreamSynchronize(stream->stream);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipStreamSynchronize: %s", hip_error(err));
        }
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_event_create(yuuka_device * dev, yuuka_event ** out) {
    return guard([&]() -> yuuka_status {
        if (dev == nullptr || out == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "device or event handle is null");
        }
        *out = nullptr;
        hipError_t err = hipSetDevice(dev->index);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipSetDevice(%d): %s", dev->index, hip_error(err));
        }
        hipEvent_t event = nullptr;
        err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipEventCreateWithFlags: %s", hip_error(err));
        }
        *out = new yuuka_event{event};
        return YUUKA_OK;
    });
}

YUUKA_API void yuuka_event_destroy(yuuka_event * event) {
    if (event == nullptr) {
        return;
    }
    if (event->event != nullptr) {
        hipError_t err = hipEventDestroy(event->event);
        if (err != hipSuccess) {
            g_last_error = std::string("hipEventDestroy: ") + hip_error(err);
        }
    }
    delete event;
}

YUUKA_API yuuka_status yuuka_event_wait(yuuka_event * event) {
    return guard([&]() -> yuuka_status {
        if (event == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "event is null");
        }
        hipError_t err = hipEventSynchronize(event->event);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipEventSynchronize: %s", hip_error(err));
        }
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_memcpy_h2d(yuuka_buffer * dst, const void * src,
                                        uint64_t bytes) {
    return guard([&]() -> yuuka_status {
        if (dst == nullptr || src == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "destination or source is null");
        }
        if (bytes > dst->bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "host-to-device copy of %llu bytes exceeds buffer of %llu bytes",
                        static_cast<unsigned long long>(bytes),
                        static_cast<unsigned long long>(dst->bytes));
        }
        hipError_t err = hipMemcpy(dst->ptr, src, bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipMemcpy host-to-device: %s", hip_error(err));
        }
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_memcpy_d2h(void * dst, const yuuka_buffer * src,
                                        uint64_t bytes) {
    return guard([&]() -> yuuka_status {
        if (dst == nullptr || src == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "destination or source is null");
        }
        if (bytes > src->bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "device-to-host copy of %llu bytes exceeds buffer of %llu bytes",
                        static_cast<unsigned long long>(bytes),
                        static_cast<unsigned long long>(src->bytes));
        }
        hipError_t err = hipMemcpy(dst, src->ptr, bytes, hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipMemcpy device-to-host: %s", hip_error(err));
        }
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_memset(yuuka_stream * stream, yuuka_buffer * dst,
                                    int value, uint64_t bytes) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || dst == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "stream or destination is null");
        }
        if (bytes > dst->bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "memset of %llu bytes exceeds buffer of %llu bytes",
                        static_cast<unsigned long long>(bytes),
                        static_cast<unsigned long long>(dst->bytes));
        }
        hipError_t err = hipMemsetAsync(dst->ptr, value, bytes, stream->stream);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "hipMemsetAsync: %s", hip_error(err));
        }
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_add_f32(yuuka_stream * stream, yuuka_buffer * dst,
                                     const yuuka_buffer * a, const yuuka_buffer * b,
                                     uint64_t n, yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || dst == nullptr || a == nullptr || b == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "null stream or buffer");
        }
        const uint64_t bytes = n * sizeof(float);
        if (n == 0) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "element count is zero");
        }
        if (bytes / sizeof(float) != n) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "element count overflows");
        }
        if (dst->bytes < bytes || a->bytes < bytes || b->bytes < bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "add_f32 requires %llu bytes per buffer",
                        static_cast<unsigned long long>(bytes));
        }
        yuuka::launch_add_f32(static_cast<float *>(dst->ptr),
                              static_cast<const float *>(a->ptr),
                              static_cast<const float *>(b->ptr), n, stream->stream);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "add_f32 launch: %s", hip_error(err));
        }
        if (completion != nullptr) {
            err = hipEventRecord(completion->event, stream->stream);
            if (err != hipSuccess) {
                return fail(YUUKA_ERR_DEVICE, "hipEventRecord: %s", hip_error(err));
            }
        }
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_gemv(yuuka_stream * stream, yuuka_buffer * out,
                                  const yuuka_buffer * x, const yuuka_buffer * w,
                                  uint64_t n_in, uint64_t n_out, uint32_t dtype,
                                  yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || out == nullptr || x == nullptr || w == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "null stream or buffer");
        }
        if (!check_gemv_shapes(n_in, dtype)) {
            return fail(YUUKA_ERR_UNSUPPORTED,
                        "unsupported gemv dtype %u or input width %llu",
                        dtype, static_cast<unsigned long long>(n_in));
        }
        if (n_out == 0) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "output width is zero");
        }
        const uint64_t input_bytes = n_in * sizeof(float);
        const uint64_t output_bytes = n_out * sizeof(float);
        const uint64_t weight_row_bytes = row_bytes(n_in, dtype);
        if (w->bytes != weight_row_bytes * n_out) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "weight buffer has %llu bytes, expected %llu",
                        static_cast<unsigned long long>(w->bytes),
                        static_cast<unsigned long long>(weight_row_bytes * n_out));
        }
        if (x->bytes < input_bytes || out->bytes < output_bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "gemv vector buffer is too small");
        }
        const float * x_ptr = static_cast<const float *>(x->ptr);
        float * out_ptr = static_cast<float *>(out->ptr);
        switch (dtype) {
            case YUUKA_DTYPE_F32:
                yuuka::launch_gemv_f32(out_ptr, x_ptr,
                                       static_cast<const float *>(w->ptr),
                                       n_in, n_out, stream->stream);
                break;
            case YUUKA_DTYPE_F16:
                yuuka::launch_gemv_f16(out_ptr, x_ptr, w->ptr, n_in, n_out,
                                       stream->stream);
                break;
            case YUUKA_DTYPE_Q8_0:
                yuuka::launch_gemv_q8_0(out_ptr, x_ptr, w->ptr, n_in, n_out,
                                        stream->stream);
                break;
            case YUUKA_DTYPE_Q4_K:
                yuuka::launch_gemv_q4_k(out_ptr, x_ptr, w->ptr, n_in, n_out,
                                        stream->stream);
                break;
            case YUUKA_DTYPE_Q5_K:
                yuuka::launch_gemv_q5_k(out_ptr, x_ptr, w->ptr, n_in, n_out,
                                        stream->stream);
                break;
            case YUUKA_DTYPE_Q6_K:
                yuuka::launch_gemv_q6_k(out_ptr, x_ptr, w->ptr, n_in, n_out,
                                        stream->stream);
                break;
            default:
                return fail(YUUKA_ERR_UNSUPPORTED, "unsupported gemv dtype %u", dtype);
        }
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "gemv launch: %s", hip_error(err));
        }
        if (completion != nullptr) {
            err = hipEventRecord(completion->event, stream->stream);
            if (err != hipSuccess) {
                return fail(YUUKA_ERR_DEVICE, "hipEventRecord: %s", hip_error(err));
            }
        }
        return YUUKA_OK;
    });
}

} // extern "C"

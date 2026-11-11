#ifndef YUUKA_HIP_H
#define YUUKA_HIP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YUUKA_HIP_ABI_VERSION 1

#if defined(_WIN32)
#define YUUKA_API __declspec(dllexport)
#elif defined(__GNUC__)
#define YUUKA_API __attribute__((visibility("default")))
#else
#define YUUKA_API
#endif

/* Opaque handles. Ownership is documented per function. */
typedef struct yuuka_device yuuka_device;
typedef struct yuuka_buffer yuuka_buffer;
typedef struct yuuka_stream yuuka_stream;
typedef struct yuuka_event yuuka_event;

typedef enum yuuka_status {
    YUUKA_OK = 0,
    YUUKA_ERR_INVALID_ARGUMENT = 1,
    YUUKA_ERR_UNSUPPORTED = 2,
    YUUKA_ERR_DEVICE = 3,
    YUUKA_ERR_ALLOCATION = 4,
    YUUKA_ERR_INTERNAL = 5
} yuuka_status;

typedef struct yuuka_device_info {
    char name[256];
    char gcn_arch[64];
    char hip_version[64];
    uint64_t total_memory;
    uint32_t compute_units;
    uint32_t clock_mhz;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} yuuka_device_info;

/* Returns the ABI version compiled into the shared library. */
YUUKA_API uint32_t yuuka_abi_version(void);

/* Returns a thread-local message for the most recent failed call. */
YUUKA_API const char * yuuka_last_error(void);

/* Opens device index and returns a handle owned by the caller. */
YUUKA_API yuuka_status yuuka_device_open(int32_t index, yuuka_device ** out);
YUUKA_API void yuuka_device_close(yuuka_device * dev);
YUUKA_API yuuka_status yuuka_device_info_get(const yuuka_device * dev, yuuka_device_info * info);

/* Device allocations are owned by the caller and freed exactly once. */
YUUKA_API yuuka_status yuuka_buffer_alloc(yuuka_device * dev, uint64_t bytes, yuuka_buffer ** out);
YUUKA_API void yuuka_buffer_free(yuuka_buffer * buf);
YUUKA_API uint64_t yuuka_buffer_size(const yuuka_buffer * buf);
YUUKA_API void * yuuka_buffer_ptr(const yuuka_buffer * buf);

/* Explicit stream and event objects. Events are single-use completion signals. */
YUUKA_API yuuka_status yuuka_stream_create(yuuka_device * dev, yuuka_stream ** out);
YUUKA_API void yuuka_stream_destroy(yuuka_stream * stream);
YUUKA_API yuuka_status yuuka_stream_synchronize(yuuka_stream * stream);
YUUKA_API yuuka_status yuuka_event_create(yuuka_device * dev, yuuka_event ** out);
YUUKA_API void yuuka_event_destroy(yuuka_event * event);
YUUKA_API yuuka_status yuuka_event_wait(yuuka_event * event);

/* Host transfers are synchronous with respect to the host buffer. */
YUUKA_API yuuka_status yuuka_memcpy_h2d(yuuka_buffer * dst, const void * src, uint64_t bytes);
YUUKA_API yuuka_status yuuka_memcpy_d2h(void * dst, const yuuka_buffer * src, uint64_t bytes);
YUUKA_API yuuka_status yuuka_memset(yuuka_stream * stream, yuuka_buffer * dst, int value, uint64_t bytes);

/* Elementwise smoke operation used by the ABI test. */
YUUKA_API yuuka_status yuuka_add_f32(yuuka_stream * stream, yuuka_buffer * dst,
                           const yuuka_buffer * a, const yuuka_buffer * b,
                           uint64_t n, yuuka_event * completion);

/* Fused quantized matrix-vector product.
 *
 * out[n_out] = W[n_out, n_in] * x[n_in]
 * W uses the packed GGUF layout selected by dtype. Supported dtype values are
 * declared in yuuka_dtype. The operation is submitted on stream and signals
 * completion after the kernel. No host synchronization is performed.
 */
typedef enum yuuka_dtype {
    YUUKA_DTYPE_F32 = 0,
    YUUKA_DTYPE_F16 = 1,
    YUUKA_DTYPE_Q8_0 = 8,
    YUUKA_DTYPE_Q4_K = 12,
    YUUKA_DTYPE_Q5_K = 13,
    YUUKA_DTYPE_Q6_K = 14
} yuuka_dtype;

YUUKA_API yuuka_status yuuka_gemv(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint32_t dtype,
                        yuuka_event * completion);

#ifdef __cplusplus
}
#endif

#endif

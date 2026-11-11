#include "yuuka_kernels.h"

#include <rocblas/rocblas.h>

#include <mutex>

namespace {

// The baseline handle is process-lifetime state. The current executor submits
// on one stream from one thread, so a shared handle with an explicit stream is
// sufficient. Concurrent use would need one handle per stream.
rocblas_handle cached_handle() {
    static rocblas_handle handle = nullptr;
    static std::once_flag once;
    std::call_once(once, []() {
        if (rocblas_create_handle(&handle) != rocblas_status_success) {
            handle = nullptr;
        }
    });
    return handle;
}

} // namespace

namespace yuuka {

int launch_gemm_blas_f16(float * out, const void * x, const void * w,
                         uint64_t n_in, uint64_t n_out, uint64_t batch,
                         hipStream_t stream) {
    rocblas_handle handle = cached_handle();
    if (handle == nullptr) {
        return -1;
    }
    rocblas_status stream_status = rocblas_set_stream(handle, stream);
    if (stream_status != rocblas_status_success) {
        return static_cast<int>(stream_status);
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // out is row-major [batch, n_out], which is column-major [n_out, batch]
    // with leading dimension n_out. W is [n_out, n_in] row-major, so it is the
    // transposed operand of a column-major [n_in, n_out] matrix.
    rocblas_status status = rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(n_out), static_cast<rocblas_int>(batch),
        static_cast<rocblas_int>(n_in), &alpha,
        w, rocblas_datatype_f16_r, static_cast<rocblas_int>(n_in),
        x, rocblas_datatype_f16_r, static_cast<rocblas_int>(n_in), &beta,
        out, rocblas_datatype_f32_r, static_cast<rocblas_int>(n_out),
        out, rocblas_datatype_f32_r, static_cast<rocblas_int>(n_out),
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    return static_cast<int>(status);
}

} // namespace yuuka

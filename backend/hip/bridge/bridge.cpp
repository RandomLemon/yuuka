#include "yuuka_hip.h"

#include "yuuka_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

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
    // Pinned staging for the two-int32 dynamic parameter update. Reused after
    // the previous replay has completed; calls are serialized per stream.
    int32_t * dynamic_staging;
};

struct yuuka_event {
    hipEvent_t event;
};

struct yuuka_plan {
    std::vector<yuuka_op> ops;
};

struct yuuka_graph {
    hipGraph_t graph;
    hipGraphExec_t exec;
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
        case YUUKA_DTYPE_Q4_0:
        case YUUKA_DTYPE_Q5_0:
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

bool checked_row_bytes(uint64_t n_in, uint32_t dtype, uint64_t * out) {
    uint64_t blocks = 0;
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            return !__builtin_mul_overflow(n_in, uint64_t(4), out);
        case YUUKA_DTYPE_F16:
            return !__builtin_mul_overflow(n_in, uint64_t(2), out);
        case YUUKA_DTYPE_Q4_0:
            blocks = n_in / 32;
            return !__builtin_mul_overflow(blocks, uint64_t(18), out);
        case YUUKA_DTYPE_Q5_0:
            blocks = n_in / 32;
            return !__builtin_mul_overflow(blocks, uint64_t(22), out);
        case YUUKA_DTYPE_Q8_0:
            blocks = n_in / 32;
            return !__builtin_mul_overflow(blocks, uint64_t(34), out);
        case YUUKA_DTYPE_Q4_K:
            blocks = n_in / 256;
            return !__builtin_mul_overflow(blocks, uint64_t(144), out);
        case YUUKA_DTYPE_Q5_K:
            blocks = n_in / 256;
            return !__builtin_mul_overflow(blocks, uint64_t(176), out);
        case YUUKA_DTYPE_Q6_K:
            blocks = n_in / 256;
            return !__builtin_mul_overflow(blocks, uint64_t(210), out);
        default:
            return false;
    }
}

bool kv_dtype_ok(uint32_t dtype) {
    return dtype == YUUKA_DTYPE_F32 || dtype == YUUKA_DTYPE_F16 ||
           dtype == YUUKA_DTYPE_Q8_0;
}

bool kv_row_bytes(uint32_t dtype, uint64_t kv_dim, uint64_t * out) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            return !__builtin_mul_overflow(kv_dim, uint64_t(4), out);
        case YUUKA_DTYPE_F16:
            return !__builtin_mul_overflow(kv_dim, uint64_t(2), out);
        case YUUKA_DTYPE_Q8_0:
            return !__builtin_mul_overflow((kv_dim + 31) / 32, uint64_t(34), out);
        default:
            return false;
    }
}

bool kv_pool_bytes(uint32_t dtype, uint64_t blocks, uint64_t block_size,
                   uint64_t kv_dim, uint64_t * out) {
    uint64_t row_bytes = 0;
    uint64_t rows = 0;
    if (!kv_row_bytes(dtype, kv_dim, &row_bytes) ||
        __builtin_mul_overflow(blocks, block_size, &rows) ||
        __builtin_mul_overflow(rows, row_bytes, out)) {
        return false;
    }
    return true;
}

bool checked_float_bytes(uint64_t elements, uint64_t * out) {
    return !__builtin_mul_overflow(elements, uint64_t(sizeof(float)), out);
}

bool buffer_range_ok(const yuuka_buffer * buf, uint64_t offset, uint64_t bytes) {
    return buf != nullptr && offset <= buf->bytes && bytes <= buf->bytes - offset;
}

// validate_gemv_plan_op checks a GEMV or GEMV_BATCH operation before it enters
// the plan. Unsupported dtypes and block-misaligned widths are capability
// failures; invalid dimensions and undersized buffers are argument failures.
// The check mirrors the direct GEMV entry points so a plan cannot silently
// submit an operation that the direct API would reject.
yuuka_status validate_gemv_plan_op(const yuuka_op & op, bool batched) {
    for (int i = 0; i < 3; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv op buffer %d is null", i);
        }
    }
    const uint64_t n_in = op.n[0];
    const uint64_t n_out = op.n[1];
    const uint64_t batch = batched ? op.n[2] : 1;
    const uint32_t dtype = op.u[0];
    if (!check_gemv_shapes(n_in, dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED,
                    "unsupported gemv dtype %u or input width %llu",
                    dtype, static_cast<unsigned long long>(n_in));
    }
    if (n_out == 0 || batch == 0 || (batched && batch > 16)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT,
                    "invalid gemv output width %llu or batch %llu",
                    static_cast<unsigned long long>(n_out),
                    static_cast<unsigned long long>(batch));
    }
    uint64_t weight_bytes = 0;
    uint64_t weight_row_bytes = 0;
    uint64_t input_elements = 0;
    uint64_t output_elements = 0;
    if (!checked_row_bytes(n_in, dtype, &weight_row_bytes) ||
        __builtin_mul_overflow(weight_row_bytes, n_out, &weight_bytes) ||
        __builtin_mul_overflow(n_in, batch, &input_elements) ||
        __builtin_mul_overflow(n_out, batch, &output_elements)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv dimensions overflow");
    }
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (__builtin_mul_overflow(input_elements, uint64_t(sizeof(float)), &input_bytes) ||
        __builtin_mul_overflow(output_elements, uint64_t(sizeof(float)), &output_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], op.off[0], weight_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv weight buffer is too small");
    }
    if (!buffer_range_ok(op.buf[1], op.off[1], input_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv input buffer is too small");
    }
    if (!buffer_range_ok(op.buf[2], op.off[2], output_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv output buffer is too small");
    }
    return YUUKA_OK;
}

// validate_gemm_plan_op checks a prefill GEMM operation. Path AUTO and WMMA
// read FP32 activations and accept every supported weight dtype. Path BLAS is
// the rocBLAS FP16 baseline and therefore requires FP16 activations.
yuuka_status validate_gemm_plan_op(const yuuka_op & op) {
    for (int i = 0; i < 3; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm op buffer %d is null", i);
        }
    }
    const uint64_t n_in = op.n[0];
    const uint64_t n_out = op.n[1];
    const uint64_t batch = op.n[2];
    const uint32_t dtype = op.u[0];
    const uint32_t path = op.u[1];
    if (!check_gemv_shapes(n_in, dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED,
                    "unsupported gemm dtype %u or input width %llu",
                    dtype, static_cast<unsigned long long>(n_in));
    }
    if (path != YUUKA_GEMM_PATH_AUTO && path != YUUKA_GEMM_PATH_BLAS &&
        path != YUUKA_GEMM_PATH_WMMA) {
        return fail(YUUKA_ERR_UNSUPPORTED, "unknown gemm path %u", path);
    }
    if (path == YUUKA_GEMM_PATH_BLAS) {
        if (dtype != YUUKA_DTYPE_F16) {
            return fail(YUUKA_ERR_UNSUPPORTED,
                        "rocBLAS baseline requires F16 weights, got %u", dtype);
        }
        if (n_in > 0x7fffffffULL || n_out > 0x7fffffffULL || batch > 0x7fffffffULL) {
            return fail(YUUKA_ERR_UNSUPPORTED, "rocBLAS dimensions exceed int32");
        }
    }
    if (n_in == 0 || n_out == 0 || batch == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm dimensions must be positive");
    }
    uint64_t weight_bytes = 0;
    uint64_t weight_row_bytes = 0;
    uint64_t input_elements = 0;
    uint64_t output_elements = 0;
    if (!checked_row_bytes(n_in, dtype, &weight_row_bytes) ||
        __builtin_mul_overflow(weight_row_bytes, n_out, &weight_bytes) ||
        __builtin_mul_overflow(n_in, batch, &input_elements) ||
        __builtin_mul_overflow(n_out, batch, &output_elements)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm dimensions overflow");
    }
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    const uint64_t input_element_bytes =
        path == YUUKA_GEMM_PATH_BLAS ? uint64_t(2) : uint64_t(sizeof(float));
    if (__builtin_mul_overflow(input_elements, input_element_bytes, &input_bytes) ||
        __builtin_mul_overflow(output_elements, uint64_t(sizeof(float)), &output_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], op.off[0], weight_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm weight buffer is too small");
    }
    if (!buffer_range_ok(op.buf[1], op.off[1], input_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm input buffer is too small");
    }
    if (!buffer_range_ok(op.buf[2], op.off[2], output_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm output buffer is too small");
    }
    return YUUKA_OK;
}

// range_elements checks a [rows, width] region with an explicit row stride.
bool range_elements(const yuuka_buffer * buf, uint64_t offset, uint64_t rows,
                    uint64_t stride, uint64_t width) {
    if (buf == nullptr || rows == 0 || stride < width) {
        return false;
    }
    uint64_t elems = 0;
    if (__builtin_mul_overflow(rows - 1, stride, &elems) ||
        __builtin_add_overflow(elems, width, &elems)) {
        return false;
    }
    uint64_t bytes = 0;
    if (__builtin_mul_overflow(elems, uint64_t(sizeof(float)), &bytes)) {
        return false;
    }
    return buffer_range_ok(buf, offset, bytes);
}

yuuka_status validate_batch_norm_plan_op(const yuuka_op & op) {
    const uint64_t cols = op.n[0];
    const uint64_t rows = op.n[1];
    const uint64_t src_stride = op.n[2];
    const uint64_t dst_stride = op.n[3];
    if (cols == 0 || rows == 0 || src_stride < cols || dst_stride < cols) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid batch norm dimensions");
    }
    if (op.buf[0] == nullptr || op.buf[1] == nullptr || op.buf[2] == nullptr) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch norm buffer is null");
    }
    if (!range_elements(op.buf[0], op.off[0], rows, src_stride, cols) ||
        !range_elements(op.buf[2], op.off[2], rows, dst_stride, cols)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch norm buffer is too small");
    }
    uint64_t weight_bytes = 0;
    if (__builtin_mul_overflow(cols, uint64_t(sizeof(float)), &weight_bytes) ||
        !buffer_range_ok(op.buf[1], 0, weight_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch norm weight is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_batch_heads_plan_op(const yuuka_op & op) {
    const uint64_t size = op.n[0];
    const uint64_t heads = op.n[1];
    const uint64_t stride = op.n[2];
    const uint64_t rows = op.n[3];
    if (size == 0 || heads == 0 || rows == 0 || stride < size) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid head norm dimensions");
    }
    if (op.buf[0] == nullptr || op.buf[1] == nullptr || op.buf[2] == nullptr) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "head norm buffer is null");
    }
    if (!range_elements(op.buf[0], op.off[0], rows, heads * stride, heads * stride) ||
        !range_elements(op.buf[2], op.off[2], rows, heads * stride, heads * stride)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "head norm buffer is too small");
    }
    uint64_t weight_bytes = 0;
    if (__builtin_mul_overflow(size, uint64_t(sizeof(float)), &weight_bytes) ||
        !buffer_range_ok(op.buf[1], 0, weight_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "head norm weight is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_rope_batch_plan_op(const yuuka_op & op) {
    const uint64_t heads = op.n[0];
    const uint64_t stride = op.n[1];
    const uint64_t head_dim = op.n[2];
    const uint64_t rot_dim = op.n[3];
    const uint64_t rows = op.n[4];
    const uint64_t row_stride = op.n[5];
    if (heads == 0 || rows == 0 || rot_dim == 0 || rot_dim % 2 != 0 ||
        rot_dim > head_dim || stride < head_dim || row_stride < heads * stride) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid batch RoPE dimensions");
    }
    if (rot_dim / 2 > 1024) {
        return fail(YUUKA_ERR_UNSUPPORTED, "batch RoPE rotation dimension is too large");
    }
    if (op.buf[0] == nullptr) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch RoPE buffer is null");
    }
    if (!range_elements(op.buf[0], op.off[0], rows, row_stride, heads * stride)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch RoPE buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_kv_write_batch_plan_op(const yuuka_op & op) {
    const uint64_t kv_dim = op.n[0];
    const uint64_t rows = op.n[1];
    const uint64_t capacity = op.u[0];
    if (kv_dim == 0 || rows == 0 || capacity < rows) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid batch KV write dimensions");
    }
    for (int i = 0; i < 4; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch KV write buffer %d is null", i);
        }
    }
    if (!range_elements(op.buf[0], op.off[0], rows, kv_dim, kv_dim) ||
        !range_elements(op.buf[1], op.off[1], rows, kv_dim, kv_dim) ||
        !range_elements(op.buf[2], op.off[2], capacity, kv_dim, kv_dim) ||
        !range_elements(op.buf[3], op.off[3], capacity, kv_dim, kv_dim)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch KV write buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_attention_prefill_plan_op(const yuuka_op & op) {
    const uint64_t kv_heads = op.n[0];
    const uint64_t heads = op.n[1];
    const uint64_t head_dim = op.n[2];
    const uint64_t kv_dim = op.n[3];
    const uint64_t rows = op.n[4];
    const uint64_t capacity = op.u[0];
    if (kv_heads == 0 || heads == 0 || head_dim == 0 || rows == 0 ||
        capacity < rows || heads % kv_heads != 0 || kv_dim != kv_heads * head_dim) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid prefill attention dimensions");
    }
    if (head_dim > 256) {
        return fail(YUUKA_ERR_UNSUPPORTED, "prefill attention head dimension exceeds 256");
    }
    for (int i = 0; i < 4; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "prefill attention buffer %d is null", i);
        }
    }
    const uint64_t q_stride = heads * (op.u[3] != 0 ? head_dim : 2 * head_dim);
    const uint64_t o_stride = heads * head_dim;
    if (!range_elements(op.buf[0], op.off[0], rows, q_stride, q_stride) ||
        !range_elements(op.buf[3], op.off[3], rows, o_stride, o_stride) ||
        !range_elements(op.buf[1], 0, capacity, kv_dim, kv_dim) ||
        !range_elements(op.buf[2], 0, capacity, kv_dim, kv_dim)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "prefill attention buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_recurrent_chunk_plan_op(const yuuka_op & op) {
    const uint64_t rows = op.n[0];
    const uint64_t groups = op.n[1];
    const uint64_t heads = op.n[2];
    const uint64_t state = op.n[3];
    const uint64_t conv_dim = op.n[4];
    const uint64_t inner = op.n[5];
    const uint64_t kernel = op.u[0];
    const uint64_t expected_inner = heads * state;
    const uint64_t expected_conv = 2 * groups * state + expected_inner;
    if (rows == 0 || groups == 0 || heads == 0 || state == 0 || kernel < 2 ||
        inner != expected_inner || conv_dim != expected_conv ||
        (state & (state - 1)) != 0 || state > 1024) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid recurrent chunk dimensions");
    }
    for (int i = 0; i < 12; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "recurrent chunk buffer %d is null", i);
        }
    }
    uint64_t need[12] = {};
    uint64_t elems = 0;
    if (__builtin_mul_overflow(conv_dim, kernel - 1, &elems)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "recurrent chunk dimensions overflow");
    }
    need[0] = elems;
    if (__builtin_mul_overflow(heads, state * state, &elems)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "recurrent chunk dimensions overflow");
    }
    need[1] = elems;
    if (__builtin_mul_overflow(rows, conv_dim, &elems)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "recurrent chunk dimensions overflow");
    }
    need[2] = elems;
    need[11] = elems;
    if (__builtin_mul_overflow(rows, inner, &elems)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "recurrent chunk dimensions overflow");
    }
    need[3] = elems;
    need[10] = elems;
    if (__builtin_mul_overflow(rows, heads, &elems)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "recurrent chunk dimensions overflow");
    }
    need[4] = elems;
    need[5] = elems;
    if (__builtin_mul_overflow(conv_dim, kernel, &elems)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "recurrent chunk dimensions overflow");
    }
    need[6] = elems;
    need[7] = heads;
    need[8] = heads;
    need[9] = state;
    for (int i = 0; i < 12; ++i) {
        uint64_t bytes = 0;
        if (__builtin_mul_overflow(need[i], uint64_t(sizeof(float)), &bytes) ||
            !buffer_range_ok(op.buf[i], op.off[i], bytes)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "recurrent chunk buffer %d is too small", i);
        }
    }
    return YUUKA_OK;
}

bool embedding_dtype_supported(uint32_t dtype) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
        case YUUKA_DTYPE_F16:
        case YUUKA_DTYPE_Q4_0:
        case YUUKA_DTYPE_Q5_0:
        case YUUKA_DTYPE_Q8_0:
        case YUUKA_DTYPE_Q4_K:
        case YUUKA_DTYPE_Q5_K:
        case YUUKA_DTYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

yuuka_status validate_embedding_plan_op(const yuuka_op & op) {
    const uint64_t cols = op.n[0];
    const uint64_t row_bytes = op.n[1];
    const uint32_t dtype = op.u[0];
    if (cols == 0 || row_bytes == 0 || op.buf[0] == nullptr || op.buf[1] == nullptr) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid embedding arguments");
    }
    if (!embedding_dtype_supported(dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED, "unsupported embedding dtype %u", dtype);
    }
    uint64_t expected_row = 0;
    if (!checked_row_bytes(cols, dtype, &expected_row) || expected_row != row_bytes) {
        return fail(YUUKA_ERR_UNSUPPORTED,
                    "embedding dtype %u does not match row layout", dtype);
    }
    uint64_t out_bytes = 0;
    if (__builtin_mul_overflow(cols, uint64_t(sizeof(float)), &out_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "embedding output size overflows");
    }
    if (!buffer_range_ok(op.buf[1], op.off[1], out_bytes) ||
        !buffer_range_ok(op.buf[0], op.off[0], row_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "embedding buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_embedding_batch_plan_op(const yuuka_op & op) {
    const uint64_t cols = op.n[0];
    const uint64_t rows = op.n[1];
    const uint64_t row_bytes = op.n[2];
    const uint64_t vocab = op.n[3];
    const uint32_t dtype = op.u[0];
    if (cols == 0 || rows == 0 || vocab == 0 || op.buf[0] == nullptr ||
        op.buf[1] == nullptr || op.buf[2] == nullptr) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid batch embedding arguments");
    }
    if (!embedding_dtype_supported(dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED, "unsupported batch embedding dtype %u", dtype);
    }
    uint64_t expected_row = 0;
    if (!checked_row_bytes(cols, dtype, &expected_row) || expected_row != row_bytes) {
        return fail(YUUKA_ERR_UNSUPPORTED,
                    "batch embedding dtype %u does not match row layout", dtype);
    }
    uint64_t token_bytes = 0;
    uint64_t out_bytes = 0;
    if (__builtin_mul_overflow(rows, uint64_t(4), &token_bytes) ||
        __builtin_mul_overflow(rows, cols, &out_bytes) ||
        __builtin_mul_overflow(out_bytes, uint64_t(sizeof(float)), &out_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch embedding dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[1], op.off[1], out_bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], token_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "batch embedding buffer is too small");
    }
    uint64_t weight_bytes = 0;
    if (__builtin_mul_overflow(row_bytes, vocab, &weight_bytes) ||
        !buffer_range_ok(op.buf[0], 0, weight_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT,
                    "batch embedding weight buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_cast_plan_op(const yuuka_op & op) {
    if (op.buf[0] == nullptr || op.buf[1] == nullptr || op.n[0] == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid cast arguments");
    }
    uint64_t src_bytes = 0;
    uint64_t dst_bytes = 0;
    if (__builtin_mul_overflow(op.n[0], uint64_t(4), &src_bytes) ||
        __builtin_mul_overflow(op.n[0], uint64_t(2), &dst_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "cast dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], op.off[0], src_bytes) ||
        !buffer_range_ok(op.buf[1], op.off[1], dst_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "cast buffer is too small");
    }
    return YUUKA_OK;
}


uint64_t paged_blocks(uint64_t capacity, uint64_t block_size) {
    return (capacity + block_size - 1) / block_size;
}

yuuka_status validate_paged_pool(const yuuka_op & op, uint64_t kv_dim,
                                 uint64_t block_size, uint64_t capacity,
                                 int table_index) {
    if (block_size == 0 || capacity == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged op needs positive geometry");
    }
    const uint64_t blocks = paged_blocks(capacity, block_size);
    uint64_t rows = 0;
    uint64_t pool_elems = 0;
    uint64_t pool_bytes = 0;
    uint64_t table_bytes = 0;
    if (__builtin_mul_overflow(blocks, block_size, &rows) ||
        __builtin_mul_overflow(rows, kv_dim, &pool_elems) ||
        __builtin_mul_overflow(pool_elems, uint64_t(sizeof(float)), &pool_bytes) ||
        __builtin_mul_overflow(blocks, uint64_t(sizeof(int32_t)), &table_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged op dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[table_index], 0, table_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged block table is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_kv_write_paged_plan_op(const yuuka_op & op, bool batched) {
    const uint64_t kv_dim = op.n[0];
    const uint64_t rows = batched ? op.n[1] : 1;
    const uint64_t block_size = op.u[0];
    const uint64_t capacity = op.u[1];
    const uint32_t dtype = op.u[2];
    if (kv_dim == 0 || rows == 0 || capacity < rows) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid paged KV write dimensions");
    }
    if (!kv_dtype_ok(dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED, "unsupported paged KV dtype %u", dtype);
    }
    for (int i = 0; i < 5; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged KV write buffer %d is null", i);
        }
    }
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (__builtin_mul_overflow(rows, kv_dim, &elems) ||
        __builtin_mul_overflow(elems, uint64_t(sizeof(float)), &bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged KV write dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], op.off[0], bytes) ||
        !buffer_range_ok(op.buf[1], op.off[1], bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged KV write source is too small");
    }
    yuuka_status status = validate_paged_pool(op, kv_dim, block_size, capacity, 4);
    if (status != YUUKA_OK) {
        return status;
    }
    const uint64_t blocks = paged_blocks(capacity, block_size);
    uint64_t pool_bytes = 0;
    if (!kv_pool_bytes(dtype, blocks, block_size, kv_dim, &pool_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged KV pool dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[2], 0, pool_bytes) ||
        !buffer_range_ok(op.buf[3], 0, pool_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged KV pool is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_attention_paged_plan_op(const yuuka_op & op) {
    const uint64_t kv_heads = op.n[0];
    const uint64_t heads = op.n[1];
    const uint64_t head_dim = op.n[2];
    const uint64_t kv_dim = op.n[3];
    const uint64_t score_stride = op.n[4];
    const uint64_t block_size = op.u[0];
    const uint64_t capacity = op.u[1];
    const uint32_t dtype = op.u[2];
    if (kv_heads == 0 || heads == 0 || head_dim == 0 || heads % kv_heads != 0 ||
        kv_dim != kv_heads * head_dim || score_stride < capacity) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid paged attention dimensions");
    }
    if (!kv_dtype_ok(dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED, "unsupported paged attention KV dtype %u", dtype);
    }
    if (head_dim > 256) {
        return fail(YUUKA_ERR_UNSUPPORTED, "paged attention head dimension exceeds 256");
    }
    for (int i = 0; i < 6; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged attention buffer %d is null", i);
        }
    }
    yuuka_status status = validate_paged_pool(op, kv_dim, block_size, capacity, 5);
    if (status != YUUKA_OK) {
        return status;
    }
    const uint64_t q_elems = heads * (op.u[3] != 0 ? head_dim : 2 * head_dim);
    uint64_t q_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t score_bytes = 0;
    uint64_t blocks = paged_blocks(capacity, block_size);
    if (__builtin_mul_overflow(q_elems, uint64_t(sizeof(float)), &q_bytes) ||
        __builtin_mul_overflow(heads * head_dim, uint64_t(sizeof(float)), &out_bytes) ||
        __builtin_mul_overflow(heads, score_stride, &score_bytes) ||
        __builtin_mul_overflow(score_bytes, uint64_t(sizeof(float)), &score_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged attention dimensions overflow");
    }
    (void)blocks;
    if (!buffer_range_ok(op.buf[0], op.off[0], q_bytes) ||
        !buffer_range_ok(op.buf[3], op.off[3], out_bytes) ||
        !buffer_range_ok(op.buf[4], op.off[4], score_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged attention buffer is too small");
    }
    uint64_t pool_bytes = 0;
    if (!kv_pool_bytes(dtype, paged_blocks(capacity, block_size), block_size, kv_dim, &pool_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged attention pool dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[1], 0, pool_bytes) ||
        !buffer_range_ok(op.buf[2], 0, pool_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged attention pool is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_attention_prefill_paged_plan_op(const yuuka_op & op) {
    const uint64_t kv_heads = op.n[0];
    const uint64_t heads = op.n[1];
    const uint64_t head_dim = op.n[2];
    const uint64_t kv_dim = op.n[3];
    const uint64_t rows = op.n[4];
    const uint64_t block_size = op.u[0];
    const uint64_t capacity = op.u[1];
    const uint32_t dtype = op.u[2];
    if (kv_heads == 0 || heads == 0 || head_dim == 0 || rows == 0 ||
        heads % kv_heads != 0 || kv_dim != kv_heads * head_dim || capacity < rows) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid paged prefill attention dimensions");
    }
    if (!kv_dtype_ok(dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED, "unsupported paged prefill KV dtype %u", dtype);
    }
    if (head_dim > 256) {
        return fail(YUUKA_ERR_UNSUPPORTED, "paged prefill attention head dimension exceeds 256");
    }
    for (int i = 0; i < 5; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "paged prefill attention buffer %d is null", i);
        }
    }
    yuuka_status status = validate_paged_pool(op, kv_dim, block_size, capacity, 4);
    if (status != YUUKA_OK) {
        return status;
    }
    const uint64_t q_stride = heads * (op.u[3] != 0 ? head_dim : 2 * head_dim);
    const uint64_t o_stride = heads * head_dim;
    if (!range_elements(op.buf[0], op.off[0], rows, q_stride, q_stride) ||
        !range_elements(op.buf[3], op.off[3], rows, o_stride, o_stride)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged prefill attention buffer is too small");
    }
    uint64_t pool_bytes = 0;
    if (!kv_pool_bytes(dtype, paged_blocks(capacity, block_size), block_size, kv_dim, &pool_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged prefill pool dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[1], 0, pool_bytes) ||
        !buffer_range_ok(op.buf[2], 0, pool_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "paged prefill attention pool is too small");
    }
    return YUUKA_OK;
}

// Legacy single-token operators share the plan ABI with the chunked and paged
// families. Validate their dimensions and buffer ranges before append so an
// undersized plan cannot reach execution.
yuuka_status validate_rmsnorm_plan_op(const yuuka_op & op) {
    const uint64_t count = op.n[0];
    if (count == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid RMSNorm size");
    }
    for (int i = 0; i < 3; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "RMSNorm buffer %d is null", i);
        }
    }
    uint64_t bytes = 0;
    if (!checked_float_bytes(count, &bytes) ||
        !buffer_range_ok(op.buf[0], op.off[0], bytes) ||
        !buffer_range_ok(op.buf[1], 0, bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "RMSNorm buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_l2norm_plan_op(const yuuka_op & op) {
    const uint64_t heads = op.n[0];
    const uint64_t size = op.n[1];
    if (heads == 0 || size == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid L2Norm dimensions");
    }
    if (op.buf[0] == nullptr || op.buf[2] == nullptr) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "L2Norm buffer is null");
    }
    uint64_t elements = 0;
    uint64_t bytes = 0;
    if (__builtin_mul_overflow(heads, size, &elements) ||
        !checked_float_bytes(elements, &bytes) ||
        !buffer_range_ok(op.buf[0], op.off[0], bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "L2Norm buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_gated_rmsnorm_plan_op(const yuuka_op & op) {
    const uint64_t size = op.n[0];
    const uint64_t heads = op.n[1];
    if (size == 0 || heads == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid gated RMSNorm dimensions");
    }
    for (int i = 0; i < 4; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "gated RMSNorm buffer %d is null", i);
        }
    }
    uint64_t elements = 0;
    uint64_t bytes = 0;
    uint64_t weight_bytes = 0;
    if (__builtin_mul_overflow(heads, size, &elements) ||
        !checked_float_bytes(elements, &bytes) ||
        !checked_float_bytes(size, &weight_bytes) ||
        !buffer_range_ok(op.buf[0], op.off[0], bytes) ||
        !buffer_range_ok(op.buf[1], 0, weight_bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], bytes) ||
        !buffer_range_ok(op.buf[3], op.off[3], bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "gated RMSNorm buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_rmsnorm_heads_plan_op(const yuuka_op & op) {
    const uint64_t size = op.n[0];
    const uint64_t heads = op.n[1];
    const uint64_t stride = op.n[2];
    if (size == 0 || heads == 0 || stride < size) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid head RMSNorm dimensions");
    }
    for (int i = 0; i < 3; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "head RMSNorm buffer %d is null", i);
        }
    }
    uint64_t elements = 0;
    uint64_t bytes = 0;
    uint64_t weight_bytes = 0;
    if (__builtin_mul_overflow(heads, stride, &elements) ||
        !checked_float_bytes(elements, &bytes) ||
        !checked_float_bytes(size, &weight_bytes) ||
        !buffer_range_ok(op.buf[0], op.off[0], bytes) ||
        !buffer_range_ok(op.buf[1], 0, weight_bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "head RMSNorm buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_rope_plan_op(const yuuka_op & op) {
    const uint64_t heads = op.n[0];
    const uint64_t stride = op.n[1];
    const uint64_t head_dim = op.n[2];
    const uint64_t rot_dim = op.n[3];
    if (heads == 0 || rot_dim == 0 || rot_dim % 2 != 0 || rot_dim > head_dim ||
        stride < head_dim) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid RoPE dimensions");
    }
    if (rot_dim / 2 > 1024) {
        return fail(YUUKA_ERR_UNSUPPORTED, "RoPE rotation dimension is too large");
    }
    if (op.buf[0] == nullptr) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "RoPE buffer is null");
    }
    uint64_t elements = 0;
    uint64_t bytes = 0;
    if (__builtin_mul_overflow(heads, stride, &elements) ||
        !checked_float_bytes(elements, &bytes) ||
        !buffer_range_ok(op.buf[0], op.off[0], bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "RoPE buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_attention_plan_op(const yuuka_op & op) {
    const uint64_t kv_heads = op.n[0];
    const uint64_t heads = op.n[1];
    const uint64_t head_dim = op.n[2];
    const uint64_t kv_dim = op.n[3];
    const uint64_t capacity = op.u[0];
    if (kv_heads == 0 || heads == 0 || head_dim == 0 || kv_dim == 0 ||
        capacity == 0 || heads % kv_heads != 0 || kv_dim != kv_heads * head_dim) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid attention dimensions");
    }
    if (head_dim > 256) {
        return fail(YUUKA_ERR_UNSUPPORTED, "attention head dimension exceeds 256");
    }
    for (int i = 0; i < 5; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "attention buffer %d is null", i);
        }
    }
    uint64_t q_elements = 0;
    uint64_t out_elements = 0;
    uint64_t kv_elements = 0;
    uint64_t score_elements = 0;
    uint64_t q_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t score_bytes = 0;
    if (__builtin_mul_overflow(heads, 2 * head_dim, &q_elements) ||
        __builtin_mul_overflow(heads, head_dim, &out_elements) ||
        __builtin_mul_overflow(capacity, kv_dim, &kv_elements) ||
        __builtin_mul_overflow(heads, capacity, &score_elements) ||
        !checked_float_bytes(q_elements, &q_bytes) ||
        !checked_float_bytes(out_elements, &out_bytes) ||
        !checked_float_bytes(kv_elements, &kv_bytes) ||
        !checked_float_bytes(score_elements, &score_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "attention dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], op.off[0], q_bytes) ||
        !buffer_range_ok(op.buf[1], op.off[1], kv_bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], kv_bytes) ||
        !buffer_range_ok(op.buf[3], op.off[3], out_bytes) ||
        !buffer_range_ok(op.buf[4], op.off[4], score_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "attention buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_conv1d_plan_op(const yuuka_op & op) {
    const uint64_t channels = op.n[0];
    const uint64_t kernel = op.n[1];
    if (channels == 0 || kernel < 2) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid convolution dimensions");
    }
    for (int i = 0; i < 4; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "convolution buffer %d is null", i);
        }
    }
    uint64_t state_elements = 0;
    uint64_t weight_elements = 0;
    uint64_t state_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t channel_bytes = 0;
    if (__builtin_mul_overflow(channels, kernel - 1, &state_elements) ||
        __builtin_mul_overflow(channels, kernel, &weight_elements) ||
        !checked_float_bytes(state_elements, &state_bytes) ||
        !checked_float_bytes(weight_elements, &weight_bytes) ||
        !checked_float_bytes(channels, &channel_bytes) ||
        !buffer_range_ok(op.buf[0], op.off[0], state_bytes) ||
        !buffer_range_ok(op.buf[1], op.off[1], weight_bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], channel_bytes) ||
        !buffer_range_ok(op.buf[3], op.off[3], channel_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "convolution buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_gdn_plan_op(const yuuka_op & op) {
    const uint64_t groups = op.n[0];
    const uint64_t heads = op.n[1];
    const uint64_t state_size = op.n[2];
    if (groups == 0 || heads == 0 || state_size == 0 || groups > heads) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid GDN dimensions");
    }
    for (int i = 0; i < 9; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "GDN buffer %d is null", i);
        }
    }
    uint64_t state_elements = 0;
    uint64_t q_elements = 0;
    uint64_t value_elements = 0;
    uint64_t state_bytes = 0;
    uint64_t q_bytes = 0;
    uint64_t value_bytes = 0;
    uint64_t head_bytes = 0;
    if (__builtin_mul_overflow(heads, state_size, &state_elements) ||
        __builtin_mul_overflow(state_elements, state_size, &state_elements) ||
        __builtin_mul_overflow(groups, state_size, &q_elements) ||
        __builtin_mul_overflow(heads, state_size, &value_elements) ||
        !checked_float_bytes(state_elements, &state_bytes) ||
        !checked_float_bytes(q_elements, &q_bytes) ||
        !checked_float_bytes(value_elements, &value_bytes) ||
        !checked_float_bytes(heads, &head_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "GDN dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], op.off[0], state_bytes) ||
        !buffer_range_ok(op.buf[1], op.off[1], q_bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], q_bytes) ||
        !buffer_range_ok(op.buf[3], op.off[3], value_bytes) ||
        !buffer_range_ok(op.buf[4], op.off[4], head_bytes) ||
        !buffer_range_ok(op.buf[5], op.off[5], head_bytes) ||
        !buffer_range_ok(op.buf[6], op.off[6], head_bytes) ||
        !buffer_range_ok(op.buf[7], op.off[7], head_bytes) ||
        !buffer_range_ok(op.buf[8], op.off[8], value_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "GDN buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_elementwise_plan_op(const yuuka_op & op, int buffers) {
    const uint64_t count = op.n[0];
    if (count == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid elementwise size");
    }
    for (int i = 0; i < buffers; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "elementwise buffer %d is null", i);
        }
    }
    uint64_t bytes = 0;
    if (!checked_float_bytes(count, &bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "elementwise dimensions overflow");
    }
    for (int i = 0; i < buffers; ++i) {
        if (!buffer_range_ok(op.buf[i], op.off[i], bytes)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "elementwise buffer %d is too small", i);
        }
    }
    return YUUKA_OK;
}

yuuka_status validate_kv_write_plan_op(const yuuka_op & op) {
    const uint64_t kv_dim = op.n[0];
    if (kv_dim == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid KV write width");
    }
    for (int i = 0; i < 4; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "KV write buffer %d is null", i);
        }
    }
    uint64_t bytes = 0;
    if (!checked_float_bytes(kv_dim, &bytes) ||
        !buffer_range_ok(op.buf[0], op.off[0], bytes) ||
        !buffer_range_ok(op.buf[1], op.off[1], bytes) ||
        !buffer_range_ok(op.buf[2], op.off[2], bytes) ||
        !buffer_range_ok(op.buf[3], op.off[3], bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "KV write buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_moe_route_plan_op(const yuuka_op & op) {
    const uint64_t rows = op.n[0];
    const uint64_t experts = op.n[1];
    const uint64_t used = op.n[2];
    if (rows == 0 || experts == 0 || used == 0 || used > experts) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid MoE route dimensions");
    }
    for (int i = 0; i < 7; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE route buffer %d is null", i);
        }
    }
    uint64_t logits = 0;
    uint64_t row_used = 0;
    uint64_t buckets = 0;
    if (__builtin_mul_overflow(rows, experts, &logits) ||
        __builtin_mul_overflow(rows, used, &row_used) ||
        __builtin_mul_overflow(experts, row_used, &buckets)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE route dimensions overflow");
    }
    uint64_t logits_bytes = 0;
    uint64_t row_used_bytes = 0;
    uint64_t count_bytes = 0;
    uint64_t bucket_bytes = 0;
    if (__builtin_mul_overflow(logits, uint64_t(4), &logits_bytes) ||
        __builtin_mul_overflow(row_used, uint64_t(4), &row_used_bytes) ||
        __builtin_mul_overflow(experts, uint64_t(4), &count_bytes) ||
        __builtin_mul_overflow(buckets, uint64_t(4), &bucket_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE route dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], 0, logits_bytes) ||
        !buffer_range_ok(op.buf[1], 0, row_used_bytes) ||
        !buffer_range_ok(op.buf[2], 0, row_used_bytes) ||
        !buffer_range_ok(op.buf[3], 0, count_bytes) ||
        !buffer_range_ok(op.buf[4], 0, bucket_bytes) ||
        !buffer_range_ok(op.buf[5], 0, bucket_bytes) ||
        !buffer_range_ok(op.buf[6], 0, bucket_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE route buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_moe_gate_up_plan_op(const yuuka_op & op) {
    const uint64_t hidden = op.n[0];
    const uint64_t ffn = op.n[1];
    const uint64_t rows = op.n[2];
    const uint64_t experts = op.n[3];
    const uint64_t used = op.n[4];
    const uint32_t dtype = op.u[0];
    if (hidden == 0 || ffn == 0 || rows == 0 || experts == 0 || used == 0 ||
        used > experts || !check_gemv_shapes(hidden, dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED, "invalid MoE gate/up dimensions or dtype");
    }
    for (int i = 0; i < 8; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE gate/up buffer %d is null", i);
        }
    }
    uint64_t hidden_row = 0;
    uint64_t weight_rows = 0;
    uint64_t weight_bytes = 0;
    uint64_t norm_bytes = 0;
    uint64_t out_elems = 0;
    uint64_t out_bytes = 0;
    uint64_t buckets = 0;
    if (!checked_row_bytes(hidden, dtype, &hidden_row) ||
        __builtin_mul_overflow(ffn, experts, &weight_rows) ||
        __builtin_mul_overflow(weight_rows, hidden_row, &weight_bytes) ||
        __builtin_mul_overflow(rows, hidden, &norm_bytes) ||
        __builtin_mul_overflow(norm_bytes, uint64_t(4), &norm_bytes) ||
        __builtin_mul_overflow(rows * used, ffn, &out_elems) ||
        __builtin_mul_overflow(out_elems, uint64_t(4), &out_bytes) ||
        __builtin_mul_overflow(experts * rows * used, uint64_t(4), &buckets)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE gate/up dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], 0, norm_bytes) ||
        !buffer_range_ok(op.buf[1], 0, weight_bytes) ||
        !buffer_range_ok(op.buf[2], 0, weight_bytes) ||
        !buffer_range_ok(op.buf[3], 0, experts * 4) ||
        !buffer_range_ok(op.buf[4], 0, buckets) ||
        !buffer_range_ok(op.buf[5], 0, buckets) ||
        !buffer_range_ok(op.buf[6], 0, out_bytes) ||
        !buffer_range_ok(op.buf[7], 0, out_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE gate/up buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_moe_down_plan_op(const yuuka_op & op) {
    const uint64_t ffn = op.n[0];
    const uint64_t hidden = op.n[1];
    const uint64_t rows = op.n[2];
    const uint64_t experts = op.n[3];
    const uint64_t used = op.n[4];
    const uint32_t dtype = op.u[0];
    if (ffn == 0 || hidden == 0 || rows == 0 || experts == 0 || used == 0 ||
        used > experts || !check_gemv_shapes(ffn, dtype)) {
        return fail(YUUKA_ERR_UNSUPPORTED, "invalid MoE down dimensions or dtype");
    }
    for (int i = 0; i < 6; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE down buffer %d is null", i);
        }
    }
    uint64_t ffn_row = 0;
    uint64_t weight_rows = 0;
    uint64_t weight_bytes = 0;
    uint64_t act_elems = 0;
    uint64_t act_bytes = 0;
    uint64_t out_elems = 0;
    uint64_t out_bytes = 0;
    uint64_t buckets = 0;
    if (!checked_row_bytes(ffn, dtype, &ffn_row) ||
        __builtin_mul_overflow(hidden, experts, &weight_rows) ||
        __builtin_mul_overflow(weight_rows, ffn_row, &weight_bytes) ||
        __builtin_mul_overflow(rows * used, ffn, &act_elems) ||
        __builtin_mul_overflow(act_elems, uint64_t(4), &act_bytes) ||
        __builtin_mul_overflow(rows * used, hidden, &out_elems) ||
        __builtin_mul_overflow(out_elems, uint64_t(4), &out_bytes) ||
        __builtin_mul_overflow(experts * rows * used, uint64_t(4), &buckets)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE down dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], 0, act_bytes) ||
        !buffer_range_ok(op.buf[1], 0, weight_bytes) ||
        !buffer_range_ok(op.buf[2], 0, experts * 4) ||
        !buffer_range_ok(op.buf[3], 0, buckets) ||
        !buffer_range_ok(op.buf[4], 0, buckets) ||
        !buffer_range_ok(op.buf[5], 0, out_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE down buffer is too small");
    }
    return YUUKA_OK;
}

yuuka_status validate_moe_reduce_plan_op(const yuuka_op & op) {
    const uint64_t hidden = op.n[0];
    const uint64_t rows = op.n[1];
    const uint64_t used = op.n[2];
    if (hidden == 0 || rows == 0 || used == 0) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "invalid MoE reduce dimensions");
    }
    for (int i = 0; i < 3; ++i) {
        if (op.buf[i] == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE reduce buffer %d is null", i);
        }
    }
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t weight_bytes = 0;
    if (__builtin_mul_overflow(rows * hidden, uint64_t(4), &x_bytes) ||
        __builtin_mul_overflow(rows * used * hidden, uint64_t(4), &out_bytes) ||
        __builtin_mul_overflow(rows * used, uint64_t(4), &weight_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE reduce dimensions overflow");
    }
    if (!buffer_range_ok(op.buf[0], 0, x_bytes) ||
        !buffer_range_ok(op.buf[1], 0, out_bytes) ||
        !buffer_range_ok(op.buf[2], 0, weight_bytes)) {
        return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE reduce buffer is too small");
    }
    return YUUKA_OK;
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

template <typename T>
T * op_ptr(yuuka_op const & op, int index) {
    return reinterpret_cast<T *>(static_cast<uint8_t *>(op.buf[index]->ptr) + op.off[index]);
}

yuuka_status launch_gemv_batch_dispatch(yuuka_op const & op, hipStream_t stream) {
    auto * out = op_ptr<float>(op, 2);
    const auto * x = op_ptr<const float>(op, 1);
    const void * w = op_ptr<const void>(op, 0);
    const uint64_t n_in = op.n[0];
    const uint64_t n_out = op.n[1];
    const uint64_t batch = op.n[2];
    switch (op.u[0]) {
        case YUUKA_DTYPE_F32:
            yuuka::launch_gemv_batch_f32(out, x, static_cast<const float *>(w), n_in, n_out, batch, stream);
            break;
        case YUUKA_DTYPE_F16:
            yuuka::launch_gemv_batch_f16(out, x, w, n_in, n_out, batch, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            yuuka::launch_gemv_batch_q4_0(out, x, w, n_in, n_out, batch, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            yuuka::launch_gemv_batch_q5_0(out, x, w, n_in, n_out, batch, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            yuuka::launch_gemv_batch_q8_0(out, x, w, n_in, n_out, batch, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            yuuka::launch_gemv_batch_q4_k(out, x, w, n_in, n_out, batch, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            yuuka::launch_gemv_batch_q5_k(out, x, w, n_in, n_out, batch, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            yuuka::launch_gemv_batch_q6_k(out, x, w, n_in, n_out, batch, stream);
            break;
        default:
            return fail(YUUKA_ERR_UNSUPPORTED, "unsupported gemv dtype %u", op.u[0]);
    }
    return YUUKA_OK;
}

yuuka_status launch_gemm_dispatch(yuuka_op const & op, hipStream_t stream) {
    auto * out = op_ptr<float>(op, 2);
    const void * w = op_ptr<const void>(op, 0);
    const uint64_t n_in = op.n[0];
    const uint64_t n_out = op.n[1];
    const uint64_t batch = op.n[2];
    const uint32_t dtype = op.u[0];
    const uint32_t path = op.u[1];
    if (path == YUUKA_GEMM_PATH_BLAS) {
        const void * x16 = op_ptr<const void>(op, 1);
        const int status = yuuka::launch_gemm_blas_f16(out, x16, w, n_in, n_out,
                                                       batch, stream);
        if (status != 0) {
            return fail(YUUKA_ERR_DEVICE, "rocBLAS gemm failed with status %d", status);
        }
        return YUUKA_OK;
    }
    if (path == YUUKA_GEMM_PATH_WMMA ||
        (path == YUUKA_GEMM_PATH_AUTO && dtype != YUUKA_DTYPE_F32)) {
        yuuka::launch_gemm_wmma_f32out(out, op_ptr<const float>(op, 1), w,
                                       n_in, n_out, batch, dtype, stream);
        return YUUKA_OK;
    }
    if (dtype != YUUKA_DTYPE_F32) {
        return fail(YUUKA_ERR_UNSUPPORTED, "unsupported gemm dtype %u", dtype);
    }
    yuuka::launch_gemm_f32(out, op_ptr<const float>(op, 1),
                           static_cast<const float *>(w), n_in, n_out, batch, stream);
    return YUUKA_OK;
}

yuuka_status launch_gemv_dispatch(yuuka_op const & op, hipStream_t stream) {
    auto * out = op_ptr<float>(op, 2);
    const auto * x = op_ptr<const float>(op, 1);
    const void * w = op_ptr<const void>(op, 0);
    const uint64_t n_in = op.n[0];
    const uint64_t n_out = op.n[1];
    switch (op.u[0]) {
        case YUUKA_DTYPE_F32:
            yuuka::launch_gemv_f32(out, x, static_cast<const float *>(w), n_in, n_out, stream);
            break;
        case YUUKA_DTYPE_F16:
            yuuka::launch_gemv_f16(out, x, w, n_in, n_out, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            yuuka::launch_gemv_q4_0(out, x, w, n_in, n_out, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            yuuka::launch_gemv_q5_0(out, x, w, n_in, n_out, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            yuuka::launch_gemv_q8_0(out, x, w, n_in, n_out, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            yuuka::launch_gemv_q4_k(out, x, w, n_in, n_out, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            yuuka::launch_gemv_q5_k(out, x, w, n_in, n_out, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            yuuka::launch_gemv_q6_k(out, x, w, n_in, n_out, stream);
            break;
        default:
            return fail(YUUKA_ERR_UNSUPPORTED, "unsupported gemv dtype %u", op.u[0]);
    }
    return YUUKA_OK;
}

yuuka_status execute_op(yuuka_op const & op, int32_t token, uint32_t position,
                        hipStream_t stream, const int32_t * dynamic = nullptr) {
    switch (op.code) {
        case YUUKA_OP_GEMV: {
            yuuka_status status = launch_gemv_dispatch(op, stream);
            if (status != YUUKA_OK) {
                return status;
            }
            break;
        }
        case YUUKA_OP_GEMM: {
            yuuka_status status = launch_gemm_dispatch(op, stream);
            if (status != YUUKA_OK) {
                return status;
            }
            break;
        }
        case YUUKA_OP_GEMV_BATCH: {
            yuuka_status status = launch_gemv_batch_dispatch(op, stream);
            if (status != YUUKA_OK) {
                return status;
            }
            break;
        }
        case YUUKA_OP_RMSNORM_BATCH: {
            yuuka::launch_rmsnorm_batch_f32(
                op_ptr<float>(op, 2), op_ptr<const float>(op, 0),
                op_ptr<const float>(op, 1), int(op.n[1]), int(op.n[0]),
                int(op.n[2]), int(op.n[3]), op.f[0], stream);
            break;
        }
        case YUUKA_OP_RMSNORM_HEADS_BATCH: {
            yuuka::launch_rmsnorm_heads_batch_f32(
                op_ptr<float>(op, 2), op_ptr<const float>(op, 0),
                op_ptr<const float>(op, 1), int(op.n[3]), int(op.n[1]),
                int(op.n[0]), int(op.n[2]), op.f[0], stream);
            break;
        }
        case YUUKA_OP_ROPE_BATCH: {
            yuuka::launch_rope_batch_f32(
                op_ptr<float>(op, 0), int(op.n[4]), int(op.n[0]), int(op.n[1]),
                int(op.n[2]), int(op.n[3]), int(position), op.f[0], int(op.n[5]),
                stream);
            break;
        }
        case YUUKA_OP_KV_WRITE_BATCH: {
            const uint64_t rows = op.n[1];
            const uint64_t capacity = op.u[0];
            if (uint64_t(position) + rows > capacity) {
                return fail(YUUKA_ERR_INVALID_ARGUMENT,
                            "batch KV write at %u with %llu rows exceeds capacity %llu",
                            position, static_cast<unsigned long long>(rows),
                            static_cast<unsigned long long>(capacity));
            }
            yuuka::launch_kv_write_batch_f32(
                op_ptr<float>(op, 2), op_ptr<float>(op, 3),
                op_ptr<const float>(op, 0), op_ptr<const float>(op, 1),
                int(rows), int(op.n[0]), position, stream);
            break;
        }
        case YUUKA_OP_ATTENTION_PREFILL: {
            const uint64_t rows = op.n[4];
            const uint64_t capacity = op.u[0];
            if (uint64_t(position) + rows > capacity) {
                return fail(YUUKA_ERR_INVALID_ARGUMENT,
                            "prefill attention at %u with %llu rows exceeds capacity %llu",
                            position, static_cast<unsigned long long>(rows),
                            static_cast<unsigned long long>(capacity));
            }
            yuuka::launch_attention_prefill_f32(
                op_ptr<float>(op, 3), op_ptr<const float>(op, 0),
                op_ptr<const float>(op, 1), op_ptr<const float>(op, 2),
                int(op.n[0]), int(op.n[1]), int(op.n[2]), int(op.n[3]),
                int(rows), int(position), int(capacity), op.f[0], stream);
            break;
        }
        case YUUKA_OP_RECURRENT_CHUNK: {
            yuuka::launch_recurrent_chunk_f32(
                op_ptr<float>(op, 0), op_ptr<float>(op, 1),
                op_ptr<const float>(op, 2), op_ptr<const float>(op, 3),
                op_ptr<const float>(op, 4), op_ptr<const float>(op, 5),
                op_ptr<const float>(op, 6), op_ptr<const float>(op, 7),
                op_ptr<const float>(op, 8), op_ptr<const float>(op, 9),
                op_ptr<float>(op, 11), op_ptr<float>(op, 10),
                int(op.n[0]), int(op.n[1]), int(op.n[2]), int(op.n[3]),
                int(op.n[4]), int(op.n[5]), int(op.u[0]), op.f[0], op.f[1],
                stream);
            break;
        }
        case YUUKA_OP_EMBEDDING_BATCH: {
            yuuka::launch_embedding_batch_f32(
                op_ptr<float>(op, 1), op_ptr<const void>(op, 0),
                op_ptr<const int32_t>(op, 2), int(op.n[1]), op.n[2],
                int64_t(op.n[0]), op.u[0], stream);
            break;
        }
        case YUUKA_OP_CAST_F16: {
            yuuka::launch_cast_f32_to_f16(
                op_ptr<void>(op, 1), op_ptr<const float>(op, 0), op.n[0], stream);
            break;
        }
        case YUUKA_OP_RMSNORM: {
            auto * dst = op_ptr<float>(op, 2);
            const auto * src = op_ptr<const float>(op, 0);
            const auto * weight = op_ptr<const float>(op, 1);
            yuuka::launch_rmsnorm_f32(dst, src, weight, int64_t(op.n[0]), op.f[0], stream);
            break;
        }
        case YUUKA_OP_L2NORM: {
            yuuka::launch_l2norm_heads_f32(
                op_ptr<float>(op, 2), op_ptr<const float>(op, 0),
                int(op.n[0]), int64_t(op.n[1]), int64_t(op.n[2]),
                op.f[0], op.f[1], stream);
            break;
        }
        case YUUKA_OP_GATED_RMSNORM: {
            yuuka::launch_gated_rmsnorm_heads_f32(
                op_ptr<float>(op, 3), op_ptr<const float>(op, 0),
                op_ptr<const float>(op, 1), op_ptr<const float>(op, 2),
                int(op.n[1]), int64_t(op.n[0]), int64_t(op.n[2]),
                op.f[0], stream);
            break;
        }
        case YUUKA_OP_RMSNORM_HEADS: {
            yuuka::launch_rmsnorm_heads_f32(
                op_ptr<float>(op, 2), op_ptr<const float>(op, 0),
                op_ptr<const float>(op, 1), int(op.n[1]), int64_t(op.n[0]),
                int64_t(op.n[2]), op.f[0], stream);
            break;
        }
        case YUUKA_OP_ROPE:
            if (dynamic != nullptr) {
                yuuka::launch_rope_dynamic_f32(op_ptr<float>(op, 0),
                                               int(op.n[0]), int(op.n[1]), int(op.n[2]),
                                               int(op.n[3]), dynamic, op.f[0], stream);
            } else {
                yuuka::launch_rope_f32(op_ptr<float>(op, 0),
                                       int(op.n[0]), int(op.n[1]), int(op.n[2]), int(op.n[3]),
                                       int(position), op.f[0], stream);
            }
            break;
        case YUUKA_OP_ATTENTION:
            if (dynamic != nullptr) {
                return fail(YUUKA_ERR_UNSUPPORTED,
                            "legacy attention cannot capture dynamic parameters");
            }
            yuuka::launch_attention_f32(
                op_ptr<float>(op, 3),
                op_ptr<const float>(op, 0),
                op_ptr<const float>(op, 1),
                op_ptr<const float>(op, 2),
                op_ptr<float>(op, 4),
                int(op.n[0]), int(op.n[1]), int(op.n[2]), int(op.n[3]),
                int(position), int(op.u[0]), op.f[0], stream);
            break;
        case YUUKA_OP_CONV1D:
            yuuka::launch_conv1d_f32(op_ptr<float>(op, 0),
                                     op_ptr<const float>(op, 1),
                                     op_ptr<const float>(op, 2),
                                     op_ptr<float>(op, 3),
                                     int(op.n[0]), int(op.n[1]), stream);
            break;
        case YUUKA_OP_GDN:
            yuuka::launch_gdn_f32(
                op_ptr<float>(op, 0),
                op_ptr<const float>(op, 1),
                op_ptr<const float>(op, 2),
                op_ptr<const float>(op, 3),
                op_ptr<const float>(op, 4),
                op_ptr<const float>(op, 5),
                op_ptr<const float>(op, 6),
                op_ptr<const float>(op, 7),
                op_ptr<float>(op, 8),
                int(op.n[0]), int(op.n[1]), int(op.n[2]), stream);
            break;
        case YUUKA_OP_SILU_MUL:
            yuuka::launch_silu_mul_f32(op_ptr<float>(op, 2),
                                       op_ptr<const float>(op, 0),
                                       op_ptr<const float>(op, 1),
                                       int64_t(op.n[0]), stream);
            break;
        case YUUKA_OP_ADD:
            yuuka::launch_add_f32_buffers(op_ptr<float>(op, 0),
                                          op_ptr<const float>(op, 1),
                                          int64_t(op.n[0]), stream);
            break;
        case YUUKA_OP_COPY:
            yuuka::launch_copy_f32(op_ptr<float>(op, 0),
                                   op_ptr<const float>(op, 1),
                                   int64_t(op.n[0]), stream);
            break;
        case YUUKA_OP_SCALE:
            yuuka::launch_scale_f32(op_ptr<float>(op, 0),
                                    int64_t(op.n[0]), op.f[0], stream);
            break;
        case YUUKA_OP_EMBEDDING:
            if (dynamic != nullptr) {
                yuuka::launch_embedding_dynamic_f32(op_ptr<float>(op, 1),
                                                    op_ptr<const void>(op, 0), dynamic,
                                                    op.n[1], int64_t(op.n[0]), op.u[0], stream);
            } else {
                yuuka::launch_embedding_f32(op_ptr<float>(op, 1),
                                            op_ptr<const void>(op, 0), token, op.n[1],
                                            int64_t(op.n[0]), op.u[0], stream);
            }
            break;
        case YUUKA_OP_KV_WRITE: {
            if (dynamic != nullptr) {
                return fail(YUUKA_ERR_UNSUPPORTED,
                            "legacy KV write cannot capture dynamic parameters");
            }
            const uint64_t bytes = op.n[0] * sizeof(float);
            const uint64_t cache_offset = uint64_t(position) * bytes;
            hipError_t err = hipMemcpyAsync(
                static_cast<uint8_t *>(op.buf[2]->ptr) + op.off[2] + cache_offset,
                op_ptr<const void>(op, 0), bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return fail(YUUKA_ERR_DEVICE, "kv write k: %s", hip_error(err));
            }
            err = hipMemcpyAsync(
                static_cast<uint8_t *>(op.buf[3]->ptr) + op.off[3] + cache_offset,
                op_ptr<const void>(op, 1), bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return fail(YUUKA_ERR_DEVICE, "kv write v: %s", hip_error(err));
            }
            break;
        }
        case YUUKA_OP_KV_WRITE_PAGED:
        case YUUKA_OP_KV_WRITE_BATCH_PAGED: {
            const uint64_t rows = op.code == YUUKA_OP_KV_WRITE_PAGED ? 1 : op.n[1];
            const uint64_t capacity = op.u[1];
            if (uint64_t(position) + rows > capacity) {
                return fail(YUUKA_ERR_INVALID_ARGUMENT,
                            "paged KV write at %u with %llu rows exceeds capacity %llu",
                            position, static_cast<unsigned long long>(rows),
                            static_cast<unsigned long long>(capacity));
            }
            if (dynamic != nullptr) {
                yuuka::launch_kv_write_paged_dynamic_f32(
                    op_ptr<void>(op, 2), op_ptr<void>(op, 3),
                    op_ptr<const float>(op, 0), op_ptr<const float>(op, 1),
                    int(rows), int(op.n[0]), op_ptr<const int32_t>(op, 4),
                    dynamic, int(op.u[0]), op.u[2], stream);
            } else {
                yuuka::launch_kv_write_paged_f32(
                    op_ptr<void>(op, 2), op_ptr<void>(op, 3),
                    op_ptr<const float>(op, 0), op_ptr<const float>(op, 1),
                    int(rows), int(op.n[0]), op_ptr<const int32_t>(op, 4),
                    int(op.u[0]), int(position), op.u[2], stream);
            }
            break;
        }
        case YUUKA_OP_ATTENTION_PAGED: {
            const uint64_t capacity = op.u[1];
            if (uint64_t(position) >= capacity) {
                return fail(YUUKA_ERR_INVALID_ARGUMENT,
                            "paged attention position %u exceeds capacity %llu",
                            position, static_cast<unsigned long long>(capacity));
            }
            if (dynamic != nullptr) {
                yuuka::launch_attention_paged_dynamic_f32(
                    op_ptr<float>(op, 3), op_ptr<const float>(op, 0),
                    op_ptr<const void>(op, 1), op_ptr<const void>(op, 2),
                    op_ptr<float>(op, 4), op_ptr<const int32_t>(op, 5), dynamic,
                    int(op.u[0]), int(op.n[0]), int(op.n[1]), int(op.n[2]),
                    int(op.n[3]), int(op.n[4]), op.f[0], op.u[2],
                    int(op.u[3]), stream);
            } else {
                yuuka::launch_attention_paged_f32(
                    op_ptr<float>(op, 3), op_ptr<const float>(op, 0),
                    op_ptr<const void>(op, 1), op_ptr<const void>(op, 2),
                    op_ptr<float>(op, 4), op_ptr<const int32_t>(op, 5),
                    int(op.u[0]), int(op.n[0]), int(op.n[1]), int(op.n[2]),
                    int(op.n[3]), int(position), int(op.n[4]), op.f[0],
                    op.u[2], int(op.u[3]), stream);
            }
            break;
        }
        case YUUKA_OP_ATTENTION_PREFILL_PAGED: {
            const uint64_t rows = op.n[4];
            const uint64_t capacity = op.u[1];
            if (uint64_t(position) + rows > capacity) {
                return fail(YUUKA_ERR_INVALID_ARGUMENT,
                            "paged prefill attention at %u with %llu rows exceeds capacity %llu",
                            position, static_cast<unsigned long long>(rows),
                            static_cast<unsigned long long>(capacity));
            }
            yuuka::launch_attention_prefill_paged_f32(
                op_ptr<float>(op, 3), op_ptr<const float>(op, 0),
                op_ptr<const void>(op, 1), op_ptr<const void>(op, 2),
                op_ptr<const int32_t>(op, 4), int(op.u[0]), int(op.n[0]),
                int(op.n[1]), int(op.n[2]), int(op.n[3]), int(rows),
                int(position), op.f[0], op.u[2], int(op.u[3]), stream);
            break;
        }
        case YUUKA_OP_MOE_ROUTE: {
            const int rows = int(op.n[0]);
            const int experts = int(op.n[1]);
            const int used = int(op.n[2]);
            const yuuka_status route_status = static_cast<yuuka_status>(
                yuuka::launch_moe_route(
                    op_ptr<const float>(op, 0), rows, experts, used, rows * used,
                    op_ptr<int32_t>(op, 1), op_ptr<float>(op, 2),
                    op_ptr<int32_t>(op, 3), op_ptr<int32_t>(op, 4),
                    op_ptr<int32_t>(op, 5), op_ptr<float>(op, 6), stream));
            if (route_status != YUUKA_OK) {
                return fail(route_status, "MoE route launch failed");
            }
            break;
        }
        case YUUKA_OP_MOE_GATE_UP: {
            const int hidden = int(op.n[0]);
            const int ffn = int(op.n[1]);
            const int rows = int(op.n[2]);
            const int experts = int(op.n[3]);
            const int used = int(op.n[4]);
            uint64_t hidden_row_bytes = 0;
            if (!checked_row_bytes(uint64_t(hidden), op.u[0], &hidden_row_bytes)) {
                return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE gate/up row size overflow");
            }
            yuuka::launch_moe_gate_up_grouped(
                op_ptr<const float>(op, 0), op_ptr<const void>(op, 1),
                op_ptr<const void>(op, 2), op_ptr<const int32_t>(op, 3),
                op_ptr<const int32_t>(op, 4), op_ptr<const int32_t>(op, 5),
                op_ptr<float>(op, 6), op_ptr<float>(op, 7), rows, experts,
                used, hidden, ffn, hidden_row_bytes, op.u[0], stream);
            break;
        }
        case YUUKA_OP_MOE_DOWN: {
            const int ffn = int(op.n[0]);
            const int hidden = int(op.n[1]);
            const int rows = int(op.n[2]);
            const int experts = int(op.n[3]);
            const int used = int(op.n[4]);
            uint64_t ffn_row_bytes = 0;
            if (!checked_row_bytes(uint64_t(ffn), op.u[0], &ffn_row_bytes)) {
                return fail(YUUKA_ERR_INVALID_ARGUMENT, "MoE down row size overflow");
            }
            yuuka::launch_moe_down_grouped(
                op_ptr<const float>(op, 0), op_ptr<const void>(op, 1),
                op_ptr<const int32_t>(op, 2), op_ptr<const int32_t>(op, 3),
                op_ptr<const int32_t>(op, 4), op_ptr<float>(op, 5), rows,
                experts, used, hidden, ffn, ffn_row_bytes, op.u[0], stream);
            break;
        }
        case YUUKA_OP_MOE_REDUCE_ADD: {
            yuuka::launch_moe_reduce_add(
                op_ptr<float>(op, 0), op_ptr<const float>(op, 1),
                op_ptr<const float>(op, 2), int(op.n[1]), int(op.n[2]),
                int(op.n[0]), stream);
            break;
        }
        default:
            return fail(YUUKA_ERR_UNSUPPORTED, "unknown plan op %u", op.code);
    }
    return YUUKA_OK;
}


bool op_is_graph_capturable(const yuuka_op & op) {
    switch (op.code) {
        // Position-dependent ops with a device-parameter kernel.
        case YUUKA_OP_EMBEDDING:
        case YUUKA_OP_ROPE:
        case YUUKA_OP_KV_WRITE_PAGED:
        case YUUKA_OP_KV_WRITE_BATCH_PAGED:
        case YUUKA_OP_ATTENTION_PAGED:
            return true;
        // Position-dependent ops without a device-parameter kernel. Capturing
        // any of these would pin the host token or position used at capture.
        case YUUKA_OP_ROPE_BATCH:
        case YUUKA_OP_KV_WRITE_BATCH:
        case YUUKA_OP_ATTENTION_PREFILL:
        case YUUKA_OP_ATTENTION_PREFILL_PAGED:
        case YUUKA_OP_ATTENTION:
        case YUUKA_OP_KV_WRITE:
            return false;
        // Token- and position-independent ops are safe to replay as-is.
        default:
            return true;
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
        int32_t * dynamic_staging = nullptr;
        err = hipHostMalloc(reinterpret_cast<void **>(&dynamic_staging),
                            2 * sizeof(int32_t), hipHostMallocPortable);
        if (err != hipSuccess) {
            (void)hipStreamDestroy(stream);
            return fail(YUUKA_ERR_ALLOCATION, "hipHostMalloc dynamic staging: %s",
                        hip_error(err));
        }
        *out = new yuuka_stream{stream, dynamic_staging};
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
    if (stream->dynamic_staging != nullptr) {
        hipError_t err = hipHostFree(stream->dynamic_staging);
        if (err != hipSuccess) {
            g_last_error = std::string("hipHostFree: ") + hip_error(err);
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

YUUKA_API yuuka_status yuuka_memcpy_d2h_at(void * dst, const yuuka_buffer * src,
                                           uint64_t offset, uint64_t bytes) {
    return guard([&]() -> yuuka_status {
        if (dst == nullptr || src == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "destination or source is null");
        }
        if (!buffer_range_ok(src, offset, bytes)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "device-to-host range [%llu,%llu) exceeds buffer of %llu bytes",
                        static_cast<unsigned long long>(offset),
                        static_cast<unsigned long long>(offset + bytes),
                        static_cast<unsigned long long>(src->bytes));
        }
        hipError_t err = hipMemcpy(dst, static_cast<const uint8_t *>(src->ptr) + offset,
                                   bytes, hipMemcpyDeviceToHost);
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


YUUKA_API yuuka_status yuuka_buffer_copy(yuuka_stream * stream,
                                         yuuka_buffer * dst, uint64_t dst_offset,
                                         const yuuka_buffer * src, uint64_t src_offset,
                                         uint64_t bytes, yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || dst == nullptr || src == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "stream or buffer is null");
        }
        if (!buffer_range_ok(dst, dst_offset, bytes) ||
            !buffer_range_ok(src, src_offset, bytes)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "buffer copy range is out of bounds");
        }
        if (bytes == 0) {
            return YUUKA_OK;
        }
        hipError_t err = hipMemcpyAsync(
            static_cast<uint8_t *>(dst->ptr) + dst_offset,
            static_cast<const uint8_t *>(src->ptr) + src_offset,
            bytes, hipMemcpyDeviceToDevice, stream->stream);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "buffer copy: %s", hip_error(err));
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

YUUKA_API yuuka_status yuuka_dynamic_set(yuuka_stream * stream, yuuka_buffer * dynamic,
                                         int32_t token, uint32_t position) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || dynamic == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "stream or dynamic buffer is null");
        }
        if (dynamic->bytes < 2 * sizeof(int32_t)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "dynamic buffer must hold two int32 values");
        }
        if (stream->dynamic_staging == nullptr) {
            return fail(YUUKA_ERR_DEVICE, "stream has no pinned dynamic staging buffer");
        }
        stream->dynamic_staging[0] = token;
        stream->dynamic_staging[1] = static_cast<int32_t>(position);
        hipError_t err = hipMemcpyAsync(dynamic->ptr, stream->dynamic_staging,
                                        2 * sizeof(int32_t),
                                        hipMemcpyHostToDevice, stream->stream);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "dynamic parameter upload: %s", hip_error(err));
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

YUUKA_API yuuka_status yuuka_gemv_batch(yuuka_stream * stream, yuuka_buffer * out,
                                        const yuuka_buffer * x, const yuuka_buffer * w,
                                        uint64_t n_in, uint64_t n_out, uint64_t batch,
                                        uint32_t dtype, yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || out == nullptr || x == nullptr || w == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "null stream or buffer");
        }
        if (!check_gemv_shapes(n_in, dtype)) {
            return fail(YUUKA_ERR_UNSUPPORTED,
                        "unsupported gemv dtype %u or input width %llu",
                        dtype, static_cast<unsigned long long>(n_in));
        }
        if (n_out == 0 || batch == 0 || batch > 16) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "output width or batch is invalid");
        }
        uint64_t weight_row_bytes = 0;
        uint64_t weight_bytes = 0;
        uint64_t input_elements = 0;
        uint64_t output_elements = 0;
        uint64_t input_bytes = 0;
        uint64_t output_bytes = 0;
        if (!checked_row_bytes(n_in, dtype, &weight_row_bytes) ||
            __builtin_mul_overflow(weight_row_bytes, n_out, &weight_bytes) ||
            __builtin_mul_overflow(n_in, batch, &input_elements) ||
            __builtin_mul_overflow(n_out, batch, &output_elements) ||
            !checked_float_bytes(input_elements, &input_bytes) ||
            !checked_float_bytes(output_elements, &output_bytes)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv batch dimensions overflow");
        }
        if (w->bytes != weight_bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "weight buffer has %llu bytes, expected %llu",
                        static_cast<unsigned long long>(w->bytes),
                        static_cast<unsigned long long>(weight_bytes));
        }
        if (x->bytes < input_bytes || out->bytes < output_bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv batch buffer is too small");
        }
        const float * x_ptr = static_cast<const float *>(x->ptr);
        float * out_ptr = static_cast<float *>(out->ptr);
        switch (dtype) {
            case YUUKA_DTYPE_F32: yuuka::launch_gemv_batch_f32(out_ptr, x_ptr, static_cast<const float *>(w->ptr), n_in, n_out, batch, stream->stream); break;
            case YUUKA_DTYPE_F16: yuuka::launch_gemv_batch_f16(out_ptr, x_ptr, w->ptr, n_in, n_out, batch, stream->stream); break;
            case YUUKA_DTYPE_Q4_0: yuuka::launch_gemv_batch_q4_0(out_ptr, x_ptr, w->ptr, n_in, n_out, batch, stream->stream); break;
            case YUUKA_DTYPE_Q5_0: yuuka::launch_gemv_batch_q5_0(out_ptr, x_ptr, w->ptr, n_in, n_out, batch, stream->stream); break;
            case YUUKA_DTYPE_Q8_0: yuuka::launch_gemv_batch_q8_0(out_ptr, x_ptr, w->ptr, n_in, n_out, batch, stream->stream); break;
            case YUUKA_DTYPE_Q4_K: yuuka::launch_gemv_batch_q4_k(out_ptr, x_ptr, w->ptr, n_in, n_out, batch, stream->stream); break;
            case YUUKA_DTYPE_Q5_K: yuuka::launch_gemv_batch_q5_k(out_ptr, x_ptr, w->ptr, n_in, n_out, batch, stream->stream); break;
            case YUUKA_DTYPE_Q6_K: yuuka::launch_gemv_batch_q6_k(out_ptr, x_ptr, w->ptr, n_in, n_out, batch, stream->stream); break;
            default: return fail(YUUKA_ERR_UNSUPPORTED, "unsupported gemv dtype %u", dtype);
        }
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "gemv batch launch: %s", hip_error(err));
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

YUUKA_API yuuka_status yuuka_gemm(yuuka_stream * stream, yuuka_buffer * out,
                                  const yuuka_buffer * x, const yuuka_buffer * w,
                                  uint64_t n_in, uint64_t n_out, uint64_t batch,
                                  uint32_t dtype, yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || out == nullptr || x == nullptr || w == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "null stream or buffer");
        }
        if (!check_gemv_shapes(n_in, dtype) || (dtype != YUUKA_DTYPE_F32 && dtype != YUUKA_DTYPE_F16)) {
            return fail(YUUKA_ERR_UNSUPPORTED,
                        "unsupported gemm dtype %u or input width %llu",
                        dtype, static_cast<unsigned long long>(n_in));
        }
        if (n_out == 0 || batch == 0) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "output width or batch is zero");
        }
        uint64_t weight_row_bytes = 0;
        uint64_t weight_bytes = 0;
        uint64_t input_elements = 0;
        uint64_t output_elements = 0;
        uint64_t input_bytes = 0;
        uint64_t output_bytes = 0;
        if (!checked_row_bytes(n_in, dtype, &weight_row_bytes) ||
            __builtin_mul_overflow(weight_row_bytes, n_out, &weight_bytes) ||
            __builtin_mul_overflow(n_in, batch, &input_elements) ||
            __builtin_mul_overflow(n_out, batch, &output_elements) ||
            !checked_float_bytes(input_elements, &input_bytes) ||
            !checked_float_bytes(output_elements, &output_bytes)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm dimensions overflow");
        }
        if (w->bytes != weight_bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "weight buffer has %llu bytes, expected %llu",
                        static_cast<unsigned long long>(w->bytes),
                        static_cast<unsigned long long>(weight_bytes));
        }
        if (x->bytes < input_bytes || out->bytes < output_bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemm buffer is too small");
        }
        const float * x_ptr = static_cast<const float *>(x->ptr);
        float * out_ptr = static_cast<float *>(out->ptr);
        if (dtype == YUUKA_DTYPE_F32) {
            yuuka::launch_gemm_f32(out_ptr, x_ptr, static_cast<const float *>(w->ptr),
                                   n_in, n_out, batch, stream->stream);
        } else {
            yuuka::launch_gemm_f16(out_ptr, x_ptr, w->ptr, n_in, n_out, batch,
                                   stream->stream);
        }
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "gemm launch: %s", hip_error(err));
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
        uint64_t weight_row_bytes = 0;
        uint64_t weight_bytes = 0;
        uint64_t input_bytes = 0;
        uint64_t output_bytes = 0;
        if (!checked_row_bytes(n_in, dtype, &weight_row_bytes) ||
            __builtin_mul_overflow(weight_row_bytes, n_out, &weight_bytes) ||
            !checked_float_bytes(n_in, &input_bytes) ||
            !checked_float_bytes(n_out, &output_bytes)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "gemv dimensions overflow");
        }
        if (w->bytes != weight_bytes) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT,
                        "weight buffer has %llu bytes, expected %llu",
                        static_cast<unsigned long long>(w->bytes),
                        static_cast<unsigned long long>(weight_bytes));
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
            case YUUKA_DTYPE_Q4_0:
                yuuka::launch_gemv_q4_0(out_ptr, x_ptr, w->ptr, n_in, n_out,
                                        stream->stream);
                break;
            case YUUKA_DTYPE_Q5_0:
                yuuka::launch_gemv_q5_0(out_ptr, x_ptr, w->ptr, n_in, n_out,
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

YUUKA_API yuuka_status yuuka_gemm_wmma(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint64_t batch,
                        uint32_t dtype, yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || out == nullptr || x == nullptr || w == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "null stream or buffer");
        }
        yuuka_op op{};
        op.code = YUUKA_OP_GEMM;
        op.buf[0] = const_cast<yuuka_buffer *>(w);
        op.buf[1] = const_cast<yuuka_buffer *>(x);
        op.buf[2] = out;
        op.n[0] = n_in;
        op.n[1] = n_out;
        op.n[2] = batch;
        op.u[0] = dtype;
        op.u[1] = YUUKA_GEMM_PATH_WMMA;
        yuuka_status status = validate_gemm_plan_op(op);
        if (status != YUUKA_OK) {
            return status;
        }
        yuuka::launch_gemm_wmma_f32out(static_cast<float *>(out->ptr),
                                       static_cast<const float *>(x->ptr),
                                       w->ptr, n_in, n_out, batch, dtype,
                                       stream->stream);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "wmma gemm launch: %s", hip_error(err));
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

YUUKA_API yuuka_status yuuka_gemm_blas(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint64_t batch,
                        yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || out == nullptr || x == nullptr || w == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "null stream or buffer");
        }
        yuuka_op op{};
        op.code = YUUKA_OP_GEMM;
        op.buf[0] = const_cast<yuuka_buffer *>(w);
        op.buf[1] = const_cast<yuuka_buffer *>(x);
        op.buf[2] = out;
        op.n[0] = n_in;
        op.n[1] = n_out;
        op.n[2] = batch;
        op.u[0] = YUUKA_DTYPE_F16;
        op.u[1] = YUUKA_GEMM_PATH_BLAS;
        yuuka_status status = validate_gemm_plan_op(op);
        if (status != YUUKA_OK) {
            return status;
        }
        const int blas = yuuka::launch_gemm_blas_f16(static_cast<float *>(out->ptr),
                                                     x->ptr, w->ptr, n_in, n_out,
                                                     batch, stream->stream);
        if (blas != 0) {
            return fail(YUUKA_ERR_DEVICE, "rocBLAS gemm failed with status %d", blas);
        }
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "rocBLAS gemm launch: %s", hip_error(err));
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

YUUKA_API yuuka_status yuuka_cast_f32_to_f16(yuuka_stream * stream, yuuka_buffer * dst,
                        const yuuka_buffer * src, uint64_t n, yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (stream == nullptr || dst == nullptr || src == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "null stream or buffer");
        }
        yuuka_op op{};
        op.code = YUUKA_OP_CAST_F16;
        op.buf[0] = const_cast<yuuka_buffer *>(src);
        op.buf[1] = dst;
        op.n[0] = n;
        yuuka_status status = validate_cast_plan_op(op);
        if (status != YUUKA_OK) {
            return status;
        }
        yuuka::launch_cast_f32_to_f16(dst->ptr, static_cast<const float *>(src->ptr),
                                       n, stream->stream);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "cast launch: %s", hip_error(err));
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


YUUKA_API yuuka_status yuuka_graph_create(yuuka_plan * plan, yuuka_stream * stream,
                                          yuuka_buffer * dynamic, int run_tail,
                                          yuuka_graph ** out) {
    return guard([&]() -> yuuka_status {
        if (plan == nullptr || stream == nullptr || dynamic == nullptr || out == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "graph arguments are null");
        }
        if (dynamic->bytes < 2 * sizeof(int32_t)) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "dynamic buffer must hold two int32 values");
        }
        *out = nullptr;
        for (const yuuka_op & op : plan->ops) {
            if ((op.flags & YUUKA_OP_FLAG_TAIL) != 0 && run_tail == 0) {
                continue;
            }
            if (!op_is_graph_capturable(op)) {
                return fail(YUUKA_ERR_UNSUPPORTED,
                            "plan op %u is position-dependent and has no "
                            "dynamic-parameter kernel", op.code);
            }
        }
        hipError_t err = hipStreamSynchronize(stream->stream);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "graph capture synchronize: %s", hip_error(err));
        }
        err = hipStreamBeginCapture(stream->stream, hipStreamCaptureModeThreadLocal);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_UNSUPPORTED, "graph capture begin: %s", hip_error(err));
        }
        for (const yuuka_op & op : plan->ops) {
            if ((op.flags & YUUKA_OP_FLAG_TAIL) != 0 && run_tail == 0) {
                continue;
            }
            for (int i = 0; i < 12; ++i) {
                if (op.buf[i] != nullptr && op.buf[i]->ptr == nullptr) {
                    hipGraph_t abandoned = nullptr;
                    (void)hipStreamEndCapture(stream->stream, &abandoned);
                    if (abandoned != nullptr) {
                        (void)hipGraphDestroy(abandoned);
                    }
                    return fail(YUUKA_ERR_INVALID_ARGUMENT,
                                "graph op %u has a null buffer", op.code);
                }
            }
            const int32_t * dynamic_ptr = reinterpret_cast<const int32_t *>(dynamic->ptr);
            yuuka_status status = execute_op(op, 0, 0, stream->stream, dynamic_ptr);
            if (status != YUUKA_OK) {
                hipGraph_t abandoned = nullptr;
                (void)hipStreamEndCapture(stream->stream, &abandoned);
                if (abandoned != nullptr) {
                    (void)hipGraphDestroy(abandoned);
                }
                return status;
            }
        }
        hipGraph_t captured = nullptr;
        err = hipStreamEndCapture(stream->stream, &captured);
        if (err != hipSuccess) {
            if (captured != nullptr) {
                (void)hipGraphDestroy(captured);
            }
            return fail(YUUKA_ERR_UNSUPPORTED, "graph capture end: %s", hip_error(err));
        }
        hipGraphExec_t executable = nullptr;
        err = hipGraphInstantiate(&executable, captured, nullptr, nullptr, 0);
        if (err != hipSuccess) {
            (void)hipGraphDestroy(captured);
            return fail(YUUKA_ERR_UNSUPPORTED, "graph instantiate: %s", hip_error(err));
        }
        yuuka_graph * graph = new yuuka_graph();
        graph->graph = captured;
        graph->exec = executable;
        *out = graph;
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_graph_replay(yuuka_graph * graph, yuuka_stream * stream,
                                          yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (graph == nullptr || graph->exec == nullptr || stream == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "graph or stream is null");
        }
        hipError_t err = hipGraphLaunch(graph->exec, stream->stream);
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "graph launch: %s", hip_error(err));
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

YUUKA_API void yuuka_graph_destroy(yuuka_graph * graph) {
    if (graph == nullptr) {
        return;
    }
    if (graph->exec != nullptr) {
        (void)hipGraphExecDestroy(graph->exec);
    }
    if (graph->graph != nullptr) {
        (void)hipGraphDestroy(graph->graph);
    }
    delete graph;
}

YUUKA_API yuuka_status yuuka_plan_create(yuuka_device * dev, yuuka_plan ** out) {
    return guard([&]() -> yuuka_status {
        if (dev == nullptr || out == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "device or plan handle is null");
        }
        *out = new yuuka_plan();
        return YUUKA_OK;
    });
}

YUUKA_API void yuuka_plan_destroy(yuuka_plan * plan) {
    delete plan;
}

YUUKA_API yuuka_status yuuka_plan_add(yuuka_plan * plan, const yuuka_op * op) {
    return guard([&]() -> yuuka_status {
        if (plan == nullptr || op == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "plan or op is null");
        }
        if (op->code < YUUKA_OP_GEMV || op->code > YUUKA_OP_MOE_REDUCE_ADD) {
            return fail(YUUKA_ERR_UNSUPPORTED, "unknown op code %u", op->code);
        }
        yuuka_status status = YUUKA_OK;
        switch (op->code) {
            case YUUKA_OP_GEMV:
                status = validate_gemv_plan_op(*op, false);
                break;
            case YUUKA_OP_GEMV_BATCH:
                status = validate_gemv_plan_op(*op, true);
                break;
            case YUUKA_OP_GEMM:
                status = validate_gemm_plan_op(*op);
                break;
            case YUUKA_OP_RMSNORM_BATCH:
                status = validate_batch_norm_plan_op(*op);
                break;
            case YUUKA_OP_RMSNORM_HEADS_BATCH:
                status = validate_batch_heads_plan_op(*op);
                break;
            case YUUKA_OP_ROPE_BATCH:
                status = validate_rope_batch_plan_op(*op);
                break;
            case YUUKA_OP_KV_WRITE_BATCH:
                status = validate_kv_write_batch_plan_op(*op);
                break;
            case YUUKA_OP_ATTENTION_PREFILL:
                status = validate_attention_prefill_plan_op(*op);
                break;
            case YUUKA_OP_RECURRENT_CHUNK:
                status = validate_recurrent_chunk_plan_op(*op);
                break;
            case YUUKA_OP_EMBEDDING:
                status = validate_embedding_plan_op(*op);
                break;
            case YUUKA_OP_EMBEDDING_BATCH:
                status = validate_embedding_batch_plan_op(*op);
                break;
            case YUUKA_OP_CAST_F16:
                status = validate_cast_plan_op(*op);
                break;
            case YUUKA_OP_KV_WRITE_PAGED:
                status = validate_kv_write_paged_plan_op(*op, false);
                break;
            case YUUKA_OP_KV_WRITE_BATCH_PAGED:
                status = validate_kv_write_paged_plan_op(*op, true);
                break;
            case YUUKA_OP_ATTENTION_PAGED:
                status = validate_attention_paged_plan_op(*op);
                break;
            case YUUKA_OP_ATTENTION_PREFILL_PAGED:
                status = validate_attention_prefill_paged_plan_op(*op);
                break;
            case YUUKA_OP_MOE_ROUTE:
                status = validate_moe_route_plan_op(*op);
                break;
            case YUUKA_OP_MOE_GATE_UP:
                status = validate_moe_gate_up_plan_op(*op);
                break;
            case YUUKA_OP_MOE_DOWN:
                status = validate_moe_down_plan_op(*op);
                break;
            case YUUKA_OP_MOE_REDUCE_ADD:
                status = validate_moe_reduce_plan_op(*op);
                break;
            case YUUKA_OP_RMSNORM:
                status = validate_rmsnorm_plan_op(*op);
                break;
            case YUUKA_OP_L2NORM:
                status = validate_l2norm_plan_op(*op);
                break;
            case YUUKA_OP_GATED_RMSNORM:
                status = validate_gated_rmsnorm_plan_op(*op);
                break;
            case YUUKA_OP_RMSNORM_HEADS:
                status = validate_rmsnorm_heads_plan_op(*op);
                break;
            case YUUKA_OP_ROPE:
                status = validate_rope_plan_op(*op);
                break;
            case YUUKA_OP_ATTENTION:
                status = validate_attention_plan_op(*op);
                break;
            case YUUKA_OP_CONV1D:
                status = validate_conv1d_plan_op(*op);
                break;
            case YUUKA_OP_GDN:
                status = validate_gdn_plan_op(*op);
                break;
            case YUUKA_OP_SILU_MUL:
                status = validate_elementwise_plan_op(*op, 3);
                break;
            case YUUKA_OP_ADD:
            case YUUKA_OP_COPY:
                status = validate_elementwise_plan_op(*op, 2);
                break;
            case YUUKA_OP_SCALE:
                status = validate_elementwise_plan_op(*op, 1);
                break;
            case YUUKA_OP_KV_WRITE:
                status = validate_kv_write_plan_op(*op);
                break;
            default:
                break;
        }
        if (status != YUUKA_OK) {
            return status;
        }
        plan->ops.push_back(*op);
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_plan_execute_profiled(
    yuuka_plan * plan, yuuka_stream * stream, int32_t token, uint32_t position,
    int run_tail, uint32_t * op_codes, uint64_t * op_times_ns,
    uint64_t capacity, uint64_t * op_count, yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (plan == nullptr || stream == nullptr || op_count == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "plan, stream, or op_count is null");
        }
        uint64_t executed = 0;
        for (const yuuka_op & op : plan->ops) {
            if ((op.flags & YUUKA_OP_FLAG_TAIL) != 0 && run_tail == 0) {
                continue;
            }
            executed++;
        }
        if (capacity < executed || (executed != 0 && (op_codes == nullptr || op_times_ns == nullptr))) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "profile output capacity %llu is too small for %llu ops",
                        static_cast<unsigned long long>(capacity),
                        static_cast<unsigned long long>(executed));
        }
        std::vector<hipEvent_t> starts;
        std::vector<hipEvent_t> stops;
        std::vector<const yuuka_op *> ops;
        starts.reserve(executed);
        stops.reserve(executed);
        ops.reserve(executed);
        auto cleanup = [&]() {
            for (hipEvent_t ev : starts) {
                hipError_t cleanup_err = hipEventDestroy(ev);
                (void)cleanup_err;
            }
            for (hipEvent_t ev : stops) {
                hipError_t cleanup_err = hipEventDestroy(ev);
                (void)cleanup_err;
            }
        };
        for (const yuuka_op & op : plan->ops) {
            if ((op.flags & YUUKA_OP_FLAG_TAIL) != 0 && run_tail == 0) {
                continue;
            }
            for (int i = 0; i < 12; ++i) {
                if (op.buf[i] != nullptr && op.buf[i]->ptr == nullptr) {
                    cleanup();
                    return fail(YUUKA_ERR_INVALID_ARGUMENT, "plan op %u has a null buffer", op.code);
                }
            }
            hipEvent_t start = nullptr;
            hipEvent_t stop = nullptr;
            hipError_t err = hipEventCreate(&start);
            if (err != hipSuccess) {
                cleanup();
                return fail(YUUKA_ERR_DEVICE, "profile event create: %s", hip_error(err));
            }
            starts.push_back(start);
            err = hipEventCreate(&stop);
            if (err != hipSuccess) {
                cleanup();
                return fail(YUUKA_ERR_DEVICE, "profile event create: %s", hip_error(err));
            }
            stops.push_back(stop);
            ops.push_back(&op);
            err = hipEventRecord(start, stream->stream);
            if (err != hipSuccess) {
                cleanup();
                return fail(YUUKA_ERR_DEVICE, "profile event record: %s", hip_error(err));
            }
            yuuka_status status = execute_op(op, token, position, stream->stream);
            if (status != YUUKA_OK) {
                cleanup();
                return status;
            }
            err = hipEventRecord(stop, stream->stream);
            if (err != hipSuccess) {
                cleanup();
                return fail(YUUKA_ERR_DEVICE, "profile event record: %s", hip_error(err));
            }
        }
        hipError_t err = hipStreamSynchronize(stream->stream);
        if (err != hipSuccess) {
            cleanup();
            return fail(YUUKA_ERR_DEVICE, "profile synchronize: %s", hip_error(err));
        }
        for (size_t i = 0; i < ops.size(); ++i) {
            float milliseconds = 0.0f;
            err = hipEventElapsedTime(&milliseconds, starts[i], stops[i]);
            if (err != hipSuccess) {
                cleanup();
                return fail(YUUKA_ERR_DEVICE, "profile elapsed time: %s", hip_error(err));
            }
            op_codes[i] = ops[i]->code;
            op_times_ns[i] = static_cast<uint64_t>(milliseconds * 1000000.0f);
        }
        *op_count = executed;
        cleanup();
        if (completion != nullptr) {
            err = hipEventRecord(completion->event, stream->stream);
            if (err != hipSuccess) {
                return fail(YUUKA_ERR_DEVICE, "hipEventRecord: %s", hip_error(err));
            }
        }
        return YUUKA_OK;
    });
}

YUUKA_API yuuka_status yuuka_plan_execute(yuuka_plan * plan, yuuka_stream * stream,
                                          int32_t token, uint32_t position,
                                          int run_tail, yuuka_event * completion) {
    return guard([&]() -> yuuka_status {
        if (plan == nullptr || stream == nullptr) {
            return fail(YUUKA_ERR_INVALID_ARGUMENT, "plan or stream is null");
        }
        for (const yuuka_op & op : plan->ops) {
            if ((op.flags & YUUKA_OP_FLAG_TAIL) != 0 && run_tail == 0) {
                continue;
            }
            for (int i = 0; i < 12; ++i) {
                if (op.buf[i] != nullptr && op.buf[i]->ptr == nullptr) {
                    return fail(YUUKA_ERR_INVALID_ARGUMENT, "plan op %u has a null buffer", op.code);
                }
            }
            yuuka_status status = execute_op(op, token, position, stream->stream);
            if (status != YUUKA_OK) {
                return status;
            }
        }
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return fail(YUUKA_ERR_DEVICE, "plan execute: %s", hip_error(err));
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

#include "yuuka_hip.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

#define CHECK_STATUS(call)                                                               \
    do {                                                                                 \
        const yuuka_status st__ = (call);                                                \
        if (st__ != YUUKA_OK) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__,      \
                         #call, int(st__), yuuka_last_error());                          \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

void put_half(uint8_t * dst, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000;
    const int32_t exp = int32_t((bits >> 23) & 0xff) - 127;
    const uint32_t mant = bits & 0x7fffff;
    uint16_t half = 0;
    if (exp > 15) {
        half = uint16_t(sign | 0x7c00);
    } else if (exp < -24) {
        half = uint16_t(sign);
    } else if (exp < -14) {
        const uint32_t shift = uint32_t(-exp - 14);
        const uint32_t full = mant | 0x800000;
        half = uint16_t(sign | ((full + (1u << (shift + 12))) >> (shift + 13)));
    } else {
        uint32_t rounded = mant + 0x1000;
        int32_t e = exp;
        if (rounded & 0x800000) {
            rounded = 0;
            e++;
            if (e > 15) {
                half = uint16_t(sign | 0x7c00);
                *reinterpret_cast<uint16_t *>(dst) = half;
                return;
            }
        }
        half = uint16_t(sign | ((e + 15) << 10) | (rounded >> 13));
    }
    *reinterpret_cast<uint16_t *>(dst) = half;
}

float half_to_float(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000) << 16;
    const uint32_t exp = (value >> 10) & 0x1f;
    const uint32_t mant = value & 0x3ff;
    uint32_t bits = 0;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

float packed_value(uint32_t dtype, const uint8_t * row, int64_t index) {
    if (dtype == YUUKA_DTYPE_Q4_0) {
        const uint8_t * blk = row + (index / 32) * 18;
        const int64_t j = index % 32;
        const uint8_t packed = blk[2 + (j & 15)];
        const int q = (j < 16) ? (packed & 0xF) : (packed >> 4);
        return half_to_float(uint16_t(blk[0] | (blk[1] << 8))) * float(q - 8);
    }
    const uint8_t * blk = row + (index / 32) * 22;
    const int64_t j = index % 32;
    const uint8_t packed = blk[6 + (j & 15)];
    const uint32_t qh = uint32_t(blk[2]) | (uint32_t(blk[3]) << 8) |
                        (uint32_t(blk[4]) << 16) | (uint32_t(blk[5]) << 24);
    const int high = (j < 16) ? int(((qh >> j) << 4) & 0x10)
                              : int((qh >> ((j & 15) + 12)) & 0x10);
    const int q = (j < 16 ? (packed & 0xF) : (packed >> 4)) | high;
    return half_to_float(uint16_t(blk[0] | (blk[1] << 8))) * float(q - 16);
}

} // namespace

int main() {
    yuuka_device * device = nullptr;
    if (yuuka_device_open(0, &device) != YUUKA_OK) {
        std::fprintf(stderr, "SKIP: no HIP device: %s\n", yuuka_last_error());
        return 77;
    }
    yuuka_stream * stream = nullptr;
    yuuka_event * event = nullptr;
    CHECK_STATUS(yuuka_stream_create(device, &stream));
    CHECK_STATUS(yuuka_event_create(device, &event));

    const int rows = 4, cols = 64, vocab = 3;
    const int32_t tokens[rows] = {2, 0, 1, 2};
    std::mt19937 rng(17);
    bool ok = true;
    for (uint32_t dtype : {YUUKA_DTYPE_Q4_0, YUUKA_DTYPE_Q5_0}) {
        const int block = dtype == YUUKA_DTYPE_Q4_0 ? 18 : 22;
        const size_t row_bytes = size_t(cols / 32) * block;
        std::vector<uint8_t> table(row_bytes * vocab);
        for (size_t i = 0; i < table.size(); ++i) table[i] = uint8_t(rng() & 0xff);
        for (int v = 0; v < vocab; ++v) {
            uint8_t * row = table.data() + v * row_bytes;
            for (int b = 0; b < cols / 32; ++b) put_half(row + b * block, 0.01f);
        }
        std::vector<float> want(size_t(rows) * cols);
        for (int r = 0; r < rows; ++r) {
            const uint8_t * row = table.data() + tokens[r] * row_bytes;
            for (int i = 0; i < cols; ++i) {
                want[size_t(r) * cols + i] = packed_value(dtype, row, i);
            }
        }
        yuuka_buffer * w_buf = nullptr;
        yuuka_buffer * t_buf = nullptr;
        yuuka_buffer * o_buf = nullptr;
        CHECK_STATUS(yuuka_buffer_alloc(device, table.size(), &w_buf));
        CHECK_STATUS(yuuka_buffer_alloc(device, sizeof(tokens), &t_buf));
        CHECK_STATUS(yuuka_buffer_alloc(device, want.size() * 4, &o_buf));
        CHECK_STATUS(yuuka_memcpy_h2d(w_buf, table.data(), table.size()));
        CHECK_STATUS(yuuka_memcpy_h2d(t_buf, tokens, sizeof(tokens)));
        yuuka_plan * plan = nullptr;
        CHECK_STATUS(yuuka_plan_create(device, &plan));
        yuuka_op op{};
        op.code = YUUKA_OP_EMBEDDING_BATCH;
        op.buf[0] = w_buf;
        op.buf[1] = o_buf;
        op.buf[2] = t_buf;
        op.n[0] = cols;
        op.n[1] = rows;
        op.n[2] = row_bytes;
        op.n[3] = vocab;
        op.u[0] = dtype;
        CHECK_STATUS(yuuka_plan_add(plan, &op));
        CHECK_STATUS(yuuka_plan_execute(plan, stream, 0, 0, 1, event));
        CHECK_STATUS(yuuka_event_wait(event));
        std::vector<float> got(want.size());
        CHECK_STATUS(yuuka_memcpy_d2h(got.data(), o_buf, got.size() * 4));
        float max_diff = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            max_diff = std::fmax(max_diff, std::fabs(got[i] - want[i]));
        }
        std::printf("embedding %s: max abs diff %.6g\n",
                    dtype == YUUKA_DTYPE_Q4_0 ? "Q4_0" : "Q5_0", max_diff);
        if (max_diff != 0) ok = false;
        yuuka_plan_destroy(plan);
        yuuka_buffer_free(o_buf);
        yuuka_buffer_free(t_buf);
        yuuka_buffer_free(w_buf);
    }
    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return ok ? 0 : 1;
}

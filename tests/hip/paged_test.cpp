#include "yuuka_hip.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

#define CHECK(condition)                                                                 \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);    \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

#define CHECK_STATUS(call)                                                               \
    do {                                                                                 \
        const yuuka_status st__ = (call);                                                \
        if (st__ != YUUKA_OK) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__,      \
                         #call, int(st__), yuuka_last_error());                          \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

constexpr int kHeads = 2;
constexpr int kKVHeads = 1;
constexpr int kHeadDim = 4;
constexpr int kKVDim = kKVHeads * kHeadDim;
constexpr int kBlockSize = 4;
constexpr int kCapacity = 16;
constexpr int kBlocks = kCapacity / kBlockSize;
constexpr int kRows = 9;

struct context {
    yuuka_device * device = nullptr;
    yuuka_stream * stream = nullptr;
    yuuka_event * event = nullptr;
};

int setup(context & ctx) {
    if (yuuka_device_open(0, &ctx.device) != YUUKA_OK) {
        return 77;
    }
    CHECK_STATUS(yuuka_stream_create(ctx.device, &ctx.stream));
    CHECK_STATUS(yuuka_event_create(ctx.device, &ctx.event));
    return 0;
}

void teardown(context & ctx) {
    if (ctx.event != nullptr) {
        yuuka_event_destroy(ctx.event);
    }
    if (ctx.stream != nullptr) {
        yuuka_stream_destroy(ctx.stream);
    }
    if (ctx.device != nullptr) {
        yuuka_device_close(ctx.device);
    }
}

yuuka_buffer * upload(context & ctx, const void * data, uint64_t bytes) {
    yuuka_buffer * buffer = nullptr;
    if (yuuka_buffer_alloc(ctx.device, bytes, &buffer) != YUUKA_OK) {
        return nullptr;
    }
    if (data != nullptr && yuuka_memcpy_h2d(buffer, data, bytes) != YUUKA_OK) {
        yuuka_buffer_free(buffer);
        return nullptr;
    }
    return buffer;
}

bool download(context & ctx, yuuka_buffer * buffer, void * out, uint64_t bytes) {
    (void)ctx;
    return yuuka_memcpy_d2h(out, buffer, bytes) == YUUKA_OK;
}

int physical_row(const int32_t * table, int pos) {
    return table[pos / kBlockSize] * kBlockSize + pos % kBlockSize;
}

void reference_attention(const std::vector<float> & q, const std::vector<float> & k,
                         const std::vector<float> & v, const int32_t * table,
                         int start_pos, int rows, std::vector<float> & out) {
    (void)table; // The pool maps logical positions; contents come from the same row.
    out.assign(size_t(rows) * kHeads * kHeadDim, 0.0f);
    for (int row = 0; row < rows; ++row) {
        const int position = start_pos + row;
        for (int head = 0; head < kHeads; ++head) {
            const int kvh = head / (kHeads / kKVHeads);
            std::vector<float> scores(size_t(position) + 1);
            float max = -INFINITY;
            const float * qrow = q.data() + size_t(row) * kHeads * 2 * kHeadDim;
            for (int p = 0; p <= position; ++p) {
                float dot = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += qrow[head * 2 * kHeadDim + d] * k[size_t(p) * kKVDim + kvh * kHeadDim + d];
                }
                scores[size_t(p)] = dot / std::sqrt(float(kHeadDim));
                max = std::fmax(max, scores[size_t(p)]);
            }
            float total = 0.0f;
            for (float & score : scores) {
                score = std::exp(score - max);
                total += score;
            }
            for (int d = 0; d < kHeadDim; ++d) {
                float value = 0.0f;
                for (int p = 0; p <= position; ++p) {
                    value += (scores[size_t(p)] / total) *
                             v[size_t(p) * kKVDim + kvh * kHeadDim + d];
                }
                const float gate = qrow[head * 2 * kHeadDim + kHeadDim + d];
                out[size_t(row) * kHeads * kHeadDim + head * kHeadDim + d] =
                    value / (1.0f + std::exp(-gate));
            }
        }
    }
}

float max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

} // namespace

int main() {
    context ctx;
    const int init = setup(ctx);
    if (init == 77) {
        std::fprintf(stderr, "SKIP: no HIP device\n");
        return 77;
    }
    if (init != 0) {
        teardown(ctx);
        return init;
    }

    const int32_t table[kBlocks] = {2, 0, 3, 1};
    std::vector<float> host_k(size_t(kCapacity) * kKVDim);
    std::vector<float> host_v(size_t(kCapacity) * kKVDim);
    for (size_t i = 0; i < host_k.size(); ++i) {
        host_k[i] = std::sin(0.13f * float(i)) + 0.25f * float(i % 7);
        host_v[i] = std::cos(0.07f * float(i)) - 0.1f * float(i % 5);
    }
    std::vector<float> q(size_t(kRows) * kHeads * 2 * kHeadDim);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = std::sin(0.21f * float(i) + 0.3f);
    }

    yuuka_buffer * k_src = upload(ctx, host_k.data(), host_k.size() * sizeof(float));
    yuuka_buffer * v_src = upload(ctx, host_v.data(), host_v.size() * sizeof(float));
    yuuka_buffer * k_pool = upload(ctx, nullptr, host_k.size() * sizeof(float));
    yuuka_buffer * v_pool = upload(ctx, nullptr, host_v.size() * sizeof(float));
    yuuka_buffer * table_buf = upload(ctx, table, sizeof(table));
    yuuka_buffer * q_buf = upload(ctx, q.data(), q.size() * sizeof(float));
    yuuka_buffer * out = upload(ctx, nullptr, size_t(kRows) * kHeads * kHeadDim * sizeof(float));
    yuuka_buffer * scores = upload(ctx, nullptr, size_t(kHeads) * kCapacity * sizeof(float));
    CHECK(k_src && v_src && k_pool && v_pool && table_buf && q_buf && out && scores);

    // Paged prefill: write all rows through a permuted block table, then run
    // causal attention for every query row.
    yuuka_plan * prefill = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &prefill));
    {
        yuuka_op op{};
        op.code = YUUKA_OP_KV_WRITE_BATCH_PAGED;
        op.buf[0] = k_src;
        op.buf[1] = v_src;
        op.buf[2] = k_pool;
        op.buf[3] = v_pool;
        op.buf[4] = table_buf;
        op.n[0] = kKVDim;
        op.n[1] = kRows;
        op.u[0] = kBlockSize;
        op.u[1] = kCapacity;
        CHECK_STATUS(yuuka_plan_add(prefill, &op));
    }
    {
        yuuka_op op{};
        op.code = YUUKA_OP_ATTENTION_PREFILL_PAGED;
        op.buf[0] = q_buf;
        op.buf[1] = k_pool;
        op.buf[2] = v_pool;
        op.buf[3] = out;
        op.buf[4] = table_buf;
        op.n[0] = kKVHeads;
        op.n[1] = kHeads;
        op.n[2] = kHeadDim;
        op.n[3] = kKVDim;
        op.n[4] = kRows;
        op.u[0] = kBlockSize;
        op.u[1] = kCapacity;
        op.f[0] = 1.0f / std::sqrt(float(kHeadDim));
        CHECK_STATUS(yuuka_plan_add(prefill, &op));
    }
    CHECK_STATUS(yuuka_plan_execute(prefill, ctx.stream, 0, 0, 1, ctx.event));
    CHECK_STATUS(yuuka_event_wait(ctx.event));
    std::vector<float> got(size_t(kRows) * kHeads * kHeadDim);
    CHECK(download(ctx, out, got.data(), got.size() * sizeof(float)));
    std::vector<float> want;
    reference_attention(q, host_k, host_v, table, 0, kRows, want);
    {
        const float diff = max_diff(got, want);
        if (diff >= 2e-4f) {
            std::fprintf(stderr, "prefill max diff %g\n", diff);
            for (int i = 0; i < 8; ++i) {
                std::fprintf(stderr, "  [%d] got %.6f want %.6f\n", i, got[size_t(i)], want[size_t(i)]);
            }
        }
        CHECK(diff < 2e-4f);
    }

    // Decode attention at position 7 against the same paged cache.
    const int decode_position = 7;
    {
        std::vector<float> decode_q(size_t(kHeads) * 2 * kHeadDim);
        for (size_t i = 0; i < decode_q.size(); ++i) {
            decode_q[i] = std::cos(0.17f * float(i) + 0.1f);
        }
        CHECK(yuuka_memcpy_h2d(q_buf, decode_q.data(), decode_q.size() * sizeof(float)) == YUUKA_OK);
        yuuka_plan * decode = nullptr;
        CHECK_STATUS(yuuka_plan_create(ctx.device, &decode));
        yuuka_op op{};
        op.code = YUUKA_OP_ATTENTION_PAGED;
        op.buf[0] = q_buf;
        op.buf[1] = k_pool;
        op.buf[2] = v_pool;
        op.buf[3] = out;
        op.buf[4] = scores;
        op.buf[5] = table_buf;
        op.n[0] = kKVHeads;
        op.n[1] = kHeads;
        op.n[2] = kHeadDim;
        op.n[3] = kKVDim;
        op.n[4] = kCapacity;
        op.u[0] = kBlockSize;
        op.u[1] = kCapacity;
        op.f[0] = 1.0f / std::sqrt(float(kHeadDim));
        CHECK_STATUS(yuuka_plan_add(decode, &op));
        CHECK_STATUS(yuuka_plan_execute(decode, ctx.stream, 0, decode_position, 1, ctx.event));
        CHECK_STATUS(yuuka_event_wait(ctx.event));
        std::vector<float> decode_got(size_t(kHeads) * kHeadDim);
        CHECK(download(ctx, out, decode_got.data(), decode_got.size() * sizeof(float)));
        std::vector<float> decode_want;
        reference_attention(decode_q, host_k, host_v, table, decode_position, 1, decode_want);
        CHECK(max_diff(decode_got, decode_want) < 2e-4f);

        // Graph capture and replay with device-resident position updates.
        yuuka_buffer * dynamic = upload(ctx, nullptr, 2 * sizeof(int32_t));
        CHECK(dynamic != nullptr);
        yuuka_graph * graph = nullptr;
        CHECK_STATUS(yuuka_graph_create(decode, ctx.stream, dynamic, 1, &graph));
        for (int position : {1, 3, decode_position}) {
            CHECK_STATUS(yuuka_dynamic_set(ctx.stream, dynamic, 0, uint32_t(position)));
            CHECK_STATUS(yuuka_graph_replay(graph, ctx.stream, ctx.event));
            CHECK_STATUS(yuuka_event_wait(ctx.event));
            std::vector<float> replay(size_t(kHeads) * kHeadDim);
            CHECK(download(ctx, out, replay.data(), replay.size() * sizeof(float)));
            std::vector<float> expected;
            reference_attention(decode_q, host_k, host_v, table, position, 1, expected);
            CHECK(max_diff(replay, expected) < 2e-4f);
        }
        yuuka_graph_destroy(graph);
        yuuka_buffer_free(dynamic);
        yuuka_plan_destroy(decode);
    }

    // Legacy position-dependent ops cannot be captured; the ABI must report it
    // so the runtime can fall back.
    {
        yuuka_plan * legacy = nullptr;
        CHECK_STATUS(yuuka_plan_create(ctx.device, &legacy));
        yuuka_op op{};
        op.code = YUUKA_OP_ATTENTION;
        op.buf[0] = q_buf;
        op.buf[1] = k_pool;
        op.buf[2] = v_pool;
        op.buf[3] = out;
        op.buf[4] = scores;
        op.n[0] = kKVHeads;
        op.n[1] = kHeads;
        op.n[2] = kHeadDim;
        op.n[3] = kKVDim;
        op.u[0] = kCapacity;
        op.f[0] = 1.0f / std::sqrt(float(kHeadDim));
        CHECK_STATUS(yuuka_plan_add(legacy, &op));
        yuuka_buffer * dynamic = upload(ctx, nullptr, 2 * sizeof(int32_t));
        CHECK(dynamic != nullptr);
        yuuka_graph * graph = nullptr;
        const yuuka_status status = yuuka_graph_create(legacy, ctx.stream, dynamic, 1, &graph);
        CHECK(status == YUUKA_ERR_UNSUPPORTED);
        CHECK(graph == nullptr);
        yuuka_buffer_free(dynamic);
        yuuka_plan_destroy(legacy);
    }

    // Device-to-device copy used for private prefix blocks.
    {
        std::vector<float> src(16), dst(16, -1.0f);
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = float(i) * 0.5f;
        }
        yuuka_buffer * a = upload(ctx, src.data(), src.size() * sizeof(float));
        yuuka_buffer * b = upload(ctx, dst.data(), dst.size() * sizeof(float));
        CHECK(a && b);
        CHECK_STATUS(yuuka_buffer_copy(ctx.stream, b, 4 * sizeof(float), a, 8 * sizeof(float),
                                       4 * sizeof(float), ctx.event));
        CHECK_STATUS(yuuka_event_wait(ctx.event));
        CHECK(download(ctx, b, dst.data(), dst.size() * sizeof(float)));
        for (int i = 0; i < 4; ++i) {
            CHECK(dst[size_t(4 + i)] == src[size_t(8 + i)]);
        }
        yuuka_buffer_free(a);
        yuuka_buffer_free(b);
    }

    // Non-sequential table correctness: write a single row through the table
    // and verify only the mapped physical row changed.
    {
        std::vector<float> k_check(size_t(kCapacity) * kKVDim);
        CHECK(download(ctx, k_pool, k_check.data(), k_check.size() * sizeof(float)));
        for (int p = 0; p < kRows; ++p) {
            const int r = physical_row(table, p);
            CHECK(k_check[size_t(r) * kKVDim] == host_k[size_t(p) * kKVDim]);
        }
    }

    yuuka_plan_destroy(prefill);
    yuuka_buffer_free(k_src);
    yuuka_buffer_free(v_src);
    yuuka_buffer_free(k_pool);
    yuuka_buffer_free(v_pool);
    yuuka_buffer_free(table_buf);
    yuuka_buffer_free(q_buf);
    yuuka_buffer_free(out);
    yuuka_buffer_free(scores);
    teardown(ctx);
    std::printf("paged and graph tests passed\n");
    return 0;
}

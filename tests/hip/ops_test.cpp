#include "yuuka_hip.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool check(yuuka_status status, const char * what) {
    if (status == YUUKA_OK) return true;
    std::fprintf(stderr, "%s: %s\n", what, yuuka_last_error());
    return false;
}

float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void put_half(uint8_t * dst, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
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
            }
        }
        if (half == 0) {
            half = uint16_t(sign | ((e + 15) << 10) | (rounded >> 13));
        }
    }
    dst[0] = uint8_t(half & 0xff);
    dst[1] = uint8_t(half >> 8);
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }

template <typename T>
bool close(const std::vector<T> & got, const std::vector<T> & want, float tolerance, const char * what) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(float(got[i] - want[i])) > tolerance * (1.0f + std::fabs(float(want[i])))) {
            std::fprintf(stderr, "%s mismatch at %zu: got %f want %f\n", what, i, float(got[i]), float(want[i]));
            return false;
        }
    }
    return true;
}

struct Context {
    yuuka_device * device = nullptr;
    yuuka_stream * stream = nullptr;
    yuuka_event * event = nullptr;
    bool init() {
        if (yuuka_device_open(0, &device) != YUUKA_OK) {
            std::fprintf(stderr, "SKIP: no HIP device: %s\n", yuuka_last_error());
            return false;
        }
        return check(yuuka_stream_create(device, &stream), "stream") &&
               check(yuuka_event_create(device, &event), "event");
    }
    void shutdown() {
        yuuka_event_destroy(event);
        yuuka_stream_destroy(stream);
        yuuka_device_close(device);
    }
    yuuka_buffer * upload(const void * data, uint64_t bytes) {
        yuuka_buffer * buffer = nullptr;
        if (!check(yuuka_buffer_alloc(device, bytes, &buffer), "alloc") ||
            !check(yuuka_memcpy_h2d(buffer, data, bytes), "upload")) {
            return nullptr;
        }
        return buffer;
    }
    yuuka_buffer * alloc(uint64_t bytes) {
        yuuka_buffer * buffer = nullptr;
        check(yuuka_buffer_alloc(device, bytes, &buffer), "alloc");
        return buffer;
    }
    bool download(yuuka_buffer * buffer, void * dst, uint64_t bytes) {
        return check(yuuka_memcpy_d2h(dst, buffer, bytes), "download");
    }
    yuuka_buffer * run(yuuka_plan * plan, int32_t token, uint32_t pos, int run_tail = 1) {
        if (!check(yuuka_plan_execute(plan, stream, token, pos, run_tail, event), "execute") ||
            !check(yuuka_event_wait(event), "wait")) {
            return nullptr;
        }
        return reinterpret_cast<yuuka_buffer *>(plan);
    }
};

yuuka_op make_op(uint32_t code) {
    yuuka_op op{};
    op.code = code;
    return op;
}

bool test_norms(Context & ctx, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1, 1);
    constexpr int n = 256;
    std::vector<float> src(n), weight(n), want(n);
    for (int i = 0; i < n; ++i) {
        src[i] = dist(rng);
        weight[i] = 0.5f + dist(rng) * 0.25f;
    }
    float sum = 0;
    for (float v : src) sum += double(v) * v;
    const float eps = 1e-6f;
    const float scale = 1.0f / std::sqrt(sum / n + eps);
    for (int i = 0; i < n; ++i) want[i] = src[i] * scale * weight[i];

    auto * src_buf = ctx.upload(src.data(), n * 4);
    auto * weight_buf = ctx.upload(weight.data(), n * 4);
    auto * dst_buf = ctx.alloc(n * 4);
    yuuka_plan * plan = nullptr;
    check(yuuka_plan_create(ctx.device, &plan), "plan");
    yuuka_op op = make_op(YUUKA_OP_RMSNORM);
    op.buf[0] = src_buf;
    op.buf[1] = weight_buf;
    op.buf[2] = dst_buf;
    op.n[0] = n;
    op.f[0] = eps;
    check(yuuka_plan_add(plan, &op), "add rmsnorm");
    ctx.run(plan, 0, 0);
    std::vector<float> got(n);
    ctx.download(dst_buf, got.data(), n * 4);
    bool ok = close(got, want, 1e-4f, "rmsnorm");

    // L2 normalization with an extra scale.
    const float extra = 0.25f;
    for (int i = 0; i < n; ++i) want[i] = src[i] * (1.0f / std::sqrt(sum + eps)) * extra;
    op = make_op(YUUKA_OP_L2NORM);
    op.buf[0] = src_buf;
    op.buf[2] = dst_buf;
    op.n[0] = 1;
    op.n[1] = n;
    op.f[0] = eps;
    op.f[1] = extra;
    yuuka_plan * plan2 = nullptr;
    check(yuuka_plan_create(ctx.device, &plan2), "plan2");
    check(yuuka_plan_add(plan2, &op), "add l2");
    ctx.run(plan2, 0, 0);
    ctx.download(dst_buf, got.data(), n * 4);
    ok = close(got, want, 1e-4f, "l2norm") && ok;

    yuuka_plan_destroy(plan2);
    yuuka_plan_destroy(plan);
    yuuka_buffer_free(dst_buf);
    yuuka_buffer_free(weight_buf);
    yuuka_buffer_free(src_buf);
    return ok;
}

bool test_rope(Context & ctx) {
    constexpr int heads = 2, stride = 64, head_dim = 64, rot = 64;
    const int pos = 7;
    const float base = 10000.0f;
    std::vector<float> data(heads * stride), want;
    for (int i = 0; i < heads * stride; ++i) data[i] = 0.01f * float(i - 32);
    want = data;
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < rot / 2; ++i) {
            const float theta = float(pos) * std::pow(base, -2.0f * i / rot);
            const float c = std::cos(theta), s = std::sin(theta);
            float * p = want.data() + h * stride;
            const float x0 = p[i], x1 = p[i + rot / 2];
            p[i] = x0 * c - x1 * s;
            p[i + rot / 2] = x0 * s + x1 * c;
        }
    }
    auto * buf = ctx.upload(data.data(), data.size() * 4);
    yuuka_plan * plan = nullptr;
    check(yuuka_plan_create(ctx.device, &plan), "plan");
    yuuka_op op = make_op(YUUKA_OP_ROPE);
    op.buf[0] = buf;
    op.n[0] = heads;
    op.n[1] = stride;
    op.n[2] = head_dim;
    op.n[3] = rot;
    op.f[0] = base;
    check(yuuka_plan_add(plan, &op), "add rope");
    ctx.run(plan, 0, pos);
    std::vector<float> got(data.size());
    ctx.download(buf, got.data(), got.size() * 4);
    bool ok = close(got, want, 1e-4f, "rope");
    yuuka_plan_destroy(plan);
    yuuka_buffer_free(buf);
    return ok;
}

bool test_conv(Context & ctx, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    constexpr int channels = 32, kernel = 4;
    std::vector<float> state((kernel - 1) * channels), weights(kernel * channels), qkv(channels);
    for (float & v : state) v = dist(rng);
    for (float & v : weights) v = dist(rng);
    for (float & v : qkv) v = dist(rng);
    std::vector<float> want_out(channels), want_state = state;
    for (int c = 0; c < channels; ++c) {
        float sum = 0;
        for (int t = 0; t < kernel - 1; ++t) sum += weights[c * kernel + t] * state[c * (kernel - 1) + t];
        sum += weights[c * kernel + kernel - 1] * qkv[c];
        want_out[c] = silu(sum);
        for (int t = 0; t < kernel - 2; ++t) want_state[c * (kernel - 1) + t] = state[c * (kernel - 1) + t + 1];
        want_state[c * (kernel - 1) + kernel - 2] = qkv[c];
    }
    auto * state_buf = ctx.upload(state.data(), state.size() * 4);
    auto * weights_buf = ctx.upload(weights.data(), weights.size() * 4);
    auto * qkv_buf = ctx.upload(qkv.data(), qkv.size() * 4);
    auto * out_buf = ctx.alloc(channels * 4);
    yuuka_plan * plan = nullptr;
    check(yuuka_plan_create(ctx.device, &plan), "plan");
    yuuka_op op = make_op(YUUKA_OP_CONV1D);
    op.buf[0] = state_buf;
    op.buf[1] = weights_buf;
    op.buf[2] = qkv_buf;
    op.buf[3] = out_buf;
    op.n[0] = channels;
    op.n[1] = kernel;
    check(yuuka_plan_add(plan, &op), "add conv");
    ctx.run(plan, 0, 0);
    std::vector<float> got_out(channels), got_state(state.size());
    ctx.download(out_buf, got_out.data(), channels * 4);
    ctx.download(state_buf, got_state.data(), state.size() * 4);
    bool ok = close(got_out, want_out, 1e-4f, "conv out") &&
              close(got_state, want_state, 1e-4f, "conv state");
    yuuka_plan_destroy(plan);
    yuuka_buffer_free(out_buf);
    yuuka_buffer_free(qkv_buf);
    yuuka_buffer_free(weights_buf);
    yuuka_buffer_free(state_buf);
    return ok;
}

bool test_gdn(Context & ctx, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    constexpr int groups = 2, heads = 4, s = 16;
    std::vector<float> state(heads * s * s), q(groups * s), k(groups * s), v(heads * s);
    std::vector<float> alpha(heads), beta(heads), dt(heads), a(heads);
    for (float & x : state) x = dist(rng);
    for (float & x : q) x = dist(rng);
    for (float & x : k) x = dist(rng);
    for (float & x : v) x = dist(rng);
    for (int i = 0; i < heads; ++i) {
        alpha[i] = dist(rng);
        beta[i] = dist(rng);
        dt[i] = dist(rng);
        a[i] = -0.1f - std::fabs(dist(rng));
    }
    std::vector<float> want_out(heads * s), want_state = state;
    for (int h = 0; h < heads; ++h) {
        const int kh = h % groups;
        const float raw = alpha[h] + dt[h];
        const float soft = raw > 20 ? raw : std::log1p(std::exp(raw));
        const float decay = std::exp(soft * a[h]);
        const float b = sigmoid(beta[h]);
        float * m = want_state.data() + h * s * s;
        for (int i = 0; i < s * s; ++i) m[i] *= decay;
        float sk[16], d[16];
        for (int j = 0; j < s; ++j) {
            sk[j] = 0;
            for (int i = 0; i < s; ++i) sk[j] += m[i * s + j] * k[kh * s + i];
            d[j] = (v[h * s + j] - sk[j]) * b;
        }
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) m[i * s + j] += k[kh * s + i] * d[j];
        }
        for (int j = 0; j < s; ++j) {
            float sum = 0;
            for (int i = 0; i < s; ++i) sum += m[i * s + j] * q[kh * s + i];
            want_out[h * s + j] = sum;
        }
    }
    auto * state_buf = ctx.upload(state.data(), state.size() * 4);
    auto * q_buf = ctx.upload(q.data(), q.size() * 4);
    auto * k_buf = ctx.upload(k.data(), k.size() * 4);
    auto * v_buf = ctx.upload(v.data(), v.size() * 4);
    auto * alpha_buf = ctx.upload(alpha.data(), alpha.size() * 4);
    auto * beta_buf = ctx.upload(beta.data(), beta.size() * 4);
    auto * dt_buf = ctx.upload(dt.data(), dt.size() * 4);
    auto * a_buf = ctx.upload(a.data(), a.size() * 4);
    auto * out_buf = ctx.alloc(heads * s * 4);
    yuuka_plan * plan = nullptr;
    check(yuuka_plan_create(ctx.device, &plan), "plan");
    yuuka_op op = make_op(YUUKA_OP_GDN);
    op.buf[0] = state_buf;
    op.buf[1] = q_buf;
    op.buf[2] = k_buf;
    op.buf[3] = v_buf;
    op.buf[4] = alpha_buf;
    op.buf[5] = beta_buf;
    op.buf[6] = dt_buf;
    op.buf[7] = a_buf;
    op.buf[8] = out_buf;
    op.n[0] = groups;
    op.n[1] = heads;
    op.n[2] = s;
    check(yuuka_plan_add(plan, &op), "add gdn");
    ctx.run(plan, 0, 0);
    std::vector<float> got_out(heads * s), got_state(state.size());
    ctx.download(out_buf, got_out.data(), heads * s * 4);
    ctx.download(state_buf, got_state.data(), state.size() * 4);
    bool ok = close(got_out, want_out, 2e-3f, "gdn out") &&
              close(got_state, want_state, 2e-3f, "gdn state");
    yuuka_plan_destroy(plan);
    yuuka_buffer_free(out_buf);
    yuuka_buffer_free(a_buf);
    yuuka_buffer_free(dt_buf);
    yuuka_buffer_free(beta_buf);
    yuuka_buffer_free(alpha_buf);
    yuuka_buffer_free(v_buf);
    yuuka_buffer_free(k_buf);
    yuuka_buffer_free(q_buf);
    yuuka_buffer_free(state_buf);
    return ok;
}

bool test_attention(Context & ctx, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1, 1);
    constexpr int heads = 2, kv_heads = 1, d = 8, cap = 8, pos = 3;
    const int kv_dim = kv_heads * d;
    std::vector<float> q(heads * 2 * d), kcache(cap * kv_dim), vcache(cap * kv_dim);
    for (float & x : q) x = dist(rng);
    for (float & x : kcache) x = dist(rng);
    for (float & x : vcache) x = dist(rng);
    std::vector<float> want(heads * d);
    const float scale = 1.0f / std::sqrt(float(d));
    for (int h = 0; h < heads; ++h) {
        float scores[cap];
        float max = -1e30f;
        for (int p = 0; p <= pos; ++p) {
            float sum = 0;
            for (int x = 0; x < d; ++x) sum += q[h * 2 * d + x] * kcache[p * kv_dim + x];
            scores[p] = sum * scale;
            max = std::max(max, scores[p]);
        }
        float total = 0;
        for (int p = 0; p <= pos; ++p) {
            scores[p] = std::exp(scores[p] - max);
            total += scores[p];
        }
        for (int x = 0; x < d; ++x) {
            float sum = 0;
            for (int p = 0; p <= pos; ++p) sum += scores[p] / total * vcache[p * kv_dim + x];
            want[h * d + x] = sum * sigmoid(q[h * 2 * d + d + x]);
        }
    }
    auto * q_buf = ctx.upload(q.data(), q.size() * 4);
    auto * k_buf = ctx.upload(kcache.data(), kcache.size() * 4);
    auto * v_buf = ctx.upload(vcache.data(), vcache.size() * 4);
    auto * out_buf = ctx.alloc(heads * d * 4);
    auto * scores_buf = ctx.alloc(heads * cap * 4);
    yuuka_plan * plan = nullptr;
    check(yuuka_plan_create(ctx.device, &plan), "plan");
    yuuka_op op = make_op(YUUKA_OP_ATTENTION);
    op.buf[0] = q_buf;
    op.buf[1] = k_buf;
    op.buf[2] = v_buf;
    op.buf[3] = out_buf;
    op.buf[4] = scores_buf;
    op.n[0] = kv_heads;
    op.n[1] = heads;
    op.n[2] = d;
    op.n[3] = kv_dim;
    op.u[0] = cap;
    op.f[0] = scale;
    check(yuuka_plan_add(plan, &op), "add attention");
    ctx.run(plan, 0, pos);
    std::vector<float> got(heads * d);
    ctx.download(out_buf, got.data(), got.size() * 4);
    bool ok = close(got, want, 2e-3f, "attention");
    yuuka_plan_destroy(plan);
    yuuka_buffer_free(scores_buf);
    yuuka_buffer_free(out_buf);
    yuuka_buffer_free(v_buf);
    yuuka_buffer_free(k_buf);
    yuuka_buffer_free(q_buf);
    return ok;
}

bool test_embedding(Context & ctx, std::mt19937 & rng) {
    constexpr int cols = 256;
    std::vector<uint8_t> row(210);
    for (int i = 0; i < 128; ++i) row[i] = uint8_t(rng() % 256);
    for (int i = 0; i < 64; ++i) row[128 + i] = uint8_t(rng() % 256);
    for (int i = 0; i < 16; ++i) row[192 + i] = uint8_t(rng() % 256);
    put_half(row.data() + 208, 0.01f);
    std::vector<uint8_t> weights(2 * 210);
    std::memcpy(weights.data(), row.data(), 210);
    auto * weight_buf = ctx.upload(weights.data(), weights.size());
    auto * out_buf = ctx.alloc(cols * 4);
    yuuka_plan * plan = nullptr;
    check(yuuka_plan_create(ctx.device, &plan), "plan");
    yuuka_op op = make_op(YUUKA_OP_EMBEDDING);
    op.buf[0] = weight_buf;
    op.buf[1] = out_buf;
    op.n[0] = cols;
    op.n[1] = 210;
    op.u[0] = YUUKA_DTYPE_Q6_K;
    check(yuuka_plan_add(plan, &op), "add embedding");
    ctx.run(plan, 0, 0);
    std::vector<float> got(cols);
    ctx.download(out_buf, got.data(), cols * 4);
    // Dequantize the same row on the host.
    std::vector<float> want(cols);
    const uint8_t * ql = row.data();
    const uint8_t * qh = row.data() + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(row.data() + 192);
    const float d = half_to_float(uint16_t(row[208] | (row[209] << 8)));
    for (int n = 0; n < 2; ++n) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int q1 = int8_t((ql[n * 64 + l] & 0xF) | (((qh[n * 32 + l] >> 0) & 3) << 4)) - 32;
            int q2 = int8_t((ql[n * 64 + l + 32] & 0xF) | (((qh[n * 32 + l] >> 2) & 3) << 4)) - 32;
            int q3 = int8_t((ql[n * 64 + l] >> 4) | (((qh[n * 32 + l] >> 4) & 3) << 4)) - 32;
            int q4 = int8_t((ql[n * 64 + l + 32] >> 4) | (((qh[n * 32 + l] >> 6) & 3) << 4)) - 32;
            want[n * 128 + l] = d * sc[n * 8 + is + 0] * q1;
            want[n * 128 + l + 32] = d * sc[n * 8 + is + 2] * q2;
            want[n * 128 + l + 64] = d * sc[n * 8 + is + 4] * q3;
            want[n * 128 + l + 96] = d * sc[n * 8 + is + 6] * q4;
        }
    }
    bool ok = close(got, want, 1e-5f, "embedding");
    yuuka_plan_destroy(plan);
    yuuka_buffer_free(out_buf);
    yuuka_buffer_free(weight_buf);
    return ok;
}

} // namespace

int main() {
    Context ctx;
    if (!ctx.init()) {
        return 77;
    }
    std::mt19937 rng(7);
    bool ok = true;
    ok = test_norms(ctx, rng) && ok;
    ok = test_rope(ctx) && ok;
    ok = test_conv(ctx, rng) && ok;
    ok = test_gdn(ctx, rng) && ok;
    ok = test_attention(ctx, rng) && ok;
    ok = test_embedding(ctx, rng) && ok;
    ctx.shutdown();
    return ok ? 0 : 1;
}

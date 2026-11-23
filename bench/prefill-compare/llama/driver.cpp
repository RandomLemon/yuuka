#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>

namespace {

bool parse_int(const char * arg, int & value) {
    const char * end = arg + std::strlen(arg);
    const auto result = std::from_chars(arg, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

int fail(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    return 1;
}

int run(const char * model_path, const char * token_path, int prompt, int chunk,
        int warmups, int repetitions, const char * logit_path, int decode) {
    std::vector<llama_token> tokens(static_cast<size_t>(prompt) + static_cast<size_t>(decode));
    std::ifstream input(token_path, std::ios::binary | std::ios::ate);
    const auto file_size = input.tellg();
    const auto bytes = static_cast<std::streamsize>(tokens.size() * sizeof(tokens[0]));
    if (file_size < bytes || file_size % sizeof(tokens[0]) != 0 ||
        !input.seekg(0) || !input.read(reinterpret_cast<char *>(tokens.data()), bytes)) {
        return fail("could not read the requested token prefix");
    }
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path, model_params), llama_model_free);
    if (!model) return fail("could not load the model");
    const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    for (llama_token token : tokens) {
        if (token < 0 || token >= vocab_size) return fail("token ID outside model vocabulary");
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = decode > 0 ? prompt + decode : prompt + 64;
    params.n_batch = chunk;
    params.n_ubatch = chunk;
    params.n_seq_max = 1;
    params.type_k = GGML_TYPE_F16;
    params.type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), params), llama_free);
    if (!context) return fail("could not initialize inference context");

    std::vector<llama_pos> positions(prompt);
    for (int i = 0; i < prompt; ++i) positions[i] = i;
    std::vector<int8_t> output_rows(prompt, 0);
    output_rows.back() = 1;
    std::vector<llama_batch> batches;
    for (int start = 0; start < prompt;) {
        const int rows = std::min(chunk, prompt - start);
        batches.push_back({rows, tokens.data() + start, nullptr, positions.data() + start,
                           nullptr, nullptr, output_rows.data() + start});
        start += rows;
    }
    std::vector<long long> samples;
    std::vector<long long> decode_samples;
    const float * logits = nullptr;
    for (int rep = -warmups; rep < repetitions; ++rep) {
        llama_memory_clear(llama_get_memory(context.get()), false);
        llama_synchronize(context.get());
        const auto start = std::chrono::steady_clock::now();
        for (const auto & batch : batches) {
            if (llama_decode(context.get(), batch) != 0) return fail("llama_decode failed");
        }
        // Fetching a row waits until the asynchronous device-to-host copy of
        // the requested logits completes.
        logits = llama_get_logits_ith(context.get(), -1);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (logits == nullptr) return fail("missing final logits");
        if (rep >= 0) {
            samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        }
        if (decode > 0) {
            const auto decode_start = std::chrono::steady_clock::now();
            for (int j = 0; j < decode; ++j) {
                llama_token token = tokens[static_cast<size_t>(prompt) + static_cast<size_t>(j)];
                llama_pos position = static_cast<llama_pos>(prompt) + static_cast<llama_pos>(j);
                int32_t n_seq_id = 1;
                llama_seq_id seq_id = 0;
                llama_seq_id * seq_ids[] = {&seq_id};
                int8_t output = 1;
                llama_batch batch{1, &token, nullptr, &position, &n_seq_id, seq_ids, &output};
                if (llama_decode(context.get(), batch) != 0) return fail("llama_decode failed");
                logits = llama_get_logits_ith(context.get(), -1);
                if (logits == nullptr) return fail("missing decode logits");
            }
            const auto decode_elapsed = std::chrono::steady_clock::now() - decode_start;
            if (rep >= 0) {
                decode_samples.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(decode_elapsed).count());
            }
        }
    }
    if (logits == nullptr) return fail("missing final logits rows");
    int top = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (!std::isfinite(logits[i])) return fail("nonfinite output logit");
        if (logits[i] > logits[top]) top = i;
    }
    std::ofstream output(logit_path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(logits),
                 static_cast<std::streamsize>(vocab_size) * sizeof(float));
    output.close();
    if (!output) return fail("could not write the final logit row");
    std::printf("{\"prompt\":%d,\"chunk\":%d,\"warmup\":%d,\"top1\":%d,\"samples_ns\":[",
                prompt, chunk, warmups, top);
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%lld", i == 0 ? "" : ",", samples[i]);
    }
    if (decode > 0) {
        std::printf("],\"decode\":%d,\"decode_samples_ns\":[", decode);
        for (size_t i = 0; i < decode_samples.size(); ++i) {
            std::printf("%s%lld", i == 0 ? "" : ",", decode_samples[i]);
        }
    }
    std::puts("]}");
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    int prompt = 0, chunk = 0, warmups = 0, repetitions = 0, decode = 0;
    if ((argc != 8 && argc != 9) || !parse_int(argv[3], prompt) || !parse_int(argv[4], chunk) ||
        !parse_int(argv[5], warmups) || !parse_int(argv[6], repetitions) ||
        prompt <= 0 || prompt > std::numeric_limits<int>::max() - 64 ||
        (argc == 9 && (!parse_int(argv[8], decode) || decode < 0 ||
                       decode > std::numeric_limits<int>::max() - prompt)) ||
        chunk <= 0 || warmups < 0 || repetitions <= 0) {
        std::fprintf(stderr,
                     "usage: %s MODEL TOKEN_FILE PROMPT CHUNK WARMUP REPS LOGITS [DECODE]\n",
                     argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    llama_backend_init();
    const int status = run(argv[1], argv[2], prompt, std::min(chunk, prompt),
                           warmups, repetitions, argv[7], decode);
    llama_backend_free();
    return status;
}

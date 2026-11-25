#include "rknn_api.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
struct Engine {
    rknn_context ctx{};
    bool initialized = false;
    size_t input_bytes = 0;
    size_t output_count = 0;

    ~Engine() {
        if (initialized) rknn_destroy(ctx);
    }
};

void check(int rc, const char* name) {
    if (rc != RKNN_SUCC)
        throw std::runtime_error(std::string(name) + " returned " + std::to_string(rc));
}

void error_message(char* error, size_t capacity, const char* message) {
    if (error && capacity) std::snprintf(error, capacity, "%s", message);
}
}  // namespace

extern "C" int jelly_rknn_create(const char* model, int core, void** handle,
                                  size_t* input_bytes, size_t* output_count,
                                  char* error, size_t error_capacity) {
    try {
        if (!model || !handle || !input_bytes || !output_count ||
            (core != -1 && core != 0 && core != 1 && core != 2))
            throw std::runtime_error("invalid model, output pointer or core");
        *handle = nullptr;
        auto engine = std::make_unique<Engine>();
        check(rknn_init(&engine->ctx, const_cast<char*>(model), 0, 0, nullptr), "rknn_init");
        engine->initialized = true;
        rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
        if (core == 0) mask = RKNN_NPU_CORE_0;
        if (core == 1) mask = RKNN_NPU_CORE_1;
        if (core == 2) mask = RKNN_NPU_CORE_2;
        check(rknn_set_core_mask(engine->ctx, mask), "rknn_set_core_mask");

        rknn_input_output_num io{};
        check(rknn_query(engine->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)),
              "RKNN_QUERY_IN_OUT_NUM");
        if (io.n_input != 1 || io.n_output != 1)
            throw std::runtime_error("model must have one input and one output");
        rknn_tensor_attr input{};
        input.index = 0;
        check(rknn_query(engine->ctx, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)),
              "RKNN_QUERY_INPUT_ATTR");
        rknn_tensor_attr output{};
        output.index = 0;
        check(rknn_query(engine->ctx, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)),
              "RKNN_QUERY_OUTPUT_ATTR");
        engine->input_bytes = input.n_elems;
        engine->output_count = output.n_elems;
        if (!engine->input_bytes || engine->input_bytes > (1u << 26) ||
            !engine->output_count || engine->output_count > 4096)
            throw std::runtime_error("invalid model tensor size");
        *input_bytes = engine->input_bytes;
        *output_count = engine->output_count;
        *handle = engine.release();
        return 0;
    } catch (const std::exception& e) {
        error_message(error, error_capacity, e.what());
        return -1;
    } catch (...) {
        error_message(error, error_capacity, "unknown C++ error");
        return -1;
    }
}

extern "C" int jelly_rknn_infer(void* handle, const uint8_t* rgb, size_t rgb_bytes,
                                 float* logits, size_t logits_capacity,
                                 char* error, size_t error_capacity) {
    try {
        auto* engine = static_cast<Engine*>(handle);
        if (!engine || !rgb || !logits || rgb_bytes != engine->input_bytes ||
            logits_capacity < engine->output_count)
            throw std::runtime_error("invalid inference input or output size");
        rknn_input input{};
        input.index = 0;
        input.buf = const_cast<uint8_t*>(rgb);
        input.size = static_cast<uint32_t>(rgb_bytes);
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.pass_through = 0;
        check(rknn_inputs_set(engine->ctx, 1, &input), "rknn_inputs_set");
        check(rknn_run(engine->ctx, nullptr), "rknn_run");
        rknn_output output{};
        output.index = 0;
        output.want_float = 1;
        check(rknn_outputs_get(engine->ctx, 1, &output, nullptr), "rknn_outputs_get");
        // Once get succeeds, release the runtime allocation even if validation fails.
        const bool valid = output.buf &&
                           output.size >= engine->output_count * sizeof(float);
        if (valid)
            std::memcpy(logits, output.buf, engine->output_count * sizeof(float));
        check(rknn_outputs_release(engine->ctx, 1, &output), "rknn_outputs_release");
        if (!valid) throw std::runtime_error("RKNN output is smaller than expected");
        return 0;
    } catch (const std::exception& e) {
        error_message(error, error_capacity, e.what());
        return -1;
    } catch (...) {
        error_message(error, error_capacity, "unknown C++ error");
        return -1;
    }
}

extern "C" void jelly_rknn_destroy(void* handle) {
    delete static_cast<Engine*>(handle);
}

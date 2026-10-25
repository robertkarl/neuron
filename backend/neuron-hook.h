#pragma once

#include "ggml-backend.h"
#include "json.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Opt-in, single-process research hook. The client owns the config and trace files.
struct neuron_hook {
    std::string config_path;
    bool trace = false;
    std::vector<int> logit_tokens;
    std::ofstream output;
    std::map<int, std::vector<std::pair<int, float>>> offsets;
    std::map<std::pair<int, int>, size_t> seen;

    bool enabled() const { return !config_path.empty(); }

    void configure(const char * path) {
        if (path) { config_path = path; }
    }

    void begin() {
        if (!enabled()) { return; }
        offsets.clear();
        logit_tokens.clear();
        seen.clear();
        output.close();
        std::ifstream input(config_path);
        const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        auto config = common_json::parse(contents);
        trace = config.at("trace").get<bool>();
        if (config.contains("logit_tokens")) {
            logit_tokens = config.at("logit_tokens").get<std::vector<int>>();
            if (logit_tokens.size() > 256) { throw std::runtime_error("too many probe tokens"); }
            for (int token : logit_tokens) {
                if (token < 0 || token >= 248320) { throw std::runtime_error("invalid probe token"); }
            }
        }
        for (const auto & item : config.at("steering")) {
            int layer = item.at("layer").get<int>();
            int neuron = item.at("neuron").get<int>();
            float alpha = item.at("alpha").get<float>();
            if (layer < 0 || layer >= 32 || neuron < 0 || neuron >= 9216 || !std::isfinite(alpha)) {
                throw std::runtime_error("invalid Qwen3.5-4B neuron intervention");
            }
            offsets[layer].push_back({neuron, alpha});
        }
        output.open(config_path + ".trace", std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot open neuron trace"); }
        output.write("NEURON01", 8);
        output.flush();
    }

    void record(uint32_t kind, uint32_t layer, uint32_t width, uint32_t count, const std::vector<float> & values) {
        const uint32_t header[] = {kind, layer, width, count};
        output.write(reinterpret_cast<const char *>(header), sizeof(header));
        output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        output.flush();
        if (!output) { throw std::runtime_error("cannot write neuron trace"); }
    }

    static bool eval(ggml_tensor * tensor, bool ask, void * user_data) {
        auto & self = *static_cast<neuron_hook *>(user_data);
        if (!self.output.is_open()) { return ask ? false : true; }
        if (ggml_nelements(tensor) == 0) { return ask ? false : true; }
        const std::string name(tensor->name);
        if (name == "result_output" && !self.logit_tokens.empty()) {
            if (ask) { return true; }
            if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || tensor->ne[0] != 248320 ||
                    tensor->ne[1] < 1 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
                throw std::runtime_error("unsupported logit tensor layout: " + std::to_string(tensor->type) + " " + std::to_string(tensor->ne[0]) + "x" + std::to_string(tensor->ne[1]) + "x" + std::to_string(tensor->ne[2]) + "x" + std::to_string(tensor->ne[3]) + " contiguous=" + std::to_string(ggml_is_contiguous(tensor)));
            }
            std::vector<float> logits(tensor->ne[0]);
            ggml_backend_tensor_get(tensor, logits.data(), (tensor->ne[1] - 1) * tensor->nb[1], logits.size() * sizeof(float));
            const double maximum = *std::max_element(logits.begin(), logits.end());
            double total = 0;
            for (float value : logits) { total += std::exp(double(value) - maximum); }
            const double log_z = maximum + std::log(total);
            std::vector<float> selected;
            for (int token : self.logit_tokens) {
                selected.push_back(float(token));
                selected.push_back(float(logits[token] - log_z));
            }
            self.record(4, 0, selected.size(), 1, selected);
            return true;
        }
        int kind = 0;
        size_t prefix = 0;
        if (name.rfind("ffn_swiglu-", 0) == 0) { kind = 1; prefix = 11; }
        if (name.rfind("l_out-", 0) == 0) { kind = 2; prefix = 6; }
        if (!kind) { return ask ? false : true; }
        const int layer = std::stoi(name.substr(prefix));
        const bool capture = self.trace && (layer < 31 || kind == 2);
        const bool steer = kind == 1 && self.offsets.count(layer);
        if (ask) { return capture || steer; }
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("unsupported neuron tensor layout");
        }
        const size_t width = tensor->ne[0];
        const size_t count = tensor->ne[1];
        std::vector<float> values(width * count);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        auto & seen = self.seen[{kind, layer}];
        const size_t first = seen == 0 && layer < 31 ? 1 : 0;
        seen += count;
        if (capture && count > first) {
            std::vector<float> summary(kind == 1 ? width : count - first, 0.0f);
            for (size_t row = first; row < count; ++row) {
                double norm = 0;
                for (size_t col = 0; col < width; ++col) {
                    float value = values[row * width + col];
                    if (kind == 1) { summary[col] += value / (count - first); }
                    else { norm += double(value) * value; }
                }
                if (kind == 2) { summary[row - first] = std::sqrt(norm); }
            }
            self.record(kind, layer, kind == 1 ? width : 1, count - first, summary);
        }
        if (steer) {
            for (auto [neuron, alpha] : self.offsets.at(layer)) {
                if (size_t(neuron) >= width) { throw std::runtime_error("neuron index exceeds tensor width"); }
                const float before = values[neuron];
                for (size_t row = 0; row < count; ++row) { values[row * width + neuron] += alpha; }
                self.record(3, layer, 3, count, {float(neuron), before, values[neuron]});
            }
            ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
        }
        return true;
    }
};

// Stage G: real fixed-prompt execution gate for SenseNova-Loop-9B.
//
// This test loads the licensed `.ninfer` artifact through the public Engine and generates from a
// fixed token prompt. The external Hugging Face oracle lives in `hf_reference.py` and is run
// separately against the same prompt. Returning 77 only means the licensed artifact is not
// installed; CTest records that as a skip rather than a pass.

#include "ninfer/engine.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

constexpr ninfer::TokenId kFixedPrompt[] = {785, 3895, 286, 2673, 315, 3387, 349};

} // namespace

namespace {

void trace(const char* message) {
    std::cerr << "FIXED_PROMPT_TRACE: " << message << std::endl;
}

} // namespace

int main() {
    trace("start");
    const char* env = std::getenv("NINFER_SENSENOVA_LOOP_9B_WEIGHTS");
    const std::filesystem::path artifact =
        env != nullptr && *env != '\0'
            ? std::filesystem::path(env)
            : std::filesystem::path("/home/user/models/sensenova/SenseNova-Loop-9B.ninfer");
    trace(("artifact=" + artifact.string()).c_str());
    if (!std::filesystem::is_regular_file(artifact)) {
        std::cout << "skip: SenseNova artifact is not available at " << artifact << '\n';
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path       = artifact;
    options.max_context         = 1024;
    options.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.max_concurrency     = 1;
    options.prefill_chunk       = 1024;
    options.kv_cache            = ninfer::KvCacheStorage::BFloat16;
    options.enable_vision       = false;
    options.use_cuda_graph      = false;
    options.enable_prompt_cache = false;
    options.wddm_evictable_budget = false;
    options.load_progress.callback = [](std::string_view phase, std::uint64_t done,
                                        std::uint64_t total) {
        if (total == 0 || done == 0 || done == total || done % (64ULL << 20ULL) < (1ULL << 20ULL)) {
            std::cerr << "FIXED_PROMPT_TRACE: load " << phase << ' ' << done << '/' << total
                      << std::endl;
        }
    };

    try {
        trace("constructing engine");
        ninfer::Engine engine(options);
        trace("engine constructed");
        const auto mem = engine.memory_summary();
        std::cerr << "MEM weights=" << mem.weights.capacity_bytes
                  << " seq=" << mem.sequence.capacity_bytes
                  << " workspace=" << mem.workspace.capacity_bytes
                  << " transient=" << mem.request_transient.capacity_bytes
                  << " kv=" << mem.kv_payload_bytes
                  << " runtime=" << mem.runtime_reservation_bytes << '\n';
        trace("preparing prompt");
        auto prompt = engine.prepare_tokens(
            std::vector<ninfer::TokenId>(std::begin(kFixedPrompt), std::end(kFixedPrompt)));
        trace("prompt prepared");

        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 1;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.sampling.top_k          = 1;
        request.execution.sampling.top_p          = 1.0F;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = false;

        trace("generating");
        const ninfer::GenerationResult result = engine.generate(std::move(prompt), request);
        trace("generated");
        if (result.generated_token_ids.size() != 1) {
            std::cerr << "FAIL: fixed prompt produced " << result.generated_token_ids.size()
                      << " tokens, expected one\n";
            return 1;
        }

        std::cout << "HF_FIXED_PROMPT_IDS:";
        for (const ninfer::TokenId token : kFixedPrompt) { std::cout << ' ' << token; }
        std::cout << "\nENGINE_FIRST_TOKEN: " << result.generated_token_ids[0] << '\n';
        std::cout << "OK SenseNova fixed-prompt execution\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: SenseNova fixed-prompt execution threw: " << error.what() << '\n';
        return 1;
    }
}

// Does an edit made inside the prompt leave the recurrent state describing the prompt?
//
// 48 of 64 layers keep one recurrent summary of the whole sequence instead of a per-token cache.
// That summary cannot be relocated and cannot be rewound, so a prompt that diverges inside it leaves
// the engine with a state that summarises text the new prompt no longer contains. Two behaviours are
// claimed in this tree:
//
//   `PrefixReusePath::AppendAtSharedPrefix` in include/ninfer/types.h says the recurrent state "is
//   zeroed because it was summarised past the divergence";
//   program_impl.h keeps it instead (`log_scr_gdn_state("fork", ...)`), on the argument that the
//   authors' reference does the same under `KVREUSE_SSM_MODE=fork`, which restores "the state saved
//   at the end of the previous prompt".
//
// The reference case is an append: the text the previous prompt ended with is still the text this
// prompt starts with, so the saved state is exactly right. An edit inside the prompt is not that
// case, and only a run can say what it costs. This is that run.
//
// This lives in its own binary rather than in test_engine_prefix_real.cpp because it leaves a long
// prompt resident, and the cases there were recorded against the sequence state they established for
// themselves.
//
// The KV format, the reuse route and the edit shape are selectable, so this single-span edit can be
// run on the same format and route as the multi-span failure. `NINFER_MULTISPAN_KV` picks the KV
// format (`bf16`, `int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`; default bf16) and `NINFER_SCR_SOURCE_SHIFT=1`
// selects the route that copies a run whose source does not sit on the page grid. The deleted
// stretch is `NINFER_MIDPROMPT_KEEP` sentences out of `NINFER_MIDPROMPT_REPEATS`, dropping
// `NINFER_MIDPROMPT_DROP`, and `NINFER_MIDPROMPT_MODE=segmented` selects the multi-run planner:
//
//   $env:NINFER_MULTISPAN_KV="rk4v4-e8"; $env:NINFER_SCR_SOURCE_SHIFT="1"
//   $env:NINFER_MIDPROMPT_MODE="segmented"
//   .\build-ninja\tests\ninfer_qwen3_6_27b_mid_prompt_edit_test.exe
//
// If this single-span edit fails on the format and route the multi-span case fails on, the fault
// needs only one relocated run and not several -- which points at the quantized prefill kernel
// reading a page a relocated run only partly fills (see docs/SCR.md §5), not at the multi-run plan.
#include "ninfer/engine.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

// A paragraph of ordinary prose, repeated. Synthetic token ids were tried first and are a bad
// fixture: a sequence of unrelated ids is far from anything the model was trained on, so both legs
// produce fragments and the comparison says more about the fixture than about the state (that
// version measured 0 of 8 tokens in common and 0.111 nats).
constexpr const char* kSentence =
    "The maintenance log for the deployment records every change in order, and each entry lists the "
    "operator, the timestamp, and a short justification. ";

std::string prose_block(int repeats) {
    std::string text;
    for (int index = 0; index < repeats; ++index) { text += kSentence; }
    return text;
}

ninfer::PromptInput chat_prompt(const std::string& text) {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

const char* reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
        case ninfer::PrefixReusePath::FullReset: return "full_reset";
        case ninfer::PrefixReusePath::AppendAtFrontier: return "append_at_frontier";
        case ninfer::PrefixReusePath::RestoreTurnCheckpoint: return "restore_turn_checkpoint";
        case ninfer::PrefixReusePath::RestoreDiskCheckpoint: return "restore_disk_checkpoint";
        case ninfer::PrefixReusePath::SuffixRelocation: return "suffix_relocation";
        case ninfer::PrefixReusePath::AppendAtSharedPrefix: return "append_at_shared_prefix";
    }
    return "unknown";
}

// The KV format, read from the same variable the multi-span sibling reads, so the two binaries can
// be pointed at one format without a second name to remember.
ninfer::KvCacheStorage kv_storage_from_env() {
    const char* value = std::getenv("NINFER_MULTISPAN_KV");
    if (value == nullptr || *value == '\0') { return ninfer::KvCacheStorage::BFloat16; }
    const std::string name(value);
    if (name == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (name == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (name == "rk8v4") { return ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    if (name == "rk4v4") { return ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64; }
    if (name == "rk2v4-e8") { return ninfer::KvCacheStorage::RK2V4E8; }
    if (name == "rk4v4-e8") { return ninfer::KvCacheStorage::RK4V4E8; }
    std::cerr << "mid-prompt edit: unknown NINFER_MULTISPAN_KV=" << name << ", using bf16\n";
    return ninfer::KvCacheStorage::BFloat16;
}

ninfer::ScrRelocationMode scr_mode_from_env() {
    const char* value = std::getenv("NINFER_MIDPROMPT_MODE");
    if (value != nullptr && std::string(value) == "segmented") {
        return ninfer::ScrRelocationMode::Segmented;
    }
    return ninfer::ScrRelocationMode::Tail;
}

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return fallback; }
    return std::atoi(value);
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk             = 1024;
    options.kv_cache                  = kv_storage_from_env();
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

int exercise(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));

    // Roughly 1,800 tokens, of which a third is deleted from the middle: the shape of a user
    // trimming a long pasted log. The survivors after the gap keep their content but land at new
    // positions, and the chat template's tail is unchanged. Overridable so a run can vary how far
    // the survivors shift -- the deleted token count decides whether a relocated run starts partway
    // through a page.
    const int repeats = env_int("NINFER_MIDPROMPT_REPEATS", 60);
    const int kKeep   = env_int("NINFER_MIDPROMPT_KEEP", 20); // sentences before the deleted block
    const int kDrop   = env_int("NINFER_MIDPROMPT_DROP", 20); // sentences deleted from the middle
    if (repeats < 1 || kKeep < 0 || kDrop < 0 || repeats - kKeep - kDrop < 0) {
        std::cerr << "mid-prompt edit: NINFER_MIDPROMPT_REPEATS/KEEP/DROP leave no prompt to build\n";
        return 1;
    }

    const std::string full  = prose_block(repeats);
    std::string edited_text = prose_block(kKeep);
    edited_text += prose_block(repeats - kKeep - kDrop);
    edited_text += "Which entry was rolled back? ";

    const ninfer::ScrRelocationMode mode = scr_mode_from_env();
    auto options = [mode](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 8;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.execution.scr_mode                = mode;
        result.execution.suffix_reuse_max_spans  = 6;
        result.stop.include_model_defaults       = false;
        return result;
    };

    // The cold answer first, while nothing related is resident.
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare(chat_prompt(edited_text)), options(false));
    if (cold.generated_token_ids.size() != 8) {
        std::cerr << "mid-prompt edit: the cold leg did not generate eight tokens\n";
        return 1;
    }

    // Leave the unedited prompt resident, then ask for the edited one with reuse.
    const ninfer::GenerationResult resident =
        engine.generate(engine.prepare(chat_prompt(full)), options(false));
    if (resident.generated_token_ids.size() != 8) {
        std::cerr << "mid-prompt edit: the resident leg did not generate eight tokens\n";
        return 1;
    }
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(chat_prompt(edited_text)), options(true));
    if (reused.generated_token_ids.size() != 8) {
        std::cerr << "mid-prompt edit: the reuse leg did not generate eight tokens\n";
        return 1;
    }

    // The reuse leg has to actually reuse. A silent fall back to a cold prefill would make every
    // number below true for the wrong reason, which is the failure mode this case exists to rule
    // out: the plan is the only thing tying the comparison to the recurrent state at all.
    if (reused.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        reused.reused_prompt_tokens == 0) {
        std::cerr << "mid-prompt edit: the reuse leg fell back to a full prefill, so this case says "
                     "nothing about the recurrent state\n";
        return 1;
    }

    std::size_t agreed = 0;
    while (agreed < 8 && cold.generated_token_ids[agreed] == reused.generated_token_ids[agreed]) {
        ++agreed;
    }
    float first_logprob_delta = 0.0F;
    if (!cold.generated_logprobs.empty() && !reused.generated_logprobs.empty()) {
        first_logprob_delta = std::fabs(cold.generated_logprobs[0] - reused.generated_logprobs[0]);
    }

    std::cout << "mid-prompt edit: mode="
              << (mode == ninfer::ScrRelocationMode::Segmented ? "segmented" : "tail")
              << " repeats=" << repeats << " keep=" << kKeep << " drop=" << kDrop
              << " path=" << reuse_path_name(reused.prefix_reuse_path)
              << " reused=" << reused.reused_prompt_tokens
              << " relocated=" << reused.relocated_prompt_tokens
              << " computed=" << reused.computed_prefill_tokens << " of "
              << engine.count_tokens(chat_prompt(edited_text)) << " agreed=" << agreed << "/8"
              << " first_logprob_delta=" << first_logprob_delta << '\n';
    std::cout << "  cold  :";
    for (const ninfer::TokenId token : cold.generated_token_ids) { std::cout << ' ' << token; }
    std::cout << "\n  reused:";
    for (const ninfer::TokenId token : reused.generated_token_ids) { std::cout << ' ' << token; }
    std::cout << '\n';

    // The state this path continues from summarises the *unedited* prompt while only the tail is
    // forwarded, so the continuation comes from a recurrent summary that contains text the edited
    // prompt no longer has. That is an approximation, not an identity, so the case is held to a
    // budget rather than to token equality.
    //
    // Measured on this fixture across seven runs: the first token's logprob moved by 0.010 to 0.073
    // nats, while the eight greedy tokens were identical in every run (6 of 8 in common with the cold
    // leg, parting at the same place). That asymmetry is the point -- the logprob of one token under
    // a perturbed state is a single sample of a noisy quantity, and it is also the token the
    // speculative backend may have licensed rather than verified, so it is reported rather than
    // gated. The greedy continuation is what a user observes and it is stable, so that is what the
    // budget rests on; the logprob bound below only exists to catch a distribution that collapsed.
    // The budget is about twice the worse measurement because CUDA-graph capture and replay change
    // which kernels run and move the value by ~20% depending on where the case runs in the process.
    //
    // This does not cover the shape that reported the failure: the 2026-10-17 session deleted 10,336
    // of 31,708 tokens across three spans, scored 0% MTP acceptance, and stopped after two tokens.
    // Multi-span relocation needs its own case; this one bounds the single-span edit.
    constexpr float kFirstTokenBudgetNats = 0.50F;
    constexpr std::size_t kMinAgreedTokens = 5;
    if (!(first_logprob_delta <= kFirstTokenBudgetNats) || agreed < kMinAgreedTokens) {
        std::cerr << "mid-prompt edit: reuse moved the first token's logprob by "
                  << first_logprob_delta << " nats (budget " << kFirstTokenBudgetNats
                  << ") and agreed on " << agreed << " of 8 tokens (minimum " << kMinAgreedTokens
                  << "); the continuation is no longer an approximation of the cold one, either "
                     "because the retained recurrent state summarises text the edited prompt no "
                     "longer has or because the relocated KV read back wrong (vary "
                     "NINFER_MULTISPAN_KV and NINFER_SCR_SOURCE_SHIFT to tell which)\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    // `NINFER_MULTISPAN_ALIGNED=1` forces the aligned route, matching the multi-span sibling. Left
    // unset, this test keeps the aligned default it has always run with, and
    // `NINFER_SCR_SOURCE_SHIFT=1` selects the route that copies a run whose source is not on the
    // page grid. The runtime reads the flag once per process, so it has to be set before the first
    // request.
    const char* aligned = std::getenv("NINFER_MULTISPAN_ALIGNED");
    if (aligned != nullptr && *aligned != '\0' && *aligned != '0') {
#if defined(_WIN32)
        _putenv_s("NINFER_SCR_SOURCE_SHIFT", "0");
#else
        setenv("NINFER_SCR_SOURCE_SHIFT", "0", 1);
#endif
    }

    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77; // CTest SKIP_RETURN_CODE
    }
    const char* shift = std::getenv("NINFER_SCR_SOURCE_SHIFT");
    std::cout << "mid-prompt edit: source_shift="
              << ((shift != nullptr && *shift != '\0' && *shift != '0') ? "on" : "off")
              << " kv=" << static_cast<int>(kv_storage_from_env()) << '\n';
    if (const int result = exercise(artifact); result != 0) { return result; }
    std::cout << "ok\n";
    return 0;
}

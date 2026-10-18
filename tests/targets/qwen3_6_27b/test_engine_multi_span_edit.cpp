// What a *multi-span* mid-prompt edit costs, measured against the single-span case.
//
// `test_engine_mid_prompt_edit.cpp` bounds one deleted region of ~600 tokens in an ~1,800-token
// prompt. The 2026-10-17 live session reported a different shape entirely: 10,336 of 31,708 tokens
// deleted across *three* surviving spans, which scored 0% MTP acceptance and stopped after two
// tokens. That run's own analysis could not tell whether the collapse was the multi-span planner or
// the retained recurrent state, because no case ever varied the two independently.
//
// This binary supplies the missing corners of a 2x2, all as one process so the recurrence is the
// same and only the edit differs:
//
//           1 span                 3 spans
//   small   (test_engine_mid_       case B
//           prompt_edit.cpp)
//   large   case A                 case C
//
// * A: one contiguous ~10k-token deletion  -> isolates the retained state at scale.
// * B: three small deletions (~2-3k total)  -> isolates the multi-span plan/apply at small scale.
// * C: three ~3.5k deletions (~10k total)   -> the req-7 shape.
//
// The read-out is the same one the single-span case uses: the greedy continuation of a cold prefill
// versus the reused leg, the first-token logprob delta, and the MTP drafted/accepted counts. The
// cases are held only to "the reuse leg actually relocated something", because a silent fall back to
// a cold prefill would make every number true for the wrong reason. A numeric budget can be imposed
// without editing the source by setting NINFER_MULTISPAN_MIN_AGREED / NINFER_MULTISPAN_FIRST_NAT.
//
// The source-shift path is what the live failure used, so it is the default here; exporting
// NINFER_SCR_SOURCE_SHIFT=0 (or running with it unset in the environment and setting
// NINFER_MULTISPAN_ALIGNED=1) selects the aligned re-pointing route instead. The two are separate
// processes because the flag is latched on first use.
//
// Measured at 228e10b5 on the host, in five of the six KV formats (rk2v4-e8 was not run), on the
// route that copies a kept run into new pages: every case matched a cold run on all eight generated
// tokens, and the first token's log-probability moved by 0.002 to 0.061 nats. Ten repeats of the
// one-deep-cut case on rk4v4-e8 stayed between 0.021 and 0.046 nats, so that is the spread to budget
// above.
//
// An earlier version of this comment reported the quantized formats failing on this route by 0.2 to
// 1.1 nats and blamed the quantized prefill kernel. That run predated 228e10b5, which rewrote the
// quantized decode kernel (a query tile that was being overwritten, plus a missing barrier), and the
// failure does not reproduce. The reasoning behind it did not hold either: bf16 uses a different
// decode kernel as well as a different prefill one, so the comparison could not separate the two.
//
// The count alone is still not something to gate on -- prefill is not reproducible
// (test_engine_prefill_determinism) -- so read `first_logprob_delta`, and use
// NINFER_MULTISPAN_MIN_AGREED / NINFER_MULTISPAN_FIRST_NAT to impose a budget. A budget of 8 agreed
// and 0.12 nats clears the observed spread and still catches the old failure.
//
// `NINFER_MULTISPAN_CASE` selects one case by letter or name (`A`, `A-one-span-large`, ...) so the
// one-span case can be run on its own; unset runs all three. Case A deletes one contiguous stretch
// into a single surviving run, so it is the single-span probe.
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kSections        = 40;
constexpr int kFillerRepeats   = 24;

// One whole section, unique in the document so the detector has exactly one place to anchor it.
// Repeated filler *within* a section is deliberate: it is what a pasted log looks like, and it is
// the content the whole-block detector is supposed to survive.
std::string section_text(int index) {
    const std::string id = std::to_string(1000 + index);
    std::string text;
    for (int repeat = 0; repeat < kFillerRepeats; ++repeat) {
        text += "The maintenance log for change " + id +
                " records every step in order, and each entry lists the operator, the timestamp, "
                "and a short justification. ";
    }
    return text;
}

std::string joined_sections(const std::vector<std::pair<int, int>>& deleted) {
    std::string text;
    for (int section = 0; section < kSections; ++section) {
        const bool removed = std::any_of(deleted.begin(), deleted.end(),
                                         [section](const std::pair<int, int>& range) {
                                             return section >= range.first && section < range.second;
                                         });
        if (!removed) { text += section_text(section); }
    }
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

ninfer::KvCacheStorage kv_storage_from_env() {
    const char* value = std::getenv("NINFER_MULTISPAN_KV");
    if (value == nullptr || *value == '\0') { return ninfer::KvCacheStorage::RK4V4E8; }
    const std::string name(value);
    if (name == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (name == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (name == "rk8v4") { return ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    if (name == "rk4v4") { return ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64; }
    if (name == "rk2v4-e8") { return ninfer::KvCacheStorage::RK2V4E8; }
    return ninfer::KvCacheStorage::RK4V4E8;
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    // A ~32k prompt plus its question, the reported shape. `explicit_capacity` tracks it so the KV
    // ceiling is fixed rather than discovered, which keeps the three cases comparable.
    constexpr std::uint32_t kContext = 40960;
    options.max_context               = kContext;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(kContext);
    options.prefill_chunk             = 1024;
    options.max_concurrency           = 1;
    options.kv_cache                  = kv_storage_from_env();
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

ninfer::RequestOptions request_options(bool reuse) {
    ninfer::RequestOptions result;
    result.execution.requested_output_tokens    = 8;
    result.execution.sampling.temperature       = 0.0F;
    result.execution.allow_prefix_reuse         = reuse;
    result.execution.enable_suffix_cache_reuse  = reuse;
    result.execution.scr_mode                   = ninfer::ScrRelocationMode::Segmented;
    result.execution.suffix_reuse_max_spans     = 6;
    result.stop.include_model_defaults          = false;
    return result;
}

struct Case {
    const char* name;
    std::vector<std::pair<int, int>> deleted;
};

double first_logprob_delta(const ninfer::GenerationResult& cold,
                           const ninfer::GenerationResult& reused) {
    if (cold.generated_logprobs.empty() || reused.generated_logprobs.empty()) { return -1.0; }
    const float left  = cold.generated_logprobs.front();
    const float right = reused.generated_logprobs.front();
    if (!std::isfinite(left) || !std::isfinite(right)) { return -1.0; }
    return std::fabs(left - right);
}

// Counted against the *cold* length on purpose. The reported failure stopped the reused leg after
// two tokens; a denominator of `min(...)` would score that 2/2 and read as a perfect match, which is
// the opposite of what happened. A reused leg that stops short must score short.
std::size_t agreement(const ninfer::GenerationResult& cold,
                      const ninfer::GenerationResult& reused) {
    std::size_t agreed = 0;
    while (agreed < cold.generated_token_ids.size() &&
           agreed < reused.generated_token_ids.size() &&
           cold.generated_token_ids[agreed] == reused.generated_token_ids[agreed]) {
        ++agreed;
    }
    return agreed;
}

void print_tokens(const char* label, const ninfer::GenerationResult& result) {
    std::cout << "  " << label << " (n=" << result.generated_token_ids.size() << ", finish="
              << static_cast<int>(result.finish_reason) << "):";
    for (const ninfer::TokenId token : result.generated_token_ids) { std::cout << ' ' << token; }
    std::cout << '\n';
}

struct Outcome {
    bool valid = true;
    std::string problem;
};

Outcome run_case(ninfer::Engine& engine, const Case& test) {
    Outcome outcome;

    const std::string full_text    = joined_sections({});
    const std::string kept_text    = joined_sections(test.deleted);
    const std::string edited_text  = kept_text + "Which change was rolled back, and why? ";

    const std::uint32_t full_tokens   = engine.count_tokens(chat_prompt(full_text));
    const std::uint32_t kept_tokens   = engine.count_tokens(chat_prompt(kept_text));
    const std::uint32_t edited_tokens = engine.count_tokens(chat_prompt(edited_text));

    // Cold first, while nothing related is resident. Then leave the unedited prompt resident, then
    // ask for the edited one with reuse, exactly as the single-span case does.
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare(chat_prompt(edited_text)), request_options(false));
    const ninfer::GenerationResult resident =
        engine.generate(engine.prepare(chat_prompt(full_text)), request_options(false));
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(chat_prompt(edited_text)), request_options(true));

    std::cout << "case " << test.name << ": full=" << full_tokens << " kept=" << kept_tokens
              << " edited=" << edited_tokens << " deleted=" << (full_tokens - kept_tokens)
              << " spans_deleted=" << test.deleted.size()
              << " path=" << reuse_path_name(reused.prefix_reuse_path)
              << " reused=" << reused.reused_prompt_tokens
              << " relocated=" << reused.relocated_prompt_tokens
              << " computed=" << reused.computed_prefill_tokens
              << " agreed=" << agreement(cold, reused) << "/" << cold.generated_token_ids.size()
              << " first_logprob_delta=" << first_logprob_delta(cold, reused) << '\n';
    std::cout << "  mtp: drafted=" << reused.speculative.drafted_tokens
              << " accepted=" << reused.speculative.accepted_tokens << '\n';
    print_tokens("cold  ", cold);
    print_tokens("reused", reused);

    if (cold.generated_token_ids.empty()) {
        outcome.valid   = false;
        outcome.problem = "the cold leg produced no tokens";
    } else if (resident.generated_token_ids.empty()) {
        outcome.valid   = false;
        outcome.problem = "the resident leg produced no tokens";
    } else if (reused.prefix_reuse_path != ninfer::PrefixReusePath::SuffixRelocation ||
               reused.reused_prompt_tokens == 0 || reused.relocated_prompt_tokens == 0) {
        // A silent fall back to a cold prefill would make every number true for the wrong reason:
        // the plan is the only thing tying the comparison to SCR at all.
        outcome.valid   = false;
        outcome.problem = "the reuse leg did not relocate (path=" +
                          std::string(reuse_path_name(reused.prefix_reuse_path)) + ")";
    }

    // Optional numeric gates, off unless asked for, so the cases can be run as a measurement first
    // and tightened to a regression budget once the numbers are known.
    if (outcome.valid) {
        const char* min_agreed = std::getenv("NINFER_MULTISPAN_MIN_AGREED");
        const char* first_nat  = std::getenv("NINFER_MULTISPAN_FIRST_NAT");
        if (min_agreed != nullptr && *min_agreed != '\0') {
            const std::size_t budget = static_cast<std::size_t>(std::stoul(min_agreed));
            if (agreement(cold, reused) < budget) {
                outcome.valid   = false;
                outcome.problem = "greedy agreement fell below the requested budget";
            }
        }
        if (first_nat != nullptr && *first_nat != '\0') {
            const double budget = std::stod(first_nat);
            const double delta  = first_logprob_delta(cold, reused);
            if (delta < 0.0 || delta > budget) {
                outcome.valid   = false;
                outcome.problem = "the first-token logprob moved past the requested budget";
            }
        }
    }
    return outcome;
}

// `NINFER_MULTISPAN_CASE` selects among the cases below, by letter or by name, so the one-span case
// can be run on its own without paying for the multi-span ones. Unset runs every case.
bool case_selected(const Case& test) {
    const char* requested = std::getenv("NINFER_MULTISPAN_CASE");
    if (requested == nullptr || *requested == '\0') { return true; }
    const std::string wanted(requested);
    return wanted == test.name || (wanted.size() == 1 && test.name[0] == wanted[0]);
}

int exercise(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));

    const std::vector<Case> cases{
        // One deep cut: the whole deleted volume in a single surviving run.
        Case{.name = "A-one-span-large", .deleted = {{10, 23}}},
        // Three shallow cuts: multi-span, small volume. The planner's shape without the state.
        Case{.name = "B-three-span-small", .deleted = {{10, 11}, {20, 21}, {30, 31}}},
        // The reported shape: three deep cuts, ~10k tokens gone.
        Case{.name = "C-three-span-large", .deleted = {{8, 12}, {18, 22}, {28, 32}}},
    };

    int failures = 0;
    int selected = 0;
    for (const Case& test : cases) {
        if (!case_selected(test)) { continue; }
        ++selected;
        const Outcome outcome = run_case(engine, test);
        if (!outcome.valid) {
            std::cerr << "multi-span edit: case " << test.name << " is not a valid measurement: "
                      << outcome.problem << '\n';
            ++failures;
        }
    }
    if (selected == 0) {
        std::cerr << "multi-span edit: NINFER_MULTISPAN_CASE=" << std::getenv("NINFER_MULTISPAN_CASE")
                  << " matched no case (A-one-span-large, B-three-span-small, C-three-span-large)\n";
        return 1;
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    // The live failure ran with the copied/shifted route enabled; make that the default for this
    // binary so the case reproduces it, without clobbering an explicit NINFER_SCR_SOURCE_SHIFT the
    // caller already exported. NINFER_MULTISPAN_ALIGNED=1 forces the aligned route. The two routes
    // are separate processes because the flag is latched on first use.
    const char* aligned = std::getenv("NINFER_MULTISPAN_ALIGNED");
    const char* existing_shift = std::getenv("NINFER_SCR_SOURCE_SHIFT");
    const char* shift_value = existing_shift != nullptr && *existing_shift != '\0'
                                  ? existing_shift
                                  : ((aligned != nullptr && *aligned != '\0' && *aligned != '0') ? "0" : "1");
#if defined(_WIN32)
    _putenv_s("NINFER_SCR_SOURCE_SHIFT", shift_value);
#else
    setenv("NINFER_SCR_SOURCE_SHIFT", shift_value, 1);
#endif

    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77; // CTest SKIP_RETURN_CODE
    }
    std::cout << "multi-span edit: source_shift="
              << (std::getenv("NINFER_SCR_SOURCE_SHIFT")[0] == '0' ? "off" : "on")
              << " kv=" << static_cast<int>(kv_storage_from_env()) << '\n';
    if (const int result = exercise(artifact); result != 0) { return result; }
    std::cout << "ok\n";
    return 0;
}

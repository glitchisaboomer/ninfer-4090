#pragma once

// Phase 5.5 of docs/SCR_ROADMAP.md scores Suffix Cache Reuse by comparing two runs of one prompt.
// When those runs are served by ninfer-serve rather than driven in-process, the only channel that
// carries per-token logprobs is the request JSONL log (`--request-log-jsonl`); this module turns
// that log into the target's own scoring input.
//
// It reads the schema that src/serve/request_log.cpp writes and nothing else, so the wire protocols
// stay unchanged and no second copy of the roadmap's thresholds exists: the arithmetic stays in
// qwen3_6::detail::evaluate_scr_quality.
//
// It also owns how a measurement run asks ninfer-serve to produce that log, because the argument
// contract between the two is otherwise only checked by spawning a server and waiting for it to
// start: ninfer-serve takes the artifact as its first *positional* argument, so an invocation that
// passes --model is rejected before anything is loaded, and the harness fails at health-check time
// with a 900-second timeout instead of at argument-parse time.

#include "ninfer/types.h"
#include "targets/qwen3_6/impl/runtime/scr_quality.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// One completed request as the log recorded it. Only the fields the comparison needs.
struct LoggedCompletion {
    std::uint64_t request_id = 0;
    std::int32_t prompt_tokens             = 0;
    std::uint32_t prefix_cache_hit_tokens  = 0;
    std::uint32_t prefix_relocated_tokens  = 0;
    std::string prefix_reuse_path;
    std::vector<ninfer::TokenId> generated_token_ids;
    // NaN where the log recorded null, which is how a path that measures no logprob reports.
    std::vector<float> generated_logprobs;
};

// Parses the `request_done` records of a request JSONL log, in file order, ignoring every other
// event kind. Records written by a schema this build does not know, a record whose `result.generated`
// is absent or malformed, or a log whose token and logprob arrays disagree in length all throw:
// a quality run that silently scored a partial record would report a number nobody can trust.
[[nodiscard]] std::vector<LoggedCompletion> parse_completion_log(std::string_view jsonl);

// Builds one comparison sample per position. Both sides must describe the same prompts in the same
// order -- `scr` is the run with Suffix Cache Reuse enabled, `baseline` the full-prefill run -- and
// every SCR record must report relocated tokens, otherwise the two sides agree because neither of
// them reused anything and the comparison would pass while measuring nothing.
[[nodiscard]] std::vector<targets::qwen3_6::detail::ScrQualitySample>
pair_scr_quality_samples(std::span<const LoggedCompletion> scr,
                         std::span<const LoggedCompletion> baseline);

// Scores every paired prompt and returns the target's own report JSON, one line per prompt plus an
// aggregate. Empty when either side is empty.
[[nodiscard]] std::string format_scr_quality_report(
    std::span<const LoggedCompletion> scr, std::span<const LoggedCompletion> baseline,
    const targets::qwen3_6::detail::ScrQualityThresholds& thresholds);

// Builds the argv for one measurement phase, excluding argv[0].
//
// The baseline phase disables prefix reuse as well as Suffix Cache Reuse, because the roadmap scores
// against the full-prefill path and two servers whose prefix caches differ would otherwise agree for
// reasons that have nothing to do with relocation. `extra` is appended verbatim so a caller can add
// flags without this function having to know about them; it is appended last so it cannot displace
// the flags that make the run mean what it claims.
[[nodiscard]] std::vector<std::string> build_scr_quality_server_arguments(
    const std::string& artifact, int port, std::uint32_t max_context, const std::string& log_path,
    bool scr, std::span<const std::string> extra = {});

} // namespace ninfer::serve
